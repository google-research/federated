# Copyright 2023, Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for training_loop."""
import collections
import os
import shutil
import tempfile
from unittest import mock

import haiku as hk
import jax
from jax import numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import tensorflow_federated as tff
import tree

from multi_epoch_dp_matrix_factorization.dp_ftrl.centralized import gradient_processors
from multi_epoch_dp_matrix_factorization.dp_ftrl.centralized import models
from multi_epoch_dp_matrix_factorization.dp_ftrl.centralized import training_loop


def _build_tree_aggregator(
    hk_model: hk.Transformed, example_batch, n_microbatches=10
) -> tff.templates.AggregationProcess:
  rng = jax.random.PRNGKey(2)
  # Since iteration order is guaranteed in python3.X+, this conversion is safe.
  # Though it does make me concerned with the way we will pass data in to the
  # aggregator 'later'. This will probably need to go in the interface impl.
  model_params_struct = tff.structure.from_container(
      hk_model.init(rng, example_batch), recursive=True
  )
  model_params_odict = tff.structure.to_odict(
      model_params_struct, recursive=True
  )

  model_params_tff_type = tff.types.type_from_tensors(model_params_odict)
  model_specs = tff.types.type_to_tf_tensor_specs(model_params_tff_type)
  return tff.aggregators.DifferentiallyPrivateFactory.tree_aggregation(
      noise_multiplier=0.0,
      clients_per_round=n_microbatches,
      l2_norm_clip=1.0,
      record_specs=model_specs,
      noise_seed=2,
      use_efficient=True,
  ).create(model_params_tff_type)


def _return_batch(batch_size):
  # Existing code is all programmed to be channels-first.
  images = jnp.ones(shape=[batch_size, 3, 32, 32], dtype=jnp.float32)
  return images


def _return_labels(batch_size):
  labels = jnp.ones(shape=[batch_size], dtype=jnp.int64)
  return labels


def _make_train_data_iterator(n_batches, batch_size):
  n_batches_processed = 0
  while n_batches_processed < n_batches:
    yield _return_batch(batch_size), _return_labels(batch_size)
    n_batches_processed += 1


def _make_data_sequence(n_batches, batch_size):
  return [
      (_return_batch(batch_size), _return_labels(batch_size))
      for _ in range(n_batches)
  ]


def _return_0_loss(x, y):
  del x, y  # Unused
  return 0.0


class SaveAndLoadTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._tmpdir = tempfile.mkdtemp()
    batch_size = 10
    rng = jax.random.PRNGKey(2)
    hk_model = models.build_vgg_model(
        n_classes=10, dense_size=128, activation_fn=jax.nn.relu
    )
    self._default_params = hk_model.init(rng, _return_batch(batch_size))
    aggregator = _build_tree_aggregator(
        hk_model, _return_batch(batch_size), batch_size
    )
    grad_processor = gradient_processors.DPAggregatorBackedGradientProcessor(
        aggregator, l2_norm_clip=1.0
    )
    self._default_grad_processor_state = grad_processor.init()
    opt = optax.sgd(1.0)
    self._default_opt_state = opt.init(self._default_params)

  def tearDown(self):
    shutil.rmtree(self._tmpdir)
    super().tearDown()

  def test_save_model_writes_file(self):
    training_loop._save_checkpoint(
        epoch=0,
        model_params=self._default_params,
        batch_grad_processor_state=self._default_grad_processor_state,
        optimizer_state=self._default_opt_state,
        checkpoint_dir=self._tmpdir,
    )
    files_in_tempdir = os.listdir(self._tmpdir)
    self.assertLen(files_in_tempdir, 1)

  def test_save_load_round_trip_equality(self):
    training_loop._save_checkpoint(
        epoch=0,
        model_params=self._default_params,
        batch_grad_processor_state=self._default_grad_processor_state,
        optimizer_state=self._default_opt_state,
        checkpoint_dir=self._tmpdir,
    )
    epoch, loaded_model, loaded_grad_processor, loaded_opt_state = (
        training_loop._load_most_recent_checkpoint(
            initial_model_params=self._default_params,
            batch_grad_processor_state=self._default_grad_processor_state,
            optimizer_state=self._default_opt_state,
            checkpoint_dir=self._tmpdir,
        )
    )
    self.assertEqual(epoch, 0)
    self.assertAllClose(self._default_params, loaded_model)
    # These grad processor states have namedtuple elements, which assertAllClose
    # can't handle.
    self.assertAllClose(
        tree.flatten(self._default_grad_processor_state),
        tree.flatten(loaded_grad_processor),
    )
    self.assertAllClose(self._default_opt_state, loaded_opt_state)

  def test_save_multiple_times_load_returns_last_saved_model(self):
    ones_like_params = tree.map_structure(jnp.ones_like, self._default_params)
    ones_like_grad_state = tree.map_structure(
        jnp.ones_like, self._default_grad_processor_state
    )
    ones_like_opt_state = tree.map_structure(
        jnp.ones_like, self._default_opt_state
    )
    training_loop._save_checkpoint(
        epoch=0,
        model_params=self._default_params,
        batch_grad_processor_state=self._default_grad_processor_state,
        optimizer_state=self._default_opt_state,
        checkpoint_dir=self._tmpdir,
    )
    training_loop._save_checkpoint(
        epoch=1,
        model_params=ones_like_params,
        batch_grad_processor_state=ones_like_grad_state,
        optimizer_state=ones_like_opt_state,
        checkpoint_dir=self._tmpdir,
    )
    epoch, loaded_model, loaded_grad_processor, loaded_opt_state = (
        training_loop._load_most_recent_checkpoint(
            initial_model_params=self._default_params,
            batch_grad_processor_state=self._default_grad_processor_state,
            optimizer_state=self._default_opt_state,
            checkpoint_dir=self._tmpdir,
        )
    )
    # We should load the ones_like version, as it was saved at a later epoch.
    self.assertEqual(epoch, 1)
    self.assertAllClose(ones_like_params, loaded_model)
    self.assertAllClose(
        tree.flatten(ones_like_grad_state), tree.flatten(loaded_grad_processor)
    )
    self.assertAllClose(ones_like_opt_state, loaded_opt_state)


class TrainOneEpochTest(tf.test.TestCase):

  def test_cifar10_model_trains_one_epoch_without_moving_zero_loss(self):
    batch_size = 10
    rng = jax.random.PRNGKey(2)
    hk_model = models.build_vgg_model(
        n_classes=10, dense_size=128, activation_fn=jax.nn.relu
    )
    initial_params = hk_model.init(rng, _return_batch(batch_size))
    aggregator = _build_tree_aggregator(
        hk_model, _return_batch(batch_size), batch_size
    )
    grad_processor = gradient_processors.DPAggregatorBackedGradientProcessor(
        aggregator, l2_norm_clip=1.0
    )
    grad_processor_state = grad_processor.init()
    train_data_iterator = _make_train_data_iterator(10, batch_size)
    opt = optax.sgd(1.0)
    opt_state = opt.init(initial_params)
    resulting_params, grad_processor_state, opt_state = (
        training_loop.train_one_epoch(
            train_data=train_data_iterator,
            model_fn=hk_model.apply,
            initial_model_params=initial_params,
            loss_fn=_return_0_loss,
            batch_grad_processor=grad_processor,
            batch_grad_processor_state=grad_processor_state,
            post_dp_optimizer=opt,
            optimizer_state=opt_state,
            rng=rng,
            metrics_managers=[tff.program.LoggingReleaseManager()],
        )
    )
    # These should be identical, since we always just return 0 loss.
    self.assertAllClose(initial_params, resulting_params)

  def test_cifar10_model_trains_one_epoch_crossentropy_loss(self):
    batch_size = 10
    rng = jax.random.PRNGKey(2)
    hk_model = models.build_vgg_model(
        n_classes=10, dense_size=128, activation_fn=jax.nn.relu
    )
    initial_params = hk_model.init(rng, _return_batch(batch_size))
    aggregator = _build_tree_aggregator(
        hk_model, _return_batch(batch_size), batch_size
    )
    grad_processor = gradient_processors.DPAggregatorBackedGradientProcessor(
        aggregator, l2_norm_clip=1.0
    )
    grad_processor_state = grad_processor.init()
    train_data_iterator = _make_train_data_iterator(10, batch_size)
    opt = optax.sgd(0.01)
    opt_state = opt.init(initial_params)
    resulting_params, grad_processor_state, opt_state = (
        training_loop.train_one_epoch(
            train_data=train_data_iterator,
            model_fn=hk_model.apply,
            initial_model_params=initial_params,
            loss_fn=optax.softmax_cross_entropy_with_integer_labels,
            batch_grad_processor=grad_processor,
            batch_grad_processor_state=grad_processor_state,
            post_dp_optimizer=opt,
            optimizer_state=opt_state,
            rng=rng,
            metrics_managers=[tff.program.LoggingReleaseManager()],
        )
    )
    # These should be have changed, since we are using a real loss function now.
    self.assertNotAllClose(initial_params, resulting_params)

  def test_cifar10_evaluate_model(self):
    batch_size = 1
    rng = jax.random.PRNGKey(2)
    hk_model = models.build_vgg_model(
        n_classes=10, dense_size=128, activation_fn=jax.nn.relu
    )
    initial_params = hk_model.init(rng, _return_batch(batch_size))
    eval_data_iterator = _make_train_data_iterator(10, batch_size)
    metrics_result = training_loop.evaluate_model(
        model_fn=hk_model.apply,
        model_params=initial_params,
        eval_fns=training_loop._EVAL_FNS,
        data=eval_data_iterator,
        rng=rng,
        epoch=1,
        metrics_managers=[tff.program.LoggingReleaseManager()],
    )
    self.assertAllClose(
        metrics_result,
        collections.OrderedDict([('eval/accuracy', np.zeros(shape=[]))]),
    )


class TrainTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._tmpdir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self._tmpdir)
    super().tearDown()

  @mock.patch.object(tff.program.LoggingReleaseManager, 'release')
  def test_cifar10_model_trains_two_epochs_with_zero_loss(
      self, mock_release_method
  ):
    batch_size = 10
    rng = jax.random.PRNGKey(2)
    hk_model = models.build_vgg_model(
        n_classes=10, dense_size=128, activation_fn=jax.nn.relu
    )
    aggregator = _build_tree_aggregator(
        hk_model, _return_batch(batch_size), batch_size
    )
    grad_processor = gradient_processors.DPAggregatorBackedGradientProcessor(
        aggregator, l2_norm_clip=1.0
    )
    opt = optax.sgd(1.0)
    train_data = _make_data_sequence(10, batch_size)
    eval_data = _make_data_sequence(5, batch_size)
    test_data = _make_data_sequence(5, batch_size)

    training_loop.train(
        train_data=train_data,
        eval_data=eval_data,
        test_data=test_data,
        model=hk_model,
        loss_fn=_return_0_loss,
        batch_grad_processor=grad_processor,
        post_dp_optimizer=opt,
        rng=rng,
        num_epochs=2,
        root_output_dir=self._tmpdir,
        run_name='test_experiment',
        hparams_dict={},
    )
    # 20 rounds, 2 evals and 1 test.
    self.assertLen(mock_release_method.call_args_list, 20 + 2 + 1)
    self.assertEqual(
        mock_release_method.call_args_list[-1],
        mock.call(
            value=collections.OrderedDict([('test/accuracy', np.array(0.0))]),
            type_signature=tff.to_type(
                collections.OrderedDict([('test/accuracy', tf.float32)])
            ),
            key=2,
        ),
    )


if __name__ == '__main__':
  tf.test.main()
