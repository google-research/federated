# Copyright 2020, The Federated Research Authors.
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
"""End-to-end example testing Posterior Averaging against the MNIST model."""

import collections
import functools

from absl import logging
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from posterior_averaging.shared import fed_pa_schedule

_Batch = collections.namedtuple('Batch', ['x', 'y'])

_client_mixedin_schedule_fn = fed_pa_schedule.create_mixin_check_fn(
    'fixed_epochs')
_client_update_delta_fn = fed_pa_schedule.create_update_delta_fn('simple_avg')
_build_simple_fed_pa_process = functools.partial(
    fed_pa_schedule.build_fed_pa_process,
    client_update_epochs=1,
    client_mixedin_schedule_fn=_client_mixedin_schedule_fn,
    client_update_delta_fn=_client_update_delta_fn,
    mask_zeros_in_client_updates=True)


def _create_client_update_fn():
  client_single_data_pass_fn = (
      fed_pa_schedule.create_client_single_data_pass_fn())
  return functools.partial(
      fed_pa_schedule.client_update,
      num_epochs=1,
      client_mixedin_fn=_client_mixedin_schedule_fn(0),
      client_update_delta_fn=_client_update_delta_fn,
      client_single_data_pass_fn=client_single_data_pass_fn)


def _batch_fn(has_nan=False):
  batch = _Batch(
      x=np.ones([1, 784], dtype=np.float32), y=np.ones([1, 1], dtype=np.int64))
  if has_nan:
    batch[0][0, 0] = np.nan
  return batch


def _create_input_spec():
  return _Batch(
      x=tf.TensorSpec(shape=[None, 784], dtype=tf.float32),
      y=tf.TensorSpec(dtype=tf.int64, shape=[None, 1]))


def _uncompiled_model_builder():
  keras_model = tff.simulation.models.mnist.create_keras_model(
      compile_model=False)
  return tff.learning.from_keras_model(
      keras_model=keras_model,
      input_spec=_create_input_spec(),
      loss=tf.keras.losses.SparseCategoricalCrossentropy())


class ModelDeltaProcessTest(tf.test.TestCase):

  def _run_rounds(self, iterproc, federated_data, num_rounds):
    train_outputs = []
    initial_state = iterproc.initialize()
    state = initial_state
    for round_num in range(num_rounds):
      state, metrics = iterproc.next(state, federated_data)
      train_outputs.append(metrics)
      logging.info('Round %d: %s', round_num, metrics)
    return state, train_outputs, initial_state

  def test_post_avg_without_schedule_decreases_loss(self):
    federated_data = [[_batch_fn()]]

    iterproc = _build_simple_fed_pa_process(
        _uncompiled_model_builder,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        server_optimizer_fn=tf.keras.optimizers.SGD)

    _, train_outputs, _ = self._run_rounds(iterproc, federated_data, 5)
    self.assertLess(train_outputs[-1]['loss'], train_outputs[0]['loss'])

  def test_post_avg_with_custom_client_weight_fn(self):
    federated_data = [[_batch_fn()]]

    def client_weight_fn(local_outputs):
      return 1.0 / (1.0 + local_outputs['loss'][-1])

    iterproc = _build_simple_fed_pa_process(
        _uncompiled_model_builder,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        server_optimizer_fn=tf.keras.optimizers.SGD,
        client_weight_fn=client_weight_fn)

    _, train_outputs, _ = self._run_rounds(iterproc, federated_data, 5)
    self.assertLess(train_outputs[-1]['loss'], train_outputs[0]['loss'])

  def test_client_update_with_finite_delta(self):
    federated_data = [_batch_fn()]
    model = _uncompiled_model_builder()
    client_optimizer = tf.keras.optimizers.SGD(0.1)
    client_update = _create_client_update_fn()
    outputs = client_update(
        model=model,
        dataset=federated_data,
        initial_weights=fed_pa_schedule._get_weights(model),
        client_optimizer=client_optimizer)
    self.assertAllEqual(self.evaluate(outputs.client_weight), 1)
    self.assertAllEqual(
        self.evaluate(outputs.optimizer_output['num_examples']), 1)

  def test_client_update_with_non_finite_delta(self):
    federated_data = [_batch_fn(has_nan=True)]
    model = _uncompiled_model_builder()
    client_optimizer = tf.keras.optimizers.SGD(0.1)
    client_update = _create_client_update_fn()
    outputs = client_update(
        model=model,
        dataset=federated_data,
        initial_weights=fed_pa_schedule._get_weights(model),
        client_optimizer=client_optimizer)
    self.assertAllEqual(self.evaluate(outputs.client_weight), 0)

  def test_server_update_with_nan_data_is_noop(self):
    federated_data = [[_batch_fn(has_nan=True)]]

    iterproc = _build_simple_fed_pa_process(
        _uncompiled_model_builder,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        server_optimizer_fn=tf.keras.optimizers.SGD)

    state, _, initial_state = self._run_rounds(iterproc, federated_data, 1)
    self.assertAllClose(state.model.trainable, initial_state.model.trainable,
                        1e-8)
    self.assertAllClose(state.model.non_trainable,
                        initial_state.model.non_trainable, 1e-8)

  def test_server_update_with_inf_weight_is_noop(self):
    federated_data = [[_batch_fn()]]
    client_weight_fn = lambda x: np.inf

    iterproc = _build_simple_fed_pa_process(
        _uncompiled_model_builder,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        server_optimizer_fn=tf.keras.optimizers.SGD,
        client_weight_fn=client_weight_fn)

    state, _, initial_state = self._run_rounds(iterproc, federated_data, 1)
    self.assertAllClose(state.model.trainable, initial_state.model.trainable,
                        1e-8)
    self.assertAllClose(state.model.non_trainable,
                        initial_state.model.non_trainable, 1e-8)

  def test_post_avg_with_client_schedule(self):
    federated_data = [[_batch_fn()]]

    @tf.function
    def lr_schedule(x):
      return 0.1 if x < 1.5 else 0.0

    iterproc = _build_simple_fed_pa_process(
        _uncompiled_model_builder,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        client_lr=lr_schedule,
        server_optimizer_fn=tf.keras.optimizers.SGD)

    _, train_outputs, _ = self._run_rounds(iterproc, federated_data, 4)
    self.assertLess(train_outputs[1]['loss'], train_outputs[0]['loss'])
    self.assertNear(
        train_outputs[2]['loss'], train_outputs[3]['loss'], err=1e-4)

  def test_post_avg_with_server_schedule(self):
    federated_data = [[_batch_fn()]]

    @tf.function
    def lr_schedule(x):
      return 1.0 if x < 1.5 else 0.0

    iterproc = _build_simple_fed_pa_process(
        _uncompiled_model_builder,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        server_optimizer_fn=tf.keras.optimizers.SGD,
        server_lr=lr_schedule)

    _, train_outputs, _ = self._run_rounds(iterproc, federated_data, 4)
    self.assertLess(train_outputs[1]['loss'], train_outputs[0]['loss'])
    self.assertNear(
        train_outputs[2]['loss'], train_outputs[3]['loss'], err=1e-4)

  def test_post_avg_with_client_and_server_schedules(self):
    federated_data = [[_batch_fn()]]

    iterproc = _build_simple_fed_pa_process(
        _uncompiled_model_builder,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        client_lr=lambda x: 0.1 / (x + 1)**2,
        server_optimizer_fn=tf.keras.optimizers.SGD,
        server_lr=lambda x: 1.0 / (x + 1)**2)

    _, train_outputs, _ = self._run_rounds(iterproc, federated_data, 6)
    self.assertLess(train_outputs[-1]['loss'], train_outputs[0]['loss'])
    train_gap_first_half = train_outputs[0]['loss'] - train_outputs[2]['loss']
    train_gap_second_half = train_outputs[3]['loss'] - train_outputs[5]['loss']
    self.assertLess(train_gap_second_half, train_gap_first_half)

  def test_build_with_preprocess_function(self):
    test_dataset = tf.data.Dataset.range(5)
    client_datasets_type = tff.FederatedType(
        tff.SequenceType(test_dataset.element_spec), tff.CLIENTS)

    @tff.tf_computation(tff.SequenceType(test_dataset.element_spec))
    def preprocess_dataset(ds):

      def to_batch(x):
        return _Batch(
            tf.fill(dims=(784,), value=float(x) * 2.0),
            tf.expand_dims(tf.cast(x + 1, dtype=tf.int64), axis=0))

      return ds.map(to_batch).batch(2)

    iterproc = _build_simple_fed_pa_process(
        _uncompiled_model_builder,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        server_optimizer_fn=tf.keras.optimizers.SGD)
    iterproc = tff.simulation.compose_dataset_computation_with_iterative_process(
        preprocess_dataset, iterproc)

    with tf.Graph().as_default():
      test_model_for_types = _uncompiled_model_builder()

    server_state_type = tff.FederatedType(
        fed_pa_schedule.ServerState(
            model=tff.framework.type_from_tensors(
                tff.learning.ModelWeights(
                    test_model_for_types.trainable_variables,
                    test_model_for_types.non_trainable_variables)),
            optimizer_state=(tf.int64,),
            round_num=tf.float32), tff.SERVER)
    metrics_type = tff.FederatedType(
        tff.StructType([('loss', tf.float32),
                        ('model_delta_zeros_percent', tf.float32),
                        ('model_delta_correction_l2_norm', tf.float32)]),
        tff.SERVER)

    expected_parameter_type = collections.OrderedDict(
        server_state=server_state_type,
        federated_dataset=client_datasets_type,
    )
    expected_result_type = (server_state_type, metrics_type)

    expected_type = tff.FunctionType(
        parameter=expected_parameter_type, result=expected_result_type)
    self.assertTrue(
        iterproc.next.type_signature.is_equivalent_to(expected_type),
        msg='{s}\n!={t}'.format(
            s=iterproc.next.type_signature, t=expected_type))

  def test_execute_with_preprocess_function(self):
    test_dataset = tf.data.Dataset.range(1)

    @tff.tf_computation(tff.SequenceType(test_dataset.element_spec))
    def preprocess_dataset(ds):

      def to_example(x):
        del x  # Unused.
        return _Batch(
            x=np.ones([784], dtype=np.float32), y=np.ones([1], dtype=np.int64))

      return ds.map(to_example).batch(1)

    iterproc = _build_simple_fed_pa_process(
        _uncompiled_model_builder,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        server_optimizer_fn=tf.keras.optimizers.SGD)
    iterproc = tff.simulation.compose_dataset_computation_with_iterative_process(
        preprocess_dataset, iterproc)

    _, train_outputs, _ = self._run_rounds(iterproc, [test_dataset], 6)
    self.assertLess(train_outputs[-1]['loss'], train_outputs[0]['loss'])
    train_gap_first_half = train_outputs[0]['loss'] - train_outputs[2]['loss']
    train_gap_second_half = train_outputs[3]['loss'] - train_outputs[5]['loss']
    self.assertLess(train_gap_second_half, train_gap_first_half)

  def test_get_model_weights(self):
    federated_data = [[_batch_fn()]]

    iterative_process = _build_simple_fed_pa_process(
        _uncompiled_model_builder,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        server_optimizer_fn=tf.keras.optimizers.SGD)
    state = iterative_process.initialize()

    self.assertIsInstance(
        iterative_process.get_model_weights(state), tff.learning.ModelWeights)
    self.assertAllClose(state.model.trainable,
                        iterative_process.get_model_weights(state).trainable)

    for _ in range(3):
      state, _ = iterative_process.next(state, federated_data)
      self.assertIsInstance(
          iterative_process.get_model_weights(state), tff.learning.ModelWeights)
      self.assertAllClose(state.model.trainable,
                          iterative_process.get_model_weights(state).trainable)


class FederatedMeanMaskedTest(tf.test.TestCase):

  def test_federated_mean_masked(self):

    value_type = tff.StructType([('a', tff.TensorType(tf.float32, shape=[3])),
                                 ('b', tff.TensorType(tf.float32, shape=[2,
                                                                         1]))])
    weight_type = tff.TensorType(tf.float32)

    federated_mean_masked_fn = fed_pa_schedule.build_federated_mean_masked(
        value_type, weight_type)

    # Check type signature.
    expected_type = tff.FunctionType(
        parameter=collections.OrderedDict(
            value=tff.FederatedType(value_type, tff.CLIENTS),
            weight=tff.FederatedType(weight_type, tff.CLIENTS)),
        result=tff.FederatedType(value_type, tff.SERVER))
    self.assertTrue(
        federated_mean_masked_fn.type_signature.is_equivalent_to(expected_type),
        msg='{s}\n!={t}'.format(
            s=federated_mean_masked_fn.type_signature, t=expected_type))

    # Check correctness of zero masking in the mean.
    values = [
        collections.OrderedDict(
            a=tf.constant([0.0, 1.0, 1.0]), b=tf.constant([[1.0], [2.0]])),
        collections.OrderedDict(
            a=tf.constant([1.0, 3.0, 0.0]), b=tf.constant([[3.0], [0.0]]))
    ]
    weights = [tf.constant(1.0), tf.constant(3.0)]
    output = federated_mean_masked_fn(values, weights)
    expected_output = collections.OrderedDict(
        a=tf.constant([1.0, 2.5, 1.0]), b=tf.constant([[2.5], [2.0]]))
    self.assertAllClose(output, expected_output)


class MixinCheckTest(tf.test.TestCase):

  def test_create_mixin_check_fn(self):
    mixin_check_scheme = 'fixed_epochs'
    mixin_check_start_round = 5
    num_mixin_epochs = 10

    mixin_check_schedule_fn = fed_pa_schedule.create_mixin_check_fn(
        mixin_check_scheme,
        start_round=mixin_check_start_round,
        num_mixin_epochs=num_mixin_epochs)
    mixin_check_fn = mixin_check_schedule_fn(round_num=5)

    for epoch in range(5, 15):
      result1 = mixin_check_fn(epoch)
      result2 = fed_pa_schedule._mixin_check_fixed_epochs_fn(
          epoch, num_mixin_epochs=num_mixin_epochs)
      self.assertEqual(result1, result2)

    mixin_check_schedule_fn = fed_pa_schedule.create_mixin_check_fn(
        mixin_check_scheme,
        start_round=mixin_check_start_round,
        num_mixin_epochs=num_mixin_epochs)
    mixin_check_fn = mixin_check_schedule_fn(round_num=0)

    for epoch in range(10, 15):
      result1 = mixin_check_fn(epoch)
      result2 = fed_pa_schedule._mixin_check_fixed_epochs_fn(
          epoch, num_mixin_epochs=num_mixin_epochs)
      self.assertNotEqual(result1, result2)

  def test_mixin_check_schedule(self):
    mixin_check_scheme = 'fixed_epochs'
    mixin_check_start_round = 10
    round_early, round_late = 1, 10

    mixin_check_schedule_fn = fed_pa_schedule.create_mixin_check_fn(
        mixin_check_scheme, start_round=mixin_check_start_round)

    mixin_check_fn = mixin_check_schedule_fn(round_num=round_early)
    self.assertFalse(mixin_check_fn(epoch=0))

    mixin_check_fn = mixin_check_schedule_fn(round_num=round_late)
    self.assertTrue(mixin_check_fn(epoch=0))

  def test_fixed_epochs_mixin_check(self):
    mixin_check_scheme = 'fixed_epochs'
    num_mixin_epochs = 10
    epoch_early, epoch_late = 1, 10

    mixin_check_schedule_fn = fed_pa_schedule.create_mixin_check_fn(
        mixin_check_scheme, num_mixin_epochs=num_mixin_epochs)

    mixin_check_fn = mixin_check_schedule_fn(0)
    self.assertFalse(mixin_check_fn(epoch=epoch_early))
    self.assertTrue(mixin_check_fn(epoch=epoch_late))


class UpdateDeltaTest(parameterized.TestCase, tf.test.TestCase):

  def test_delta_update_output(self):
    federated_data = [_batch_fn()]
    model = _uncompiled_model_builder()
    optimizer = tf.keras.optimizers.SGD(0.1)
    data_pass_fn = fed_pa_schedule.create_client_single_data_pass_fn()

    data_pass_outputs = data_pass_fn(model, federated_data, optimizer)
    initial_weights = fed_pa_schedule._get_weights(model)
    initial_updates = (
        fed_pa_schedule.DeltaUpdateOutput.from_weights(
            initial_weights=initial_weights.trainable,
            updated_weights=data_pass_outputs.model_weights_trainable_sample))
    self.assertIsInstance(initial_updates, fed_pa_schedule.DeltaUpdateOutput)
    self.assertEqual(initial_updates.num_samples, 0.)
    self.assertAllClose(initial_updates.weights_sample_mean,
                        data_pass_outputs.model_weights_trainable_sample)

  @parameterized.named_parameters([
      {
          'testcase_name': 'simple',
          'update_scheme': 'simple_avg'
      },
      {
          'testcase_name': 'posterior',
          'update_scheme': 'posterior_avg'
      },
  ])
  def test_create_update_delta_fn(self, update_scheme):
    update_delta_fn = fed_pa_schedule.create_update_delta_fn(update_scheme)

    federated_data = [_batch_fn()]
    model = _uncompiled_model_builder()
    optimizer = tf.keras.optimizers.SGD(0.1)
    data_pass_fn = fed_pa_schedule.create_client_single_data_pass_fn()
    data_pass_outputs = data_pass_fn(model, federated_data, optimizer)
    initial_weights = fed_pa_schedule._get_weights(model)

    @tf.function
    def _run_update_delta(mixedin):
      # We have to wrap creation of `DeltaUpdateOutput` in a `tf.function` here
      # because of the issues passing `tf.TensorArray`s between TF modes.
      initial_updates = (
          fed_pa_schedule.DeltaUpdateOutput.from_weights(
              initial_weights=initial_weights.trainable,
              updated_weights=data_pass_outputs.model_weights_trainable_sample))
      updates = update_delta_fn(
          mixedin=mixedin,
          initial_weights=initial_weights,
          data_pass_outputs=data_pass_outputs,
          previous_updates=initial_updates)
      return updates

    # Test delta update when not mixed-in.
    updates = _run_update_delta(mixedin=False)
    expected_weights_delta = tf.nest.map_structure(
        lambda a, b: a - b, data_pass_outputs.model_weights_trainable,
        initial_weights.trainable)
    self.assertIsInstance(updates, fed_pa_schedule.DeltaUpdateOutput)
    self.assertEqual(updates.num_samples, 0.)
    self.assertAllClose(updates.weights_delta, expected_weights_delta)
    self.assertAllClose(updates.weights_sample_mean,
                        data_pass_outputs.model_weights_trainable_sample)

    # Test delta update after the first sample.
    updates = _run_update_delta(mixedin=True)
    expected_weights_delta = tf.nest.map_structure(
        lambda a, b: a - b, data_pass_outputs.model_weights_trainable_sample,
        initial_weights.trainable)
    self.assertIsInstance(updates, fed_pa_schedule.DeltaUpdateOutput)
    self.assertEqual(updates.num_samples, 1.)
    self.assertAllClose(updates.weights_delta, expected_weights_delta)
    self.assertAllClose(updates.weights_sample_mean,
                        data_pass_outputs.model_weights_trainable_sample)

  @parameterized.named_parameters([
      {
          'testcase_name': '5_updates',
          'num_updates': 5
      },
      {
          'testcase_name': '10_updates',
          'num_updates': 10
      },
      {
          'testcase_name': '20_updates',
          'num_updates': 20
      },
  ])
  def test_update_delta_simple_avg(self, num_updates):
    # Generate initial weights and weight samples.
    initial_weights = (tf.constant([0., 1., 2., 3.], dtype=tf.float32),
                       tf.constant([[0., 1.], [2., 3.]], dtype=tf.float32))
    weights_samples = []
    previous_weights = initial_weights
    for _ in range(num_updates):
      weights_sample = tf.nest.map_structure(
          lambda x: x + tf.random.normal(x.shape), previous_weights)
      weights_samples.append(weights_sample)
      previous_weights = weights_sample

    @tf.function
    def _run_update_delta():
      # We have to wrap creation of `DeltaUpdateOutput` in a `tf.function` here
      # because of the issues passing `tf.TensorArray`s between TF modes.
      updates = fed_pa_schedule.DeltaUpdateOutput.from_weights(
          initial_weights=initial_weights,
          updated_weights=weights_samples[0],
          num_samples=1)
      for weights_sample in weights_samples[1:]:
        outputs = fed_pa_schedule.DataPassOutput(
            loss=None,
            num_examples=None,
            model_weights_trainable=weights_sample,
            model_weights_trainable_sample=weights_sample)
        updates = fed_pa_schedule._update_delta_simple_mean(outputs, updates)
      return updates

    # Compute updates.
    updates = _run_update_delta()
    self.assertEqual(updates.num_samples, num_updates)

    # Compute expected outputs as the mean of weight samples.
    expected_weights_sample_sum = tf.nest.map_structure(tf.zeros_like,
                                                        initial_weights)
    for weights_sample in weights_samples:
      expected_weights_sample_sum = tf.nest.map_structure(
          lambda a, b: a + b, expected_weights_sample_sum, weights_sample)
    expected_weights_sample_mean = tf.nest.map_structure(
        lambda x: x / num_updates, expected_weights_sample_sum)
    expected_weights_delta = tf.nest.map_structure(
        tf.subtract, expected_weights_sample_mean, initial_weights)

    self.assertAllClose(updates.weights_sample_mean,
                        expected_weights_sample_mean)
    self.assertAllClose(updates.weights_delta, expected_weights_delta)

  def struct_to_tensor(self, struct):
    struct_flat = tf.nest.map_structure(lambda x: tf.reshape(x, [-1]), struct)
    tensor = tf.concat(struct_flat, axis=0)
    return tensor

  @parameterized.named_parameters([
      {
          'testcase_name': '1_update',
          'num_updates': 1,
          'rho': 1.0
      },
      {
          'testcase_name': '2_updates-1',
          'num_updates': 2,
          'rho': 1.0
      },
      {
          'testcase_name': '2_updates-2',
          'num_updates': 2,
          'rho': 5.0
      },
      {
          'testcase_name': '5_updates-1',
          'num_updates': 5,
          'rho': 1.0
      },
      {
          'testcase_name': '5_updates-2',
          'num_updates': 5,
          'rho': 5.0
      },
  ])
  def test_update_delta_posterior_avg(self, num_updates, rho):
    # Generate initial weights and weight samples.
    initial_weights = (tf.constant([0., 1., 2., 3.], dtype=tf.float32),
                       tf.constant([[0., 1.], [2., 3.]], dtype=tf.float32))
    weights_samples = []
    weights_sample = initial_weights
    for i in range(num_updates):
      index = i % 4
      weights_sample = tf.nest.map_structure(
          lambda x, y=index: x + tf.reshape(  # pylint: disable=g-long-lambda
              tf.one_hot(y, 4, dtype=x.dtype),
              shape=x.shape),
          weights_sample)
      weights_samples.append(weights_sample)

    @tf.function
    def _run_update_delta():
      # We have to wrap creation of `DeltaUpdateOutput` in a `tf.function` here
      # because of the issues passing `tf.TensorArray`s between TF modes.
      weights_sample = tf.nest.map_structure(
          lambda x: x + tf.reshape(  # pylint: disable=g-long-lambda
              tf.one_hot(0, 4, dtype=x.dtype),
              shape=x.shape),
          initial_weights)
      updates = fed_pa_schedule.DeltaUpdateOutput.from_weights(
          initial_weights=initial_weights,
          updated_weights=weights_sample,
          num_samples=1)
      for i in tf.range(1, num_updates):
        index = tf.math.mod(i, 4)
        weights_sample = tf.nest.map_structure(
            lambda x, i=index: x + tf.reshape(  # pylint: disable=g-long-lambda
                tf.one_hot(i, 4, dtype=x.dtype),
                shape=x.shape),
            weights_sample)
        outputs = fed_pa_schedule.DataPassOutput(
            loss=None,
            num_examples=None,
            model_weights_trainable=weights_sample,
            model_weights_trainable_sample=weights_sample)
        updates = fed_pa_schedule._update_delta_posterior_mean(
            outputs, updates, rho=rho)
      return updates

    # Compute updates.
    updates = _run_update_delta()
    self.assertEqual(updates.num_samples, num_updates)

    # Compute expected weights sample mean.
    expected_weights_sample_sum = tf.nest.map_structure(tf.zeros_like,
                                                        initial_weights)
    for weights_sample in weights_samples:
      expected_weights_sample_sum = tf.nest.map_structure(
          lambda a, b: a + b, expected_weights_sample_sum, weights_sample)
    expected_weights_sample_mean = tf.nest.map_structure(
        lambda x: x / num_updates, expected_weights_sample_sum)
    self.assertAllClose(updates.weights_sample_mean,
                        expected_weights_sample_mean)

    # Compute expected delta by inverting shrinkage covariance estimate.
    weights_delta_np = self.struct_to_tensor(updates.weights_delta).numpy()
    initial_weights_np = self.struct_to_tensor(initial_weights).numpy()
    shrinkage_rho = 1. / (1 + (num_updates - 1) * rho)
    ws = np.stack([self.struct_to_tensor(ws).numpy() for ws in weights_samples])
    ws_mean = np.mean(ws, axis=0)
    ws_cov = shrinkage_rho * np.eye(ws_mean.shape[0])
    if num_updates > 1:
      ws_cov += (1. - shrinkage_rho) * np.cov(ws.T)
    expected_weights_delta_np = np.linalg.solve(ws_cov,
                                                ws_mean - initial_weights_np)

    self.assertAllClose(weights_delta_np, expected_weights_delta_np)


if __name__ == '__main__':
  tf.test.main()
