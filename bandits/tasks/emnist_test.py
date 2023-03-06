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
"""Tests for emnist."""
import collections
import functools
from typing import Any
from absl.testing import parameterized

import tensorflow as tf
import tensorflow_federated as tff

from bandits import bandits_utils
from bandits.algs import epsilon_greedy
from bandits.tasks import emnist
from bandits.tasks import task_utils

_IMAGE_SIZE = 28
_BANDITS_DATA_SPEC = collections.OrderedDict(
    x=tf.TensorSpec(
        shape=(None, _IMAGE_SIZE, _IMAGE_SIZE, 1), dtype=tf.float32
    ),
    y=collections.OrderedDict(
        label=tf.TensorSpec(shape=(None,), dtype=tf.int32),
        action=tf.TensorSpec(shape=(None,), dtype=tf.int32),
        reward=tf.TensorSpec(shape=(None,), dtype=tf.float32),
        prob=tf.TensorSpec(shape=(None,), dtype=tf.float32),
    ),
)


build_epsilon_greedy_bandit_data_fn = functools.partial(
    epsilon_greedy.build_epsilon_greedy_bandit_data_fn, reward_fn=None
)


def _get_synthetic_batch(
    batch_size: int = 4, total_arms: int = 10
) -> dict[Any, Any]:
  # This synthetic batch uses random rewards, actions, etc.
  action = tf.random.uniform(
      shape=[batch_size], minval=0, maxval=total_arms, dtype=tf.int32
  )
  reward = tf.cast(
      tf.random.uniform(shape=[batch_size], minval=0, maxval=2, dtype=tf.int32),
      dtype=tf.float32,
  )
  # Set the probability to 0.5 as the action is random from two arms.
  prob = tf.constant(0.5, shape=[batch_size], dtype=tf.float32)
  label = tf.random.uniform(
      shape=[batch_size], minval=0, maxval=total_arms, dtype=tf.int32
  )
  features = tf.random.uniform(shape=[batch_size, _IMAGE_SIZE, _IMAGE_SIZE, 1])
  return collections.OrderedDict(
      x=features,
      y=collections.OrderedDict(
          label=label, action=action, reward=reward, prob=prob
      ),
  )


class EmnistTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      (
          'digits_bandits',
          True,
          task_utils.BanditsMSELoss(),
          emnist.ModelType.LENET,
      ),
      (
          'chars_bandits',
          False,
          task_utils.BanditsMSELoss(),
          emnist.ModelType.LENET,
      ),
      (
          'digits_supervised',
          True,
          task_utils.SupervisedMSELoss(10),
          emnist.ModelType.LENET,
      ),
      (
          'chars_supervised',
          False,
          task_utils.SupervisedMSELoss(62),
          emnist.ModelType.LENET,
      ),
      (
          'digits_bandits_linear',
          True,
          task_utils.BanditsMSELoss(),
          emnist.ModelType.LINEAR,
      ),
      (
          'chars_bandits_linear',
          False,
          task_utils.BanditsMSELoss(),
          emnist.ModelType.LINEAR,
      ),
      (
          'digits_supervised_linear',
          True,
          task_utils.SupervisedMSELoss(10),
          emnist.ModelType.LINEAR,
      ),
      (
          'chars_supervised_linear',
          False,
          task_utils.SupervisedMSELoss(62),
          emnist.ModelType.LINEAR,
      ),
  )
  def test_create_model(self, only_digits, loss, model_type):
    total_arms = 10 if only_digits else 62
    batch_size = 4
    model_fn = emnist.create_emnist_bandits_model_fn(
        _BANDITS_DATA_SPEC, only_digits=only_digits, loss=loss, model=model_type
    )
    tff_model = model_fn()
    self.assertIsInstance(tff_model, tff.learning.models.VariableModel)
    self.assertEqual(tff_model.input_spec['x'], _BANDITS_DATA_SPEC['x'])
    self.assertEqual(
        tff_model.input_spec['y'][bandits_utils.BanditsKeys.label],
        _BANDITS_DATA_SPEC['y'][bandits_utils.BanditsKeys.label],
    )
    self.assertEqual(
        tff_model.input_spec['y'][bandits_utils.BanditsKeys.action],
        _BANDITS_DATA_SPEC['y'][bandits_utils.BanditsKeys.action],
    )
    self.assertEqual(
        tff_model.input_spec['y'][bandits_utils.BanditsKeys.reward],
        _BANDITS_DATA_SPEC['y'][bandits_utils.BanditsKeys.reward],
    )
    self.assertEqual(
        tff_model.input_spec['y'][bandits_utils.BanditsKeys.prob],
        _BANDITS_DATA_SPEC['y'][bandits_utils.BanditsKeys.prob],
    )
    batch = _get_synthetic_batch(batch_size=batch_size, total_arms=total_arms)
    pred = tff_model.predict_on_batch(batch['x'])
    self.assertAllEqual(tf.shape(pred), [batch_size, total_arms])

  @parameterized.named_parameters(
      ('multilabel_mse', task_utils.MultiLabelMSELoss()),
      ('multilabel_ce', task_utils.MultiLabelCELoss()),
  )
  def test_create_model_raise_loss(self, loss):
    with self.assertRaisesRegex(ValueError, 'for EMNIST with sparse labels'):
      emnist.create_emnist_bandits_model_fn(_BANDITS_DATA_SPEC, loss=loss)

  @parameterized.named_parameters(('digits', True), ('chars', False))
  def test_create_dataset(self, only_digits):
    datasets = emnist.create_emnist_preprocessed_datasets(
        train_client_batch_size=4,
        test_client_batch_size=8,
        only_digits=only_digits,
        use_synthetic_data=True,
    )
    self.assertIsInstance(
        datasets, tff.simulation.baselines.BaselineTaskDatasets
    )
    _, bandit_data_spec = build_epsilon_greedy_bandit_data_fn(
        datasets.element_type_structure
    )
    self.assertEqual(bandit_data_spec['x'], _BANDITS_DATA_SPEC['x'])
    self.assertEqual(
        bandit_data_spec['y'][bandits_utils.BanditsKeys.label],
        _BANDITS_DATA_SPEC['y'][bandits_utils.BanditsKeys.label],
    )
    self.assertEqual(
        bandit_data_spec['y'][bandits_utils.BanditsKeys.action],
        _BANDITS_DATA_SPEC['y'][bandits_utils.BanditsKeys.action],
    )
    self.assertEqual(
        bandit_data_spec['y'][bandits_utils.BanditsKeys.reward],
        _BANDITS_DATA_SPEC['y'][bandits_utils.BanditsKeys.reward],
    )
    self.assertEqual(
        bandit_data_spec['y'][bandits_utils.BanditsKeys.prob],
        _BANDITS_DATA_SPEC['y'][bandits_utils.BanditsKeys.prob],
    )

  @parameterized.named_parameters(
      (
          'digits_bandits',
          True,
          task_utils.BanditsMSELoss(),
          emnist.ModelType.LENET,
      ),
      (
          'chars_bandits',
          False,
          task_utils.BanditsMSELoss(),
          emnist.ModelType.LENET,
      ),
      (
          'digits_supervised',
          True,
          task_utils.SupervisedMSELoss(10),
          emnist.ModelType.LENET,
      ),
      (
          'chars_supervised',
          False,
          task_utils.SupervisedMSELoss(62),
          emnist.ModelType.LENET,
      ),
      (
          'digits_bandits_linear',
          True,
          task_utils.BanditsMSELoss(),
          emnist.ModelType.LINEAR,
      ),
      (
          'chars_bandits_linear',
          False,
          task_utils.BanditsMSELoss(),
          emnist.ModelType.LINEAR,
      ),
      (
          'digits_supervised_linear',
          True,
          task_utils.SupervisedMSELoss(10),
          emnist.ModelType.LINEAR,
      ),
      (
          'chars_supervised_linear',
          False,
          task_utils.SupervisedMSELoss(62),
          emnist.ModelType.LINEAR,
      ),
  )
  def test_model_forward_backward_pass(self, only_digits, loss, model_type):
    train_client_batch_size, test_client_batch_size = 4, 8
    total_arms = 10 if only_digits else 62
    task_datasets = emnist.create_emnist_preprocessed_datasets(
        train_client_batch_size=train_client_batch_size,
        test_client_batch_size=test_client_batch_size,
        only_digits=only_digits,
        use_synthetic_data=True,
    )
    bandits_data_fn, bandit_data_spec = build_epsilon_greedy_bandit_data_fn(
        task_datasets.element_type_structure
    )
    model_fn = emnist.create_emnist_bandits_model_fn(
        bandit_data_spec, only_digits=only_digits, loss=loss, model=model_type
    )
    tff_model = model_fn()
    # Training
    sampled_clients = task_datasets.sample_train_clients(
        num_clients=1, random_seed=42
    )
    bandits_dataset = bandits_data_fn(
        tff_model,
        tff.learning.models.ModelWeights.from_model(tff_model),
        sampled_clients[0],
    )
    batch = iter(bandits_dataset).get_next()
    with tf.GradientTape() as tape:
      batch_output = tff_model.forward_pass(batch, training=True)
    self.assertGreaterEqual(batch_output.loss, 0)
    self.assertEqual(batch_output.num_examples, train_client_batch_size)
    grads = tape.gradient(batch_output.loss, tff_model.trainable_variables)
    tf.nest.map_structure(
        lambda x, y: self.assertShapeEqual(x, tf.convert_to_tensor(y)),
        grads,
        tff_model.trainable_variables,
    )
    metrics = tff_model.report_local_unfinalized_metrics()
    acc = (
        metrics['sparse_categorical_accuracy'][0]
        / metrics['sparse_categorical_accuracy'][1]
    )
    loss = metrics['loss'][0] / metrics['loss'][1]
    self.assertEqual(metrics['num_batches'][0], 1)
    self.assertEqual(metrics['num_examples'][0], train_client_batch_size)
    self.assertNear(batch_output.loss, loss, err=1e-5)
    self.assertLess(acc, 1.0)
    # Validation
    test_dataset = bandits_utils.dataset_format_map(
        task_datasets.get_centralized_test_data()
    )
    test_batch = iter(test_dataset).get_next()
    pred = tff_model.predict_on_batch(test_batch['x'])
    self.assertAllEqual(tf.shape(pred), [test_client_batch_size, total_arms])


if __name__ == '__main__':
  tf.test.main()
