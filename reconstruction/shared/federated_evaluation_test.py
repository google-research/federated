# Copyright 2020, Google LLC.
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
"""Tests for federated_evaluation.py."""

import collections

from absl.testing import parameterized
import attr
import tensorflow as tf

from reconstruction import keras_utils
from reconstruction import reconstruction_model
from reconstruction.shared import federated_evaluation


def _create_input_spec():
  return collections.OrderedDict(
      x=tf.TensorSpec(shape=[None, 1], dtype=tf.float32),
      y=tf.TensorSpec(shape=[None, 1], dtype=tf.float32))


@attr.s(eq=False, frozen=True)
class LinearModelVariables(object):
  """Structure for variables in `LinearModel`."""
  weights = attr.ib()
  bias = attr.ib()


class LinearModel(reconstruction_model.ReconstructionModel):
  """An implementation of an MNIST `ReconstructionModel` without Keras."""

  def __init__(self, global_variables_only=False):
    self._variables = LinearModelVariables(
        weights=tf.Variable(
            lambda: tf.zeros(dtype=tf.float32, shape=(1, 1)),
            name='weights',
            trainable=True),
        bias=tf.Variable(
            lambda: tf.zeros(dtype=tf.float32, shape=(1,)),
            name='bias',
            trainable=True))
    self._global_variables_only = global_variables_only

  @property
  def global_trainable_variables(self):
    if self._global_variables_only:
      return [self._variables.weights, self._variables.bias]
    return [self._variables.weights]

  @property
  def global_non_trainable_variables(self):
    return []

  @property
  def local_trainable_variables(self):
    if self._global_variables_only:
      return []
    return [self._variables.bias]

  @property
  def local_non_trainable_variables(self):
    return []

  @property
  def input_spec(self):
    return _create_input_spec()

  @tf.function
  def forward_pass(self, batch, training=True):
    del training

    y = batch['x'] * self._variables.weights + self._variables.bias
    return reconstruction_model.BatchOutput(
        predictions=y, labels=batch['y'], num_examples=tf.size(batch['y']))


class BiasLayer(tf.keras.layers.Layer):
  """Adds a bias to inputs."""

  def build(self, input_shape):
    self.bias = self.add_weight(
        'bias', shape=input_shape[1:], initializer='zeros', trainable=True)

  def call(self, x):
    return x + self.bias


def keras_linear_model_fn(global_variables_only=False):
  """Should produce the same results as `LinearModel`."""
  inputs = tf.keras.layers.Input(shape=[1])
  scaled_input = tf.keras.layers.Dense(
      1, use_bias=False, kernel_initializer='zeros')(
          inputs)
  outputs = BiasLayer()(scaled_input)
  keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
  input_spec = _create_input_spec()
  if global_variables_only:
    global_layers = keras_model.layers
    local_layers = []
  else:
    global_layers = keras_model.layers[:-1]
    local_layers = keras_model.layers[-1:]
  return keras_utils.from_keras_model(
      keras_model=keras_model,
      global_layers=global_layers,
      local_layers=local_layers,
      input_spec=input_spec)


class NumExamplesCounter(tf.keras.metrics.Sum):
  """A `tf.keras.metrics.Metric` that counts the number of examples seen.

  This metric counts label examples.
  """

  def __init__(self, name: str = 'num_examples_total', dtype=tf.float32):  # pylint: disable=useless-super-delegation
    super().__init__(name, dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    return super().update_state(tf.shape(y_true)[0])


class NumOverCounter(tf.keras.metrics.Sum):
  """A `tf.keras.metrics.Metric` that counts examples greater than a constant.

  This metric counts label examples greater than a threshold.
  """

  def __init__(self,
               threshold: float,
               name: str = 'num_over',
               dtype=tf.float32):
    super().__init__(name, dtype)
    self.threshold = threshold

  def update_state(self, y_true, y_pred, sample_weight=None):
    num_over = tf.reduce_sum(
        tf.cast(tf.greater(y_true, self.threshold), tf.float32))
    return super().update_state(num_over)

  def get_config(self):
    config = {'threshold': self.threshold}
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


def create_client_data():
  client1_data = collections.OrderedDict([('x', [[1.0], [2.0], [3.0]]),
                                          ('y', [[5.0], [6.0], [8.0]])])
  client2_data = collections.OrderedDict([('x', [[1.0], [2.0], [3.0]]),
                                          ('y', [[5.0], [5.0], [9.0]])])

  client1_dataset = tf.data.Dataset.from_tensor_slices(client1_data).batch(1)
  client2_dataset = tf.data.Dataset.from_tensor_slices(client2_data).batch(1)

  return [client1_dataset, client2_dataset]


class FederatedFinetuneEvaluationTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('non_keras_model', lambda: LinearModel(global_variables_only=False)),
      ('keras_model',
       lambda: keras_linear_model_fn(global_variables_only=False)))
  def test_federated_finetune_evaluation_invalid_model_fn(self, model_fn):

    with self.assertRaisesRegex(
        ValueError,
        '`model_fn` should return a model with only global variables.'):
      federated_evaluation.build_federated_finetune_evaluation(
          model_fn, loss_fn=tf.keras.losses.MeanSquaredError)

  @parameterized.named_parameters(
      ('non_keras_model', lambda: LinearModel(global_variables_only=True)),
      ('keras_model', lambda: keras_linear_model_fn(global_variables_only=True))
  )
  def test_federated_finetune_evaluation_no_finetune(self, model_fn):

    def loss_fn():
      return tf.keras.losses.MeanSquaredError()

    def metrics_fn():
      return [NumExamplesCounter(), NumOverCounter(5.0)]

    def dataset_split_fn(client_dataset, round_num):
      del round_num
      return client_dataset, client_dataset

    evaluate = federated_evaluation.build_federated_finetune_evaluation(
        model_fn,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        # Set fine-tune optimizer LR to 0 so fine-tuning has no effect.
        finetune_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.0),
        dataset_split_fn=dataset_split_fn)

    self.assertEqual(
        str(evaluate.type_signature),
        '(<server_model_weights=<trainable=<float32[1,1],float32[1]>,'
        'non_trainable=<>>@SERVER,federated_dataset={<x=float32[?,1],'
        'y=float32[?,1]>*}@CLIENTS> -> '
        '<loss=float32,num_examples_total=float32,num_over=float32>@SERVER)')

    result = evaluate(
        collections.OrderedDict([
            ('trainable', [[[1.0]], [0.0]]),
            ('non_trainable', []),
        ]), create_client_data())

    expected_keys = ['loss', 'num_examples_total', 'num_over']
    self.assertCountEqual(result.keys(), expected_keys)
    # The weight is 1.0 and bias is 0.0. Both variables are not fine-tuned. The
    # MSE is (y - 1 * x)^2 for each example, for a mean of
    # (4^2 + 4^2 + 5^2 + 4^2 + 3^2 + 6^2) / 6 = 59/3.
    self.assertAlmostEqual(result['loss'], 19.666666)
    self.assertAlmostEqual(result['num_examples_total'], 6.0)
    self.assertAlmostEqual(result['num_over'], 3.0)

  @parameterized.named_parameters(
      ('non_keras_model', lambda: LinearModel(global_variables_only=True)),
      ('keras_model', lambda: keras_linear_model_fn(global_variables_only=True))
  )
  def test_federated_finetune_evaluation_loss_decreases(self, model_fn):

    def loss_fn():
      return tf.keras.losses.MeanSquaredError()

    def dataset_split_fn(client_dataset, round_num):
      del round_num
      return client_dataset.repeat(2), client_dataset.repeat(1)

    evaluate = federated_evaluation.build_federated_finetune_evaluation(
        model_fn,
        loss_fn=loss_fn,
        metrics_fn=None,
        finetune_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.01),
        dataset_split_fn=dataset_split_fn)
    self.assertEqual(
        str(evaluate.type_signature),
        '(<server_model_weights=<trainable=<float32[1,1],float32[1]>,'
        'non_trainable=<>>@SERVER,federated_dataset={<x=float32[?,1],'
        'y=float32[?,1]>*}@CLIENTS> -> <loss=float32>@SERVER)')

    result = evaluate(
        collections.OrderedDict([
            ('trainable', [[[1.0]], [0.0]]),
            ('non_trainable', []),
        ]), create_client_data())

    expected_keys = ['loss']
    self.assertCountEqual(result.keys(), expected_keys)
    # We use the same dataset for fine-tuning and evaluation. The loss on the
    # evaluation dataset should decrease after fine-tuning. The initial weight
    # is 1.0 and bias is 0.0, so the inital MSE is (y - 1 * x)^2 for each
    # example, for a mean of (4^2 + 4^2 + 5^2 + 4^2 + 3^2 + 6^2) / 6 = 59/3.
    self.assertLess(result['loss'], 19.666666)

  @parameterized.named_parameters(
      ('non_keras_model', lambda: LinearModel(global_variables_only=True)),
      ('keras_model', lambda: keras_linear_model_fn(global_variables_only=True))
  )
  def test_federated_finetune_evaluation_split_data(self, model_fn):

    def loss_fn():
      return tf.keras.losses.MeanSquaredError()

    def metrics_fn():
      return [NumExamplesCounter(), NumOverCounter(5.0)]

    def dataset_split_fn(client_dataset, round_num):
      del round_num
      return client_dataset.take(1).repeat(2), client_dataset.skip(1).repeat(5)

    evaluate = federated_evaluation.build_federated_finetune_evaluation(
        model_fn,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        finetune_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.1),
        dataset_split_fn=dataset_split_fn)
    self.assertEqual(
        str(evaluate.type_signature),
        '(<server_model_weights=<trainable=<float32[1,1],float32[1]>,'
        'non_trainable=<>>@SERVER,federated_dataset={<x=float32[?,1],'
        'y=float32[?,1]>*}@CLIENTS> -> '
        '<loss=float32,num_examples_total=float32,num_over=float32>@SERVER)')

    result = evaluate(
        collections.OrderedDict([
            ('trainable', [[[1.0]], [0.0]]),
            ('non_trainable', []),
        ]), create_client_data())

    expected_keys = ['loss', 'num_examples_total', 'num_over']
    self.assertCountEqual(result.keys(), expected_keys)
    self.assertAlmostEqual(result['num_examples_total'], 20.0)
    self.assertAlmostEqual(result['num_over'], 15.0)


if __name__ == '__main__':
  tf.test.main()
