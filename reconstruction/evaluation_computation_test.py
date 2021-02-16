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
"""Tests for evaluation_computation.py."""

import collections

from absl.testing import parameterized
import attr
import tensorflow as tf

from reconstruction import evaluation_computation
from reconstruction import keras_utils
from reconstruction import reconstruction_model
from reconstruction import reconstruction_utils


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

  def __init__(self):
    self._variables = LinearModelVariables(
        weights=tf.Variable(
            lambda: tf.zeros(dtype=tf.float32, shape=(1, 1)),
            name='weights',
            trainable=True),
        bias=tf.Variable(
            lambda: tf.zeros(dtype=tf.float32, shape=(1, 1)),
            name='bias',
            trainable=True))

  @property
  def global_trainable_variables(self):
    return [self._variables.weights]

  @property
  def global_non_trainable_variables(self):
    return []

  @property
  def local_trainable_variables(self):
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


def keras_linear_model_fn():
  """Should produce the same results as `LinearModel`."""
  inputs = tf.keras.layers.Input(shape=[1])
  scaled_input = tf.keras.layers.Dense(
      1, use_bias=False, kernel_initializer='zeros')(
          inputs)
  outputs = BiasLayer()(scaled_input)
  keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
  input_spec = _create_input_spec()
  return keras_utils.from_keras_model(
      keras_model=keras_model,
      global_layers=keras_model.layers[:-1],
      local_layers=keras_model.layers[-1:],
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


class FedreconEvaluationTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if tf.config.list_logical_devices('GPU'):
      self.skipTest('skip GPU test')

  @parameterized.named_parameters(('non_keras_model', LinearModel),
                                  ('keras_model', keras_linear_model_fn))
  def test_federated_reconstruction_no_split_data(self, model_fn):

    def loss_fn():
      return tf.keras.losses.MeanSquaredError()

    def metrics_fn():
      return [NumExamplesCounter(), NumOverCounter(5.0)]

    dataset_split_fn = reconstruction_utils.build_dataset_split_fn()

    evaluate = evaluation_computation.build_federated_reconstruction_evaluation(
        model_fn,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        reconstruction_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.1),
        dataset_split_fn=dataset_split_fn)
    self.assertEqual(
        str(evaluate.type_signature),
        '(<server_model_weights=<trainable=<float32[1,1]>,'
        'non_trainable=<>>@SERVER,federated_dataset={<x=float32[?,1],'
        'y=float32[?,1]>*}@CLIENTS> -> '
        '<loss=float32,num_examples_total=float32,num_over=float32>@SERVER)')

    result = evaluate(
        collections.OrderedDict([
            ('trainable', [[[5.0]]]),
            ('non_trainable', []),
        ]), create_client_data())

    expected_keys = ['loss', 'num_examples_total', 'num_over']
    self.assertCountEqual(result.keys(), expected_keys)
    self.assertAlmostEqual(result['num_examples_total'], 6.0)
    self.assertAlmostEqual(result['num_over'], 3.0)

  @parameterized.named_parameters(('non_keras_model', LinearModel),
                                  ('keras_model', keras_linear_model_fn))
  def test_federated_reconstruction_split_data(self, model_fn):

    def loss_fn():
      return tf.keras.losses.MeanSquaredError()

    def metrics_fn():
      return [NumExamplesCounter(), NumOverCounter(5.0)]

    evaluate = evaluation_computation.build_federated_reconstruction_evaluation(
        model_fn,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        reconstruction_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.1))
    self.assertEqual(
        str(evaluate.type_signature),
        '(<server_model_weights=<trainable=<float32[1,1]>,'
        'non_trainable=<>>@SERVER,federated_dataset={<x=float32[?,1],'
        'y=float32[?,1]>*}@CLIENTS> -> '
        '<loss=float32,num_examples_total=float32,num_over=float32>@SERVER)')

    result = evaluate(
        collections.OrderedDict([
            ('trainable', [[[5.0]]]),
            ('non_trainable', []),
        ]), create_client_data())

    expected_keys = ['loss', 'num_examples_total', 'num_over']
    self.assertCountEqual(result.keys(), expected_keys)
    self.assertAlmostEqual(result['num_examples_total'], 2.0)
    self.assertAlmostEqual(result['num_over'], 1.0)

  @parameterized.named_parameters(('non_keras_model', LinearModel),
                                  ('keras_model', keras_linear_model_fn))
  def test_federated_reconstruction_split_data_multiple_epochs(self, model_fn):

    def loss_fn():
      return tf.keras.losses.MeanSquaredError()

    def metrics_fn():
      return [NumExamplesCounter(), NumOverCounter(5.0)]

    dataset_split_fn = reconstruction_utils.build_dataset_split_fn(
        recon_epochs_max=2,
        post_recon_epochs=10,
        post_recon_steps_max=7,
        split_dataset=True)

    evaluate = evaluation_computation.build_federated_reconstruction_evaluation(
        model_fn,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        reconstruction_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.1),
        dataset_split_fn=dataset_split_fn)
    self.assertEqual(
        str(evaluate.type_signature),
        '(<server_model_weights=<trainable=<float32[1,1]>,'
        'non_trainable=<>>@SERVER,federated_dataset={<x=float32[?,1],'
        'y=float32[?,1]>*}@CLIENTS> -> '
        '<loss=float32,num_examples_total=float32,num_over=float32>@SERVER)')

    result = evaluate(
        collections.OrderedDict([
            ('trainable', [[[5.0]]]),
            ('non_trainable', []),
        ]), create_client_data())

    expected_keys = ['loss', 'num_examples_total', 'num_over']
    self.assertCountEqual(result.keys(), expected_keys)
    self.assertAlmostEqual(result['num_examples_total'], 14.0)
    self.assertAlmostEqual(result['num_over'], 7.0)

  @parameterized.named_parameters(('non_keras_model', LinearModel),
                                  ('keras_model', keras_linear_model_fn))
  def test_federated_reconstruction_recon_lr_0(self, model_fn):

    def loss_fn():
      return tf.keras.losses.MeanSquaredError()

    def metrics_fn():
      return [NumExamplesCounter(), NumOverCounter(5.0)]

    dataset_split_fn = reconstruction_utils.build_dataset_split_fn()

    evaluate = evaluation_computation.build_federated_reconstruction_evaluation(
        model_fn,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        # Set recon optimizer LR to 0 so reconstruction has no effect.
        reconstruction_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.0),
        dataset_split_fn=dataset_split_fn)
    self.assertEqual(
        str(evaluate.type_signature),
        '(<server_model_weights=<trainable=<float32[1,1]>,'
        'non_trainable=<>>@SERVER,federated_dataset={<x=float32[?,1],'
        'y=float32[?,1]>*}@CLIENTS> -> '
        '<loss=float32,num_examples_total=float32,num_over=float32>@SERVER)')

    result = evaluate(
        collections.OrderedDict([
            ('trainable', [[[1.0]]]),
            ('non_trainable', []),
        ]), create_client_data())

    expected_keys = ['loss', 'num_examples_total', 'num_over']
    self.assertCountEqual(result.keys(), expected_keys)
    # Now have an expectation for loss since the local bias is initialized at 0
    # and not reconstructed. MSE is (y - 1 * x)^2 for each example, for a mean
    # of (4^2 + 4^2 + 5^2 + 4^2 + 3^2 + 6^2) / 6 = 59/3.
    self.assertAlmostEqual(result['loss'], 19.666666)
    self.assertAlmostEqual(result['num_examples_total'], 6.0)
    self.assertAlmostEqual(result['num_over'], 3.0)

  @parameterized.named_parameters(('non_keras_model', LinearModel),
                                  ('keras_model', keras_linear_model_fn))
  def test_federated_reconstruction_skip_recon(self, model_fn):

    def loss_fn():
      return tf.keras.losses.MeanSquaredError()

    def metrics_fn():
      return [NumExamplesCounter(), NumOverCounter(5.0)]

    # Ensure reconstruction is skipped if `recon_dataset` is empty. This also
    # ensures `round_num` is 0 for evaluation and loss doesn't change if
    # `eval_dataset` is repeated.
    def dataset_split_fn(client_dataset, round_num):
      return client_dataset.repeat(round_num), client_dataset.repeat(2)

    evaluate = evaluation_computation.build_federated_reconstruction_evaluation(
        model_fn,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        reconstruction_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.1),
        dataset_split_fn=dataset_split_fn)
    self.assertEqual(
        str(evaluate.type_signature),
        '(<server_model_weights=<trainable=<float32[1,1]>,'
        'non_trainable=<>>@SERVER,federated_dataset={<x=float32[?,1],'
        'y=float32[?,1]>*}@CLIENTS> -> '
        '<loss=float32,num_examples_total=float32,num_over=float32>@SERVER)')

    result = evaluate(
        collections.OrderedDict([
            ('trainable', [[[1.0]]]),
            ('non_trainable', []),
        ]), create_client_data())

    expected_keys = ['loss', 'num_examples_total', 'num_over']
    self.assertCountEqual(result.keys(), expected_keys)
    # Now have an expectation for loss since the local bias is initialized at 0
    # and not reconstructed. MSE is (y - 1 * x)^2 for each example, for a mean
    # of (4^2 + 4^2 + 5^2 + 4^2 + 3^2 + 6^2) / 6 = 59/3
    self.assertAlmostEqual(result['loss'], 19.666666)
    self.assertAlmostEqual(result['num_examples_total'], 12.0)
    self.assertAlmostEqual(result['num_over'], 6.0)

  @parameterized.named_parameters(('non_keras_model', LinearModel),
                                  ('keras_model', keras_linear_model_fn))
  def test_federated_reconstruction_metrics_none_loss_decreases(self, model_fn):

    def loss_fn():
      return tf.keras.losses.MeanSquaredError()

    dataset_split_fn = reconstruction_utils.build_dataset_split_fn(
        recon_epochs_max=3)

    evaluate = evaluation_computation.build_federated_reconstruction_evaluation(
        model_fn,
        loss_fn=loss_fn,
        metrics_fn=None,
        reconstruction_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.01),
        dataset_split_fn=dataset_split_fn)
    self.assertEqual(
        str(evaluate.type_signature),
        '(<server_model_weights=<trainable=<float32[1,1]>,'
        'non_trainable=<>>@SERVER,federated_dataset={<x=float32[?,1],'
        'y=float32[?,1]>*}@CLIENTS> -> <loss=float32>@SERVER)')

    result = evaluate(
        collections.OrderedDict([
            ('trainable', [[[1.0]]]),
            ('non_trainable', []),
        ]), create_client_data())

    expected_keys = ['loss']
    self.assertCountEqual(result.keys(), expected_keys)
    # Ensure loss decreases from reconstruction vs. initializing the bias to 0.
    # MSE is (y - 1 * x)^2 for each example, for a mean of
    # (4^2 + 4^2 + 5^2 + 4^2 + 3^2 + 6^2) / 6 = 59/3.
    self.assertLess(result['loss'], 19.666666)

  @parameterized.named_parameters(('non_keras_model', LinearModel),
                                  ('keras_model', keras_linear_model_fn))
  def test_federated_reconstruction_evaluation_process(self, model_fn):

    def loss_fn():
      return tf.keras.losses.MeanSquaredError()

    def metrics_fn():
      return [NumExamplesCounter(), NumOverCounter(5.0)]

    dataset_split_fn = reconstruction_utils.build_dataset_split_fn(
        recon_epochs_max=2,
        post_recon_epochs=10,
        post_recon_steps_max=7,
        split_dataset=True)

    evaluator = evaluation_computation.build_federated_reconstruction_evaluation_process(
        model_fn,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        reconstruction_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.1),
        dataset_split_fn=dataset_split_fn)
    self.assertEqual(
        str(evaluator.initialize.type_signature),
        '( -> <model=<trainable=<float32[1,1]>,non_trainable=<>>,'
        'optimizer_state=<>,round_num=int64,aggregator_state=<>>@SERVER)')
    self.assertEqual(
        str(evaluator.next.type_signature),
        '(<state=<model=<trainable=<float32[1,1]>,non_trainable=<>>,'
        'optimizer_state=<>,round_num=int64,aggregator_state=<>>@SERVER,'
        'data={<x=float32[?,1],y=float32[?,1]>*}@CLIENTS> -> '
        '<<model=<trainable=<float32[1,1]>,non_trainable=<>>,'
        'optimizer_state=<>,round_num=int64,aggregator_state=<>>@SERVER,'
        '<loss=float32,num_examples_total=float32,num_over=float32>@SERVER>)')

    state = evaluator.initialize()
    state, metrics = evaluator.next(state, create_client_data())

    expected_keys = ['loss', 'num_examples_total', 'num_over']
    self.assertCountEqual(metrics.keys(), expected_keys)
    self.assertAlmostEqual(metrics['num_examples_total'], 14.0)
    self.assertAlmostEqual(metrics['num_over'], 7.0)

  @parameterized.named_parameters(('non_keras_model', LinearModel),
                                  ('keras_model', keras_linear_model_fn))
  def test_federated_reconstruction_evaluation_process_no_recon(self, model_fn):

    def loss_fn():
      return tf.keras.losses.MeanSquaredError()

    def metrics_fn():
      return [NumExamplesCounter(), NumOverCounter(5.0)]

    dataset_split_fn = reconstruction_utils.build_dataset_split_fn(
        recon_epochs_max=0, post_recon_epochs=2)

    evaluator = evaluation_computation.build_federated_reconstruction_evaluation_process(
        model_fn,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        reconstruction_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.1),
        dataset_split_fn=dataset_split_fn)
    self.assertEqual(
        str(evaluator.initialize.type_signature),
        '( -> <model=<trainable=<float32[1,1]>,non_trainable=<>>,'
        'optimizer_state=<>,round_num=int64,aggregator_state=<>>@SERVER)')
    self.assertEqual(
        str(evaluator.next.type_signature),
        '(<state=<model=<trainable=<float32[1,1]>,non_trainable=<>>,'
        'optimizer_state=<>,round_num=int64,aggregator_state=<>>@SERVER,'
        'data={<x=float32[?,1],y=float32[?,1]>*}@CLIENTS> -> '
        '<<model=<trainable=<float32[1,1]>,non_trainable=<>>,'
        'optimizer_state=<>,round_num=int64,aggregator_state=<>>@SERVER,'
        '<loss=float32,num_examples_total=float32,num_over=float32>@SERVER>)')

    state = evaluator.initialize()
    state, metrics = evaluator.next(state, create_client_data())

    expected_keys = ['loss', 'num_examples_total', 'num_over']
    self.assertCountEqual(metrics.keys(), expected_keys)
    self.assertAlmostEqual(metrics['num_examples_total'], 12.0)
    self.assertAlmostEqual(metrics['num_over'], 6.0)

    # Without reconstruction and with an initialized model, we can expect an
    # exact value for loss.
    state = reconstruction_utils.ServerState(
        model=collections.OrderedDict([
            ('trainable', [[[1.0]]]),
            ('non_trainable', []),
        ]),
        optimizer_state=(),
        round_num=tf.constant(0, dtype=tf.int64),
        aggregator_state=(),
    )

    state, metrics = evaluator.next(state, create_client_data())

    expected_keys = ['loss', 'num_examples_total', 'num_over']
    self.assertCountEqual(metrics.keys(), expected_keys)
    # MSE is (y - 1 * x)^2 for each example, for a mean of
    # (4^2 + 4^2 + 5^2 + 4^2 + 3^2 + 6^2) / 6 = 59/3.
    self.assertAlmostEqual(metrics['loss'], 19.666666)
    self.assertAlmostEqual(metrics['num_examples_total'], 12.0)
    self.assertAlmostEqual(metrics['num_over'], 6.0)

  @parameterized.named_parameters(('non_keras_model', LinearModel),
                                  ('keras_model', keras_linear_model_fn))
  def test_federated_reconstruction_evaluation_process_one_client_split(
      self, model_fn):

    def loss_fn():
      return tf.keras.losses.MeanSquaredError()

    def metrics_fn():
      return [NumExamplesCounter(), NumOverCounter(5.0)]

    dataset_split_fn = reconstruction_utils.build_dataset_split_fn(
        recon_epochs_max=0, post_recon_epochs=2, split_dataset=True)

    evaluator = evaluation_computation.build_federated_reconstruction_evaluation_process(
        model_fn,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        reconstruction_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.1),
        dataset_split_fn=dataset_split_fn)
    self.assertEqual(
        str(evaluator.initialize.type_signature),
        '( -> <model=<trainable=<float32[1,1]>,non_trainable=<>>,'
        'optimizer_state=<>,round_num=int64,aggregator_state=<>>@SERVER)')
    self.assertEqual(
        str(evaluator.next.type_signature),
        '(<state=<model=<trainable=<float32[1,1]>,non_trainable=<>>,'
        'optimizer_state=<>,round_num=int64,aggregator_state=<>>@SERVER,'
        'data={<x=float32[?,1],y=float32[?,1]>*}@CLIENTS> -> '
        '<<model=<trainable=<float32[1,1]>,non_trainable=<>>,'
        'optimizer_state=<>,round_num=int64,aggregator_state=<>>@SERVER,'
        '<loss=float32,num_examples_total=float32,num_over=float32>@SERVER>)')

    state = evaluator.initialize()
    state, metrics = evaluator.next(state, create_client_data()[1:])

    expected_keys = ['loss', 'num_examples_total', 'num_over']
    self.assertCountEqual(metrics.keys(), expected_keys)
    self.assertAlmostEqual(metrics['num_examples_total'], 2.0)
    self.assertAlmostEqual(metrics['num_over'], 0.0)

    # Without reconstruction and with an initialized model, we can expect an
    # exact value for loss.
    state = reconstruction_utils.ServerState(
        model=collections.OrderedDict([
            ('trainable', [[[1.0]]]),
            ('non_trainable', []),
        ]),
        optimizer_state=(),
        round_num=tf.constant(0, dtype=tf.int64),
        aggregator_state=(),
    )

    state, metrics = evaluator.next(state, create_client_data()[1:])

    expected_keys = ['loss', 'num_examples_total', 'num_over']
    self.assertCountEqual(metrics.keys(), expected_keys)
    # MSE is (y - 1 * x)^2 for each example, for a mean of 3^2 / 1 = 9.
    self.assertAlmostEqual(metrics['loss'], 9.0)
    self.assertAlmostEqual(metrics['num_examples_total'], 2.0)
    self.assertAlmostEqual(metrics['num_over'], 0.0)


if __name__ == '__main__':
  tf.test.main()
