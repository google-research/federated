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
"""Tests for training_process.py."""

import collections
import functools

import attr
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import tensorflow_privacy as tfp

from reconstruction import keras_utils
from reconstruction import reconstruction_model
from reconstruction import reconstruction_utils
from reconstruction import training_process


def _create_input_spec():
  return collections.OrderedDict(
      x=tf.TensorSpec(shape=[None, 784], dtype=tf.float32),
      y=tf.TensorSpec(dtype=tf.int32, shape=[None, 1]))


def global_recon_model_fn():
  """Keras MNIST model with no local variables."""
  keras_model = tff.simulation.models.mnist.create_keras_model(
      compile_model=False)
  input_spec = _create_input_spec()
  return keras_utils.from_keras_model(
      keras_model=keras_model,
      global_layers=keras_model.layers,
      local_layers=[],
      input_spec=input_spec)


def local_recon_model_fn():
  """Keras MNIST model with final dense layer local."""
  keras_model = tff.simulation.models.mnist.create_keras_model(
      compile_model=False)
  input_spec = _create_input_spec()
  return keras_utils.from_keras_model(
      keras_model=keras_model,
      global_layers=keras_model.layers[:-1],
      local_layers=keras_model.layers[-1:],
      input_spec=input_spec)


@attr.s(eq=False, frozen=True)
class MnistVariables(object):
  """Structure for variables in an MNIST model."""
  weights = attr.ib()
  bias = attr.ib()


class MnistModel(reconstruction_model.ReconstructionModel):
  """An implementation of an MNIST `ReconstructionModel` without Keras.

  Applies a single dense layer followed by softmax. The weights of the dense
  layer are global, and the biases are local.
  """

  def __init__(self):
    self._variables = MnistVariables(
        weights=tf.Variable(
            lambda: tf.zeros(dtype=tf.float32, shape=(784, 10)),
            name='weights',
            trainable=True),
        bias=tf.Variable(
            lambda: tf.zeros(dtype=tf.float32, shape=(10)),
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
    return collections.OrderedDict([('x', tf.TensorSpec([None, 784],
                                                        tf.float32)),
                                    ('y', tf.TensorSpec([None, 1], tf.int32))])

  @tf.function
  def forward_pass(self, batch, training=True):
    del training

    y = tf.nn.softmax(
        tf.matmul(batch['x'], self._variables.weights) + self._variables.bias)
    return reconstruction_model.BatchOutput(
        predictions=y, labels=batch['y'], num_examples=tf.size(batch['y']))


class NumExamplesCounter(tf.keras.metrics.Sum):
  """A `tf.keras.metrics.Metric` that counts the number of examples seen.

  This metric counts label examples.
  """

  def __init__(self, name: str = 'num_examples_total', dtype=tf.float32):  # pylint: disable=useless-super-delegation
    super().__init__(name, dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    return super().update_state(tf.shape(y_true)[0])


class NumBatchesCounter(tf.keras.metrics.Sum):
  """A `tf.keras.metrics.Metric` that counts the number of batches seen."""

  def __init__(self, name: str = 'num_batches_total', dtype=tf.float32):  # pylint: disable=useless-super-delegation
    super().__init__(name, dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    return super().update_state(1)


def create_emnist_client_data():
  np.random.seed(42)
  emnist_data = collections.OrderedDict([('x', [
      np.random.randn(784).astype(np.float32),
      np.random.randn(784).astype(np.float32),
      np.random.randn(784).astype(np.float32)
  ]), ('y', [[5], [5], [9]])])

  dataset = tf.data.Dataset.from_tensor_slices(emnist_data)

  def client_data(batch_size=2):
    return dataset.batch(batch_size)

  return client_data


class _DPMean(tff.aggregators.UnweightedAggregationFactory):

  def __init__(self, dp_sum_factory):
    self._dp_sum = dp_sum_factory
    self._clear_sum = tff.aggregators.SumFactory()

  def create(self, value_type: tff.Type) -> tff.templates.AggregationProcess:
    self._dp_sum_process = self._dp_sum.create(value_type)

    @tff.federated_computation()
    def init():
      # Invoke here to instantiate anything we need
      return self._dp_sum_process.initialize()

    @tff.tf_computation(value_type, tf.int32)
    def div(x, y):
      # Opaque shape manipulations
      return [tf.squeeze(tf.math.divide_no_nan(x, tf.cast(y, tf.float32)), 0)]

    @tff.federated_computation(init.type_signature.result,
                               tff.type_at_clients(value_type))
    def next_fn(state, value):
      one_at_clients = tff.federated_value(1, tff.CLIENTS)
      dp_sum = self._dp_sum_process.next(state, value)
      summed_one = tff.federated_sum(one_at_clients)
      return tff.templates.MeasuredProcessOutput(
          state=dp_sum.state,
          result=tff.federated_map(div, (dp_sum.result, summed_one)),
          measurements=dp_sum.measurements)

    return tff.templates.AggregationProcess(initialize_fn=init, next_fn=next_fn)


class TrainingProcessTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    if tf.config.list_logical_devices('GPU'):
      self.skipTest('skip GPU test')

  def _run_rounds(self, iterproc, federated_data, num_rounds):
    train_outputs = []
    initial_state = iterproc.initialize()
    state = initial_state
    for _ in range(num_rounds):
      state, metrics = iterproc.next(state, federated_data)
      train_outputs.append(metrics)
    return state, train_outputs, initial_state

  def test_build_train_iterative_process(self):

    def loss_fn():
      return tf.keras.losses.SparseCategoricalCrossentropy()

    def metrics_fn():
      return [
          NumExamplesCounter(),
          NumBatchesCounter(),
          tf.keras.metrics.SparseCategoricalAccuracy()
      ]

    it_process = training_process.build_federated_reconstruction_process(
        local_recon_model_fn,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        client_optimizer_fn=functools.partial(tf.keras.optimizers.SGD, 0.1))

    self.assertIsInstance(it_process, tff.templates.IterativeProcess)
    federated_data_type = it_process.next.type_signature.parameter[1]
    self.assertEqual(
        str(federated_data_type), '{<x=float32[?,784],y=int32[?,1]>*}@CLIENTS')

  def test_fed_recon_with_custom_client_weight_fn(self):
    client_data = create_emnist_client_data()
    federated_data = [client_data()]

    def client_weight_fn(local_outputs):
      return 1.0 / (1.0 + local_outputs['loss'][-1])

    it_process = training_process.build_federated_reconstruction_process(
        local_recon_model_fn,
        loss_fn=tf.keras.losses.SparseCategoricalCrossentropy,
        client_optimizer_fn=functools.partial(tf.keras.optimizers.SGD, 0.001),
        reconstruction_optimizer_fn=functools.partial(tf.keras.optimizers.SGD,
                                                      0.001),
        client_weight_fn=client_weight_fn)

    _, train_outputs, _ = self._run_rounds(it_process, federated_data, 5)
    self.assertLess(train_outputs[-1]['loss'], train_outputs[0]['loss'])

  def test_server_update_with_inf_weight_is_noop(self):
    client_data = create_emnist_client_data()
    federated_data = [client_data()]
    client_weight_fn = lambda x: np.inf

    it_process = training_process.build_federated_reconstruction_process(
        local_recon_model_fn,
        loss_fn=tf.keras.losses.SparseCategoricalCrossentropy,
        client_optimizer_fn=functools.partial(tf.keras.optimizers.SGD, 0.001),
        reconstruction_optimizer_fn=functools.partial(tf.keras.optimizers.SGD,
                                                      0.001),
        client_weight_fn=client_weight_fn)

    state, _, initial_state = self._run_rounds(it_process, federated_data, 1)
    self.assertAllClose(state.model.trainable, initial_state.model.trainable,
                        1e-8)
    self.assertAllClose(state.model.trainable, initial_state.model.trainable,
                        1e-8)

  def test_keras_global_model(self):

    def loss_fn():
      return tf.keras.losses.SparseCategoricalCrossentropy()

    def metrics_fn():
      return [
          NumExamplesCounter(),
          NumBatchesCounter(),
          tf.keras.metrics.SparseCategoricalAccuracy()
      ]

    it_process = training_process.build_federated_reconstruction_process(
        global_recon_model_fn,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        client_optimizer_fn=functools.partial(tf.keras.optimizers.SGD, 0.001),
        reconstruction_optimizer_fn=functools.partial(tf.keras.optimizers.SGD,
                                                      0.001))

    server_state = it_process.initialize()

    client_data = create_emnist_client_data()
    federated_data = [client_data(), client_data()]

    server_states = []
    outputs = []
    loss_list = []
    for _ in range(5):
      server_state, output = it_process.next(server_state, federated_data)
      server_states.append(server_state)
      outputs.append(output)
      loss_list.append(output['loss'])

    expected_keys = [
        'sparse_categorical_accuracy', 'loss', 'num_examples_total',
        'num_batches_total'
    ]
    self.assertCountEqual(outputs[0].keys(), expected_keys)
    self.assertNotAllClose(server_states[0].model.trainable,
                           server_states[1].model.trainable)
    self.assertLess(np.mean(loss_list[2:]), np.mean(loss_list[:2]))
    self.assertEqual(outputs[0]['num_examples_total'], 6)
    self.assertEqual(outputs[1]['num_batches_total'], 4)
    self.assertEqual(outputs[0]['num_examples_total'], 6)
    self.assertEqual(outputs[1]['num_batches_total'], 4)

  def test_keras_local_layer(self):

    def loss_fn():
      return tf.keras.losses.SparseCategoricalCrossentropy()

    def metrics_fn():
      return [
          NumExamplesCounter(),
          NumBatchesCounter(),
          tf.keras.metrics.SparseCategoricalAccuracy()
      ]

    it_process = training_process.build_federated_reconstruction_process(
        local_recon_model_fn,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        client_optimizer_fn=functools.partial(tf.keras.optimizers.SGD, 0.001),
        reconstruction_optimizer_fn=functools.partial(tf.keras.optimizers.SGD,
                                                      0.001))

    server_state = it_process.initialize()

    client_data = create_emnist_client_data()
    federated_data = [client_data(), client_data()]

    server_states = []
    outputs = []
    loss_list = []
    for _ in range(5):
      server_state, output = it_process.next(server_state, federated_data)
      server_states.append(server_state)
      outputs.append(output)
      loss_list.append(output['loss'])

    expected_keys = [
        'sparse_categorical_accuracy', 'loss', 'num_examples_total',
        'num_batches_total'
    ]
    self.assertCountEqual(outputs[0].keys(), expected_keys)
    self.assertNotAllClose(server_states[0].model.trainable,
                           server_states[1].model.trainable)
    self.assertLess(np.mean(loss_list[2:]), np.mean(loss_list[:2]))
    self.assertEqual(outputs[0]['num_examples_total'], 6)
    self.assertEqual(outputs[1]['num_batches_total'], 4)
    self.assertEqual(outputs[0]['num_examples_total'], 6)
    self.assertEqual(outputs[1]['num_batches_total'], 4)

  def test_keras_local_layer_metrics_empty_list(self):

    def loss_fn():
      return tf.keras.losses.SparseCategoricalCrossentropy()

    def metrics_fn():
      return []

    it_process = training_process.build_federated_reconstruction_process(
        local_recon_model_fn,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        client_optimizer_fn=functools.partial(tf.keras.optimizers.SGD, 0.001),
        reconstruction_optimizer_fn=functools.partial(tf.keras.optimizers.SGD,
                                                      0.001))

    server_state = it_process.initialize()

    client_data = create_emnist_client_data()
    federated_data = [client_data(), client_data()]

    server_states = []
    outputs = []
    loss_list = []
    for _ in range(5):
      server_state, output = it_process.next(server_state, federated_data)
      server_states.append(server_state)
      outputs.append(output)
      loss_list.append(output['loss'])

    expected_keys = ['loss']
    self.assertCountEqual(outputs[0].keys(), expected_keys)
    self.assertLess(np.mean(loss_list[2:]), np.mean(loss_list[:2]))
    self.assertNotAllClose(server_states[0].model.trainable,
                           server_states[1].model.trainable)

  def test_keras_local_layer_metrics_none(self):

    def loss_fn():
      return tf.keras.losses.SparseCategoricalCrossentropy()

    it_process = training_process.build_federated_reconstruction_process(
        local_recon_model_fn,
        loss_fn=loss_fn,
        metrics_fn=None,
        client_optimizer_fn=functools.partial(tf.keras.optimizers.SGD, 0.001),
        reconstruction_optimizer_fn=functools.partial(tf.keras.optimizers.SGD,
                                                      0.001))

    server_state = it_process.initialize()

    client_data = create_emnist_client_data()
    federated_data = [client_data(), client_data()]

    server_states = []
    outputs = []
    loss_list = []
    for _ in range(5):
      server_state, output = it_process.next(server_state, federated_data)
      server_states.append(server_state)
      outputs.append(output)
      loss_list.append(output['loss'])

    expected_keys = ['loss']
    self.assertCountEqual(outputs[0].keys(), expected_keys)
    self.assertLess(np.mean(loss_list[2:]), np.mean(loss_list[:2]))
    self.assertNotAllClose(server_states[0].model.trainable,
                           server_states[1].model.trainable)

  def test_keras_joint_training(self):

    def loss_fn():
      return tf.keras.losses.SparseCategoricalCrossentropy()

    def metrics_fn():
      return [
          NumExamplesCounter(),
          NumBatchesCounter(),
          tf.keras.metrics.SparseCategoricalAccuracy()
      ]

    it_process = training_process.build_federated_reconstruction_process(
        local_recon_model_fn,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        client_optimizer_fn=functools.partial(tf.keras.optimizers.SGD, 0.001),
        reconstruction_optimizer_fn=functools.partial(tf.keras.optimizers.SGD,
                                                      0.001),
        jointly_train_variables=True)

    server_state = it_process.initialize()

    client_data = create_emnist_client_data()
    federated_data = [client_data(), client_data()]

    server_states = []
    outputs = []
    loss_list = []
    for _ in range(5):
      server_state, output = it_process.next(server_state, federated_data)
      server_states.append(server_state)
      outputs.append(output)
      loss_list.append(output['loss'])

    expected_keys = [
        'sparse_categorical_accuracy', 'loss', 'num_examples_total',
        'num_batches_total'
    ]
    self.assertCountEqual(outputs[0].keys(), expected_keys)
    self.assertNotAllClose(server_states[0].model.trainable,
                           server_states[1].model.trainable)
    self.assertLess(np.mean(loss_list[2:]), np.mean(loss_list[:2]))
    self.assertEqual(outputs[0]['num_examples_total'], 6)
    self.assertEqual(outputs[1]['num_batches_total'], 4)
    self.assertEqual(outputs[0]['num_examples_total'], 6)
    self.assertEqual(outputs[1]['num_batches_total'], 4)

  def test_keras_eval_reconstruction(self):

    def loss_fn():
      return tf.keras.losses.SparseCategoricalCrossentropy()

    def metrics_fn():
      return [
          NumExamplesCounter(),
          NumBatchesCounter(),
          tf.keras.metrics.SparseCategoricalAccuracy()
      ]

    it_process = training_process.build_federated_reconstruction_process(
        local_recon_model_fn,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        client_optimizer_fn=functools.partial(tf.keras.optimizers.SGD, 0.001),
        reconstruction_optimizer_fn=functools.partial(tf.keras.optimizers.SGD,
                                                      0.001),
        evaluate_reconstruction=True)

    server_state = it_process.initialize()

    client_data = create_emnist_client_data()
    federated_data = [client_data(), client_data()]

    server_states = []
    outputs = []
    loss_list = []
    for _ in range(5):
      server_state, output = it_process.next(server_state, federated_data)
      server_states.append(server_state)
      outputs.append(output)
      loss_list.append(output['loss'])

    expected_keys = [
        'sparse_categorical_accuracy', 'loss', 'num_examples_total',
        'num_batches_total'
    ]
    self.assertCountEqual(outputs[0].keys(), expected_keys)
    self.assertNotAllClose(server_states[0].model.trainable,
                           server_states[1].model.trainable)
    self.assertLess(np.mean(loss_list[2:]), np.mean(loss_list[:2]))
    self.assertEqual(outputs[0]['num_examples_total'], 12)
    self.assertEqual(outputs[1]['num_batches_total'], 8)
    self.assertEqual(outputs[0]['num_examples_total'], 12)
    self.assertEqual(outputs[1]['num_batches_total'], 8)

  def test_keras_eval_reconstruction_joint_training(self):

    def loss_fn():
      return tf.keras.losses.SparseCategoricalCrossentropy()

    def metrics_fn():
      return [
          NumExamplesCounter(),
          NumBatchesCounter(),
          tf.keras.metrics.SparseCategoricalAccuracy()
      ]

    it_process = training_process.build_federated_reconstruction_process(
        local_recon_model_fn,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        client_optimizer_fn=functools.partial(tf.keras.optimizers.SGD, 0.001),
        reconstruction_optimizer_fn=functools.partial(tf.keras.optimizers.SGD,
                                                      0.001),
        evaluate_reconstruction=True,
        jointly_train_variables=True)

    server_state = it_process.initialize()

    client_data = create_emnist_client_data()
    federated_data = [client_data(), client_data()]

    server_states = []
    outputs = []
    loss_list = []
    for _ in range(5):
      server_state, output = it_process.next(server_state, federated_data)
      server_states.append(server_state)
      outputs.append(output)
      loss_list.append(output['loss'])

    expected_keys = [
        'sparse_categorical_accuracy', 'loss', 'num_examples_total',
        'num_batches_total'
    ]
    self.assertCountEqual(outputs[0].keys(), expected_keys)
    self.assertLess(np.mean(loss_list[2:]), np.mean(loss_list[:2]))
    self.assertNotAllClose(server_states[0].model.trainable,
                           server_states[1].model.trainable)
    self.assertEqual(outputs[0]['num_examples_total'], 12)
    self.assertEqual(outputs[1]['num_batches_total'], 8)
    self.assertEqual(outputs[0]['num_examples_total'], 12)
    self.assertEqual(outputs[1]['num_batches_total'], 8)

  def test_custom_model_no_recon(self):
    client_data = create_emnist_client_data()
    train_data = [client_data(), client_data()]

    def loss_fn():
      return tf.keras.losses.SparseCategoricalCrossentropy()

    def metrics_fn():
      return [
          NumExamplesCounter(),
          NumBatchesCounter(),
          tf.keras.metrics.SparseCategoricalAccuracy()
      ]

    # Disable reconstruction via 0 learning rate to ensure post-recon loss
    # matches exact expectations round 0 and decreases by the next round.
    trainer = training_process.build_federated_reconstruction_process(
        MnistModel,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        server_optimizer_fn=functools.partial(tf.keras.optimizers.SGD, 0.01),
        client_optimizer_fn=functools.partial(tf.keras.optimizers.SGD, 0.001),
        reconstruction_optimizer_fn=functools.partial(tf.keras.optimizers.SGD,
                                                      0.0))
    state = trainer.initialize()

    outputs = []
    states = []
    for _ in range(2):
      state, output = trainer.next(state, train_data)
      outputs.append(output)
      states.append(state)

    # All weights and biases are initialized to 0, so initial logits are all 0
    # and softmax probabilities are uniform over 10 classes. So negative log
    # likelihood is -ln(1/10). This is on expectation, so increase tolerance.
    self.assertAllClose(outputs[0]['loss'], tf.math.log(10.0), rtol=1e-4)
    self.assertLess(outputs[1]['loss'], outputs[0]['loss'])
    self.assertNotAllClose(states[0].model.trainable, states[1].model.trainable)

    # Expect 6 reconstruction examples, 6 training examples. Only training
    # included in metrics.
    self.assertEqual(outputs[0]['num_examples_total'], 6.0)
    self.assertEqual(outputs[1]['num_examples_total'], 6.0)

    # Expect 4 reconstruction batches and 4 training batches. Only training
    # included in metrics.
    self.assertEqual(outputs[0]['num_batches_total'], 4.0)
    self.assertEqual(outputs[1]['num_batches_total'], 4.0)

  def test_custom_model_adagrad_server_optimizer(self):
    client_data = create_emnist_client_data()
    train_data = [client_data(), client_data()]

    def loss_fn():
      return tf.keras.losses.SparseCategoricalCrossentropy()

    def metrics_fn():
      return [
          NumExamplesCounter(),
          NumBatchesCounter(),
          tf.keras.metrics.SparseCategoricalAccuracy()
      ]

    # Disable reconstruction via 0 learning rate to ensure post-recon loss
    # matches exact expectations round 0 and decreases by the next round.
    trainer = training_process.build_federated_reconstruction_process(
        MnistModel,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        server_optimizer_fn=functools.partial(tf.keras.optimizers.Adagrad,
                                              0.01),
        client_optimizer_fn=functools.partial(tf.keras.optimizers.SGD, 0.001),
        reconstruction_optimizer_fn=functools.partial(tf.keras.optimizers.SGD,
                                                      0.0))
    state = trainer.initialize()

    outputs = []
    states = []
    for _ in range(2):
      state, output = trainer.next(state, train_data)
      outputs.append(output)
      states.append(state)

    # All weights and biases are initialized to 0, so initial logits are all 0
    # and softmax probabilities are uniform over 10 classes. So negative log
    # likelihood is -ln(1/10). This is on expectation, so increase tolerance.
    self.assertAllClose(outputs[0]['loss'], tf.math.log(10.0), rtol=1e-4)
    self.assertLess(outputs[1]['loss'], outputs[0]['loss'])
    self.assertNotAllClose(states[0].model.trainable, states[1].model.trainable)

    # Expect 6 reconstruction examples, 6 training examples. Only training
    # included in metrics.
    self.assertEqual(outputs[0]['num_examples_total'], 6.0)
    self.assertEqual(outputs[1]['num_examples_total'], 6.0)

    # Expect 4 reconstruction batches and 4 training batches. Only training
    # included in metrics.
    self.assertEqual(outputs[0]['num_batches_total'], 4.0)
    self.assertEqual(outputs[1]['num_batches_total'], 4.0)

  def test_custom_model_zeroing_clipping_aggregator_factory(self):
    client_data = create_emnist_client_data()
    train_data = [client_data(), client_data()]

    def loss_fn():
      return tf.keras.losses.SparseCategoricalCrossentropy()

    def metrics_fn():
      return [
          NumExamplesCounter(),
          NumBatchesCounter(),
          tf.keras.metrics.SparseCategoricalAccuracy()
      ]

    # No values should be clipped and zeroed
    aggregation_factory = tff.aggregators.zeroing_factory(
        zeroing_norm=float('inf'),
        inner_agg_factory=tff.aggregators.MeanFactory())

    # Disable reconstruction via 0 learning rate to ensure post-recon loss
    # matches exact expectations round 0 and decreases by the next round.
    trainer = training_process.build_federated_reconstruction_process(
        MnistModel,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        server_optimizer_fn=functools.partial(tf.keras.optimizers.SGD, 0.01),
        client_optimizer_fn=functools.partial(tf.keras.optimizers.SGD, 0.001),
        reconstruction_optimizer_fn=functools.partial(tf.keras.optimizers.SGD,
                                                      0.0),
        aggregation_factory=aggregation_factory,
    )
    state = trainer.initialize()

    outputs = []
    states = []
    for _ in range(2):
      state, output = trainer.next(state, train_data)
      outputs.append(output)
      states.append(state)

    # All weights and biases are initialized to 0, so initial logits are all 0
    # and softmax probabilities are uniform over 10 classes. So negative log
    # likelihood is -ln(1/10). This is on expectation, so increase tolerance.
    self.assertAllClose(outputs[0]['loss'], tf.math.log(10.0), rtol=1e-4)
    self.assertLess(outputs[1]['loss'], outputs[0]['loss'])
    self.assertNotAllClose(states[0].model.trainable, states[1].model.trainable)

    # Expect 6 reconstruction examples, 6 training examples. Only training
    # included in metrics.
    self.assertEqual(outputs[0]['num_examples_total'], 6.0)
    self.assertEqual(outputs[1]['num_examples_total'], 6.0)

    # Expect 4 reconstruction batches and 4 training batches. Only training
    # included in metrics.
    self.assertEqual(outputs[0]['num_batches_total'], 4.0)
    self.assertEqual(outputs[1]['num_batches_total'], 4.0)

  def test_iterative_process_builds_with_dp_agg_and_client_weight_fn(self):

    def loss_fn():
      return tf.keras.losses.SparseCategoricalCrossentropy()

    def metrics_fn():
      return [
          NumExamplesCounter(),
          NumBatchesCounter(),
          tf.keras.metrics.SparseCategoricalAccuracy()
      ]

    # No values should be changed, but working with inf directly zeroes out all
    # updates. Preferring very large value, but one that can be handled in
    # multiplication/division
    gaussian_sum_query = tfp.GaussianSumQuery(l2_norm_clip=1e10, stddev=0)
    dp_sum_factory = tff.aggregators.DifferentiallyPrivateFactory(
        query=gaussian_sum_query,
        record_aggregation_factory=tff.aggregators.SumFactory())
    dp_mean_factory = _DPMean(dp_sum_factory)

    def client_weight_fn(local_outputs):
      del local_outputs  # Unused
      return 1.0

    # Ensure this builds, as some builders raise if an unweighted aggregation is
    # specified with a client_weight_fn.
    trainer = training_process.build_federated_reconstruction_process(
        MnistModel,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        server_optimizer_fn=functools.partial(tf.keras.optimizers.SGD, 0.01),
        client_optimizer_fn=functools.partial(tf.keras.optimizers.SGD, 0.001),
        reconstruction_optimizer_fn=functools.partial(tf.keras.optimizers.SGD,
                                                      0.0),
        aggregation_factory=dp_mean_factory,
        client_weight_fn=client_weight_fn,
    )
    self.assertIsInstance(trainer, tff.templates.IterativeProcess)

  def test_execution_with_custom_dp_query(self):
    client_data = create_emnist_client_data()
    train_data = [client_data(), client_data()]

    def loss_fn():
      return tf.keras.losses.SparseCategoricalCrossentropy()

    def metrics_fn():
      return [
          NumExamplesCounter(),
          NumBatchesCounter(),
          tf.keras.metrics.SparseCategoricalAccuracy()
      ]

    # No values should be changed, but working with inf directly zeroes out all
    # updates. Preferring very large value, but one that can be handled in
    # multiplication/division
    gaussian_sum_query = tfp.GaussianSumQuery(l2_norm_clip=1e10, stddev=0)
    dp_sum_factory = tff.aggregators.DifferentiallyPrivateFactory(
        query=gaussian_sum_query,
        record_aggregation_factory=tff.aggregators.SumFactory())
    dp_mean_factory = _DPMean(dp_sum_factory)

    # Disable reconstruction via 0 learning rate to ensure post-recon loss
    # matches exact expectations round 0 and decreases by the next round.
    trainer = training_process.build_federated_reconstruction_process(
        MnistModel,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        server_optimizer_fn=functools.partial(tf.keras.optimizers.SGD, 0.01),
        client_optimizer_fn=functools.partial(tf.keras.optimizers.SGD, 0.001),
        reconstruction_optimizer_fn=functools.partial(tf.keras.optimizers.SGD,
                                                      0.0),
        aggregation_factory=dp_mean_factory,
    )
    state = trainer.initialize()

    outputs = []
    states = []
    for _ in range(2):
      state, output = trainer.next(state, train_data)
      outputs.append(output)
      states.append(state)

    # All weights and biases are initialized to 0, so initial logits are all 0
    # and softmax probabilities are uniform over 10 classes. So negative log
    # likelihood is -ln(1/10). This is on expectation, so increase tolerance.
    self.assertAllClose(outputs[0]['loss'], tf.math.log(10.0), rtol=1e-4)
    self.assertLess(outputs[1]['loss'], outputs[0]['loss'])
    self.assertNotAllClose(states[0].model.trainable, states[1].model.trainable)

    # Expect 6 reconstruction examples, 6 training examples. Only training
    # included in metrics.
    self.assertEqual(outputs[0]['num_examples_total'], 6.0)
    self.assertEqual(outputs[1]['num_examples_total'], 6.0)

    # Expect 4 reconstruction batches and 4 training batches. Only training
    # included in metrics.
    self.assertEqual(outputs[0]['num_batches_total'], 4.0)
    self.assertEqual(outputs[1]['num_batches_total'], 4.0)

  def test_custom_model_eval_reconstruction_multiple_epochs(self):
    client_data = create_emnist_client_data()
    train_data = [client_data(), client_data()]

    def loss_fn():
      return tf.keras.losses.SparseCategoricalCrossentropy()

    def metrics_fn():
      return [
          NumExamplesCounter(),
          NumBatchesCounter(),
          tf.keras.metrics.SparseCategoricalAccuracy()
      ]

    dataset_split_fn = reconstruction_utils.build_dataset_split_fn(
        recon_epochs_max=3,
        recon_epochs_constant=False,
        post_recon_epochs=4,
        post_recon_steps_max=3)
    trainer = training_process.build_federated_reconstruction_process(
        MnistModel,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        client_optimizer_fn=functools.partial(tf.keras.optimizers.SGD, 0.001),
        reconstruction_optimizer_fn=functools.partial(tf.keras.optimizers.SGD,
                                                      0.001),
        evaluate_reconstruction=True,
        dataset_split_fn=dataset_split_fn)
    state = trainer.initialize()

    outputs = []
    states = []
    for _ in range(2):
      state, output = trainer.next(state, train_data)
      outputs.append(output)
      states.append(state)

    self.assertLess(outputs[1]['loss'], outputs[0]['loss'])
    self.assertNotAllClose(states[0].model.trainable, states[1].model.trainable)

    # Expect 6 reconstruction examples, 10 training examples.
    self.assertEqual(outputs[0]['num_examples_total'], 16.0)
    # Expect 12 reconstruction examples, 10 training examples.
    self.assertEqual(outputs[1]['num_examples_total'], 22.0)

    # Expect 4 reconstruction batches and 6 training batches.
    self.assertEqual(outputs[0]['num_batches_total'], 10.0)
    # Expect 8 reconstruction batches and 6 training batches.
    self.assertEqual(outputs[1]['num_batches_total'], 14.0)

  def test_custom_model_eval_reconstruction_split_multiple_epochs(self):
    client_data = create_emnist_client_data()
    # 3 batches per user, each with one example. Since data will be split for
    # each user, each user will have 2 unique recon examples, and 1 unique
    # post-recon example (even-indices are allocated to recon during splitting).
    train_data = [client_data(batch_size=1), client_data(batch_size=1)]

    def loss_fn():
      return tf.keras.losses.SparseCategoricalCrossentropy()

    def metrics_fn():
      return [
          NumExamplesCounter(),
          NumBatchesCounter(),
          tf.keras.metrics.SparseCategoricalAccuracy()
      ]

    dataset_split_fn = reconstruction_utils.build_dataset_split_fn(
        recon_epochs_max=3, split_dataset=True, post_recon_epochs=5)
    trainer = training_process.build_federated_reconstruction_process(
        MnistModel,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        client_optimizer_fn=functools.partial(tf.keras.optimizers.SGD, 0.001),
        evaluate_reconstruction=True,
        dataset_split_fn=dataset_split_fn)
    state = trainer.initialize()

    outputs = []
    states = []
    for _ in range(2):
      state, output = trainer.next(state, train_data)
      outputs.append(output)
      states.append(state)

    self.assertLess(outputs[1]['loss'], outputs[0]['loss'])
    self.assertNotAllClose(states[0].model.trainable, states[1].model.trainable)

    # Expect 12 reconstruction examples, 10 training examples.
    self.assertEqual(outputs[0]['num_examples_total'], 22.0)
    self.assertEqual(outputs[1]['num_examples_total'], 22.0)

    # Expect 12 reconstruction batches and 10 training batches.
    self.assertEqual(outputs[0]['num_batches_total'], 22.0)
    self.assertEqual(outputs[1]['num_batches_total'], 22.0)

  def test_custom_model_eval_reconstruction_disable_post_recon(self):
    """Ensures we can disable post-recon on a client via custom `DatasetSplitFn`."""
    client_data = create_emnist_client_data()
    train_data = [client_data(batch_size=3), client_data(batch_size=2)]

    def loss_fn():
      return tf.keras.losses.SparseCategoricalCrossentropy()

    def metrics_fn():
      return [
          NumExamplesCounter(),
          NumBatchesCounter(),
          tf.keras.metrics.SparseCategoricalAccuracy()
      ]

    def dataset_split_fn(client_dataset, round_num):
      del round_num
      recon_dataset = client_dataset.repeat(2)
      # One user gets 1 batch with 1 example, the other user gets 0 batches.
      post_recon_dataset = client_dataset.skip(1)
      return recon_dataset, post_recon_dataset

    trainer = training_process.build_federated_reconstruction_process(
        MnistModel,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        client_optimizer_fn=functools.partial(tf.keras.optimizers.SGD, 0.001),
        evaluate_reconstruction=True,
        jointly_train_variables=True,
        dataset_split_fn=dataset_split_fn)
    state = trainer.initialize()

    outputs = []
    states = []
    for _ in range(2):
      state, output = trainer.next(state, train_data)
      outputs.append(output)
      states.append(state)

    # One client should still have a delta that updates the global weights, so
    # there should be a change in the server state and loss should still
    # decrease.
    self.assertLess(outputs[1]['loss'], outputs[0]['loss'])
    self.assertNotAllClose(states[0].model.trainable, states[1].model.trainable)

    # Expect 12 reconstruction examples, 1 training examples.
    self.assertEqual(outputs[0]['num_examples_total'], 13.0)
    self.assertEqual(outputs[1]['num_examples_total'], 13.0)

    # Expect 6 reconstruction batches and 1 training batches.
    self.assertEqual(outputs[0]['num_batches_total'], 7.0)
    self.assertEqual(outputs[1]['num_batches_total'], 7.0)

  def test_get_model_weights(self):
    client_data = create_emnist_client_data()
    federated_data = [client_data()]

    it_process = training_process.build_federated_reconstruction_process(
        local_recon_model_fn,
        loss_fn=tf.keras.losses.SparseCategoricalCrossentropy,
        client_optimizer_fn=functools.partial(tf.keras.optimizers.SGD, 0.001),
        reconstruction_optimizer_fn=functools.partial(tf.keras.optimizers.SGD,
                                                      0.001))
    state = it_process.initialize()

    self.assertIsInstance(
        it_process.get_model_weights(state), tff.learning.ModelWeights)
    self.assertAllClose(state.model.trainable,
                        it_process.get_model_weights(state).trainable)

    for _ in range(3):
      state, _ = it_process.next(state, federated_data)
      self.assertIsInstance(
          it_process.get_model_weights(state), tff.learning.ModelWeights)
      self.assertAllClose(state.model.trainable,
                          it_process.get_model_weights(state).trainable)


if __name__ == '__main__':
  tf.test.main()
