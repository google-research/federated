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
"""End-to-end example testing Federated Averaging."""

import collections
from collections.abc import Callable
import functools

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from multi_epoch_dp_matrix_factorization import tff_aggregator
from multi_epoch_dp_matrix_factorization.dp_ftrl import dp_fedavg


def _create_test_cnn_model():
  """A simple CNN model for test."""
  data_format = 'channels_last'
  input_shape = [28, 28, 1]

  max_pool = functools.partial(
      tf.keras.layers.MaxPooling2D,
      pool_size=(2, 2),
      padding='same',
      data_format=data_format,
  )
  conv2d = functools.partial(
      tf.keras.layers.Conv2D,
      kernel_size=5,
      padding='same',
      data_format=data_format,
      activation=tf.nn.relu,
  )

  model = tf.keras.models.Sequential([
      conv2d(filters=32, input_shape=input_shape),
      max_pool(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(10),
      tf.keras.layers.Activation(tf.nn.softmax),
  ])

  return model


def _create_random_batch():
  return collections.OrderedDict(
      x=tf.random.uniform(tf.TensorShape([1, 28, 28, 1]), dtype=tf.float32),
      y=tf.constant(1, dtype=tf.int32, shape=[1]),
  )


def _simple_fedavg_model_fn():
  keras_model = _create_test_cnn_model()
  loss = tf.keras.losses.SparseCategoricalCrossentropy()
  input_spec = collections.OrderedDict(
      x=tf.TensorSpec([None, 28, 28, 1], tf.float32),
      y=tf.TensorSpec([None], tf.int32),
  )
  return dp_fedavg.KerasModelWrapper(
      keras_model=keras_model, input_spec=input_spec, loss=loss
  )


def _tff_learning_model_fn():
  keras_model = _create_test_cnn_model()
  loss = tf.keras.losses.SparseCategoricalCrossentropy()
  input_spec = collections.OrderedDict(
      x=tf.TensorSpec([None, 28, 28, 1], tf.float32),
      y=tf.TensorSpec([None], tf.int32),
  )
  return tff.learning.from_keras_model(
      keras_model=keras_model, input_spec=input_spec, loss=loss
  )


MnistVariables = collections.namedtuple(
    'MnistVariables', 'weights bias num_examples loss_sum accuracy_sum'
)


def _create_mnist_variables():
  return MnistVariables(
      weights=tf.Variable(
          lambda: tf.zeros(dtype=tf.float32, shape=(784, 10)),
          name='weights',
          trainable=True,
      ),
      bias=tf.Variable(
          lambda: tf.zeros(dtype=tf.float32, shape=(10)),
          name='bias',
          trainable=True,
      ),
      num_examples=tf.Variable(0.0, name='num_examples', trainable=False),
      loss_sum=tf.Variable(0.0, name='loss_sum', trainable=False),
      accuracy_sum=tf.Variable(0.0, name='accuracy_sum', trainable=False),
  )


def _mnist_predict_on_batch(variables, batch):
  y = tf.nn.softmax(tf.matmul(batch, variables.weights) + variables.bias)
  predictions = tf.cast(tf.argmax(y, 1), tf.int32)
  return y, predictions


def _mnist_forward_pass(variables, batch):
  y = tf.nn.softmax(tf.matmul(batch['x'], variables.weights) + variables.bias)
  predictions = tf.cast(tf.argmax(y, 1), tf.int32)

  flat_labels = tf.reshape(batch['y'], [-1])
  loss = -tf.reduce_mean(
      tf.reduce_sum(tf.one_hot(flat_labels, 10) * tf.math.log(y), axis=[1])
  )
  accuracy = tf.reduce_mean(
      tf.cast(tf.equal(predictions, flat_labels), tf.float32)
  )

  num_examples = tf.cast(tf.size(batch['y']), tf.float32)

  variables.num_examples.assign_add(num_examples)
  variables.loss_sum.assign_add(loss * num_examples)
  variables.accuracy_sum.assign_add(accuracy * num_examples)

  return tff.learning.BatchOutput(
      loss=loss, predictions=predictions, num_examples=num_examples
  )


class MnistModel(tff.learning.models.VariableModel):

  def __init__(self):
    self._variables = _create_mnist_variables()

  @property
  def trainable_variables(self):
    return [self._variables.weights, self._variables.bias]

  @property
  def non_trainable_variables(self):
    return []

  @property
  def weights(self):
    return tff.learning.models.ModelWeights(
        trainable=self.trainable_variables,
        non_trainable=self.non_trainable_variables,
    )

  @property
  def local_variables(self):
    return [
        self._variables.num_examples,
        self._variables.loss_sum,
        self._variables.accuracy_sum,
    ]

  @property
  def input_spec(self):
    return collections.OrderedDict(
        x=tf.TensorSpec([None, 784], tf.float32),
        y=tf.TensorSpec([None, 1], tf.int32),
    )

  @tf.function
  def predict_on_batch(self, batch, training=True):
    del training
    return _mnist_predict_on_batch(self._variables, batch)

  @tf.function
  def forward_pass(self, batch, training=True):
    del training
    return _mnist_forward_pass(self._variables, batch)

  @tf.function
  def report_local_unfinalized_metrics(
      self,
  ) -> dict[str, list[tf.Tensor]]:
    """Creates an `OrderedDict` of metric names to unfinalized values."""
    return collections.OrderedDict(
        num_examples=[self._variables.num_examples],
        loss=[self._variables.loss_sum, self._variables.num_examples],
        accuracy=[self._variables.accuracy_sum, self._variables.num_examples],
    )

  def metric_finalizers(
      self,
  ) -> dict[str, Callable[[list[tf.Tensor]], tf.Tensor]]:
    """Creates an `OrderedDict` of metric names to finalizers."""
    return collections.OrderedDict(
        num_examples=tf.function(func=lambda x: x[0]),
        loss=tf.function(func=lambda x: x[0] / x[1]),
        accuracy=tf.function(func=lambda x: x[0] / x[1]),
    )

  @tf.function
  def reset_metrics(self):
    """Resets metrics variables to initial value."""
    raise NotImplementedError(
        "The `reset_metrics` method isn't implemented for your custom"
        ' `tff.learning.models.VariableModel`. Please implement it before using'
        " this method. You can leave this method unimplemented if you won't use"
        ' this method.'
    )


def _create_client_data():
  emnist_batch = collections.OrderedDict(
      label=[[5]], pixels=[np.random.rand(28, 28).astype(np.float32)]
  )
  dataset = tf.data.Dataset.from_tensor_slices(emnist_batch)

  def client_data():
    return (
        tff.simulation.models.mnist.keras_dataset_from_emnist(dataset)
        .repeat(2)
        .batch(2)
    )

  return client_data


def _create_test_rnn_model(
    vocab_size: int = 6,
    sequence_length: int = 5,
    mask_zero: bool = True,
    seed: int = 1,
) -> tf.keras.Model:
  """A simple RNN model for test."""
  initializer = tf.keras.initializers.GlorotUniform(seed=seed)
  model = tf.keras.Sequential()
  model.add(
      tf.keras.layers.Embedding(
          input_dim=vocab_size,
          input_length=sequence_length,
          output_dim=8,
          mask_zero=mask_zero,
          embeddings_initializer=initializer,
      )
  )
  model.add(
      tf.keras.layers.LSTM(
          units=16,
          kernel_initializer=initializer,
          recurrent_initializer='zeros',
          return_sequences=True,
          stateful=False,
      )
  )
  model.add(tf.keras.layers.Dense(vocab_size, kernel_initializer=initializer))
  return model


def _create_rnn_model_fn(use_tff_learning=True):
  def _rnn_model_fn():
    keras_model = _create_test_rnn_model()
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    input_spec = collections.OrderedDict(
        x=tf.TensorSpec([None, 5], tf.int32),
        y=tf.TensorSpec([None, 5], tf.int32),
    )
    if use_tff_learning:
      return tff.learning.from_keras_model(
          keras_model=keras_model, input_spec=input_spec, loss=loss
      )
    else:
      return dp_fedavg.KerasModelWrapper(
          keras_model=keras_model, input_spec=input_spec, loss=loss
      )

  return _rnn_model_fn


class TFFLearningDPFTRLTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('default', True, False, 0.1, 0.9),
      ('nomomentum', True, False, 0.1, 0.0),
      ('nomomentum_nonoise', True, False, 0.0, 0.0),
      ('reduce_nonoise', False, False, 0.0, 0.9),
      ('nesterov_nonoise', True, True, 0.0, 0.9),
      ('nesterov', True, True, 0.1, 0.9),
  )
  def test_dpftrl_training_with_factorized_matrix(
      self, simulation_flag, use_nesterov, noise_multiplier, momentum
  ):
    total_rounds, learning_rate, clip_norm, clients_per_round, seed = (
        10,
        0.1,
        1.0,
        1,
        1,
    )

    model_fn = _create_rnn_model_fn()

    clear_aggregator = (
        tff.aggregators.DifferentiallyPrivateFactory.gaussian_fixed(
            noise_multiplier=0.0,
            clients_per_round=clients_per_round,
            clip=clip_norm,
        )
    )

    clear_process = dp_fedavg.build_dpftrl_fedavg_process(
        model_fn,
        server_learning_rate=learning_rate,
        server_momentum=momentum,
        server_nesterov=use_nesterov,
        use_experimental_simulation_loop=simulation_flag,
        dp_aggregator_factory=clear_aggregator,
    )

    gaussian_aggregator = (
        tff.aggregators.DifferentiallyPrivateFactory.gaussian_fixed(
            noise_multiplier=noise_multiplier,
            clients_per_round=clients_per_round,
            clip=clip_norm,
        )
    )

    gaussian_process = dp_fedavg.build_dpftrl_fedavg_process(
        model_fn,
        server_learning_rate=learning_rate,
        server_momentum=momentum,
        server_nesterov=use_nesterov,
        use_experimental_simulation_loop=simulation_flag,
        dp_aggregator_factory=gaussian_aggregator,
    )

    s_matrix = tf.constant(
        np.tril(np.ones(shape=[total_rounds, total_rounds], dtype=np.float64))
    )
    h_matrix = tf.eye(total_rounds, dtype=np.float64)
    model_weight_specs = tff.types.type_to_tf_tensor_specs(
        tff.learning.models.weights_type_from_model(model_fn).trainable
    )

    # This parameterization should be equivalent to the Gaussian aggregator with
    # identical parameters. We will assert that the average difference between
    # these two implementations and the clear trajectory are close.
    matrix_gaussian_aggregator = (
        tff_aggregator.create_residual_prefix_sum_dp_factory(
            tensor_specs=model_weight_specs,
            l2_norm_clip=clip_norm,
            noise_multiplier=noise_multiplier,
            w_matrix=s_matrix,
            h_matrix=h_matrix,
            clients_per_round=clients_per_round,
            seed=seed,
        )
    )

    matrix_gaussian_process = dp_fedavg.build_dpftrl_fedavg_process(
        model_fn,
        server_learning_rate=learning_rate,
        server_momentum=momentum,
        server_nesterov=use_nesterov,
        use_experimental_simulation_loop=simulation_flag,
        dp_aggregator_factory=matrix_gaussian_aggregator,
    )

    def deterministic_batch():
      return collections.OrderedDict(
          x=np.array([[0, 1, 2, 3, 4]], dtype=np.int32),
          y=np.array([[1, 2, 3, 4, 0]], dtype=np.int32),
      )

    batch = tff.tf_computation(deterministic_batch)()
    federated_data = [[batch]]

    clear_state = clear_process.initialize()
    gaussian_state = gaussian_process.initialize()
    matrix_state = matrix_gaussian_process.initialize()
    # Ensure model initializations are identical.
    initial_weights = clear_process.get_model_weights(clear_state)
    gaussian_state = gaussian_process.set_model_weights(
        gaussian_state, initial_weights
    )
    matrix_state = matrix_gaussian_process.set_model_weights(
        matrix_state, initial_weights
    )

    gaussian_norm_deltas = []
    matrix_norm_deltas = []
    for _ in range(total_rounds):
      clear_next_output = clear_process.next(clear_state, federated_data)
      clear_state = clear_next_output.state
      gaussian_next_output = gaussian_process.next(
          gaussian_state, federated_data
      )
      gaussian_state = gaussian_next_output.state
      matrix_gaussian_next_output = matrix_gaussian_process.next(
          matrix_state, federated_data
      )
      matrix_state = matrix_gaussian_next_output.state

      clear_weights = clear_process.get_model_weights(clear_state)
      gaussian_weights = gaussian_process.get_model_weights(gaussian_state)
      matrix_weights = matrix_gaussian_process.get_model_weights(matrix_state)

      gaussian_deltas = tf.nest.map_structure(
          lambda x, y: x - y,
          clear_weights.trainable,
          gaussian_weights.trainable,
      )
      matrix_deltas = tf.nest.map_structure(
          lambda x, y: x - y, clear_weights.trainable, matrix_weights.trainable
      )
      gaussian_norm_deltas.append(
          tf.linalg.global_norm(tf.nest.flatten(gaussian_deltas))
      )
      matrix_norm_deltas.append(
          tf.linalg.global_norm(tf.nest.flatten(matrix_deltas))
      )

    self.assertAllClose(
        np.mean(gaussian_norm_deltas), np.mean(matrix_norm_deltas), rtol=0.05
    )


if __name__ == '__main__':
  tf.test.main()
