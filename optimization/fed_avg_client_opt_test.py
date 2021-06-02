# Copyright 2019, Google LLC.
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

import functools

from absl import logging
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from optimization import fed_avg_client_opt


def _batch_fn(batch_size=1):
  return (np.ones([batch_size, 10],
                  dtype=np.float32), np.ones([batch_size, 1], dtype=np.int64))


def _uncompiled_model_builder():
  keras_model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(
          10,
          activation='softmax',
          input_shape=(10,)
          ),
  ])
  input_spec = (tf.TensorSpec(shape=[None, 10], dtype=tf.float32),
                tf.TensorSpec(shape=[None, 1], dtype=np.int64))
  return tff.learning.from_keras_model(
      keras_model=keras_model,
      input_spec=input_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy())


class AdaptiveClientProcessTest(tf.test.TestCase, parameterized.TestCase):

  def _run_rounds(self, iterative_process, federated_data, num_rounds):
    train_outputs = []
    state = iterative_process.initialize()
    for round_num in range(num_rounds):
      state, metrics = iterative_process.next(state, federated_data)
      train_outputs.append(metrics)
      logging.info('Round %d: %s', round_num, metrics)
    return state, train_outputs

  def test_fed_avg_decreases_loss(self):
    federated_data = [[_batch_fn()]]

    client_optimizer_fn = tf.keras.optimizers.SGD
    server_optimizer_fn = tf.keras.optimizers.SGD

    iterative_process = fed_avg_client_opt.build_iterative_process(
        _uncompiled_model_builder,
        client_optimizer_fn=client_optimizer_fn,
        server_optimizer_fn=server_optimizer_fn)

    _, train_outputs = self._run_rounds(iterative_process, federated_data, 5)
    self.assertLess(train_outputs[4]['loss'], train_outputs[0]['loss'])

  @parameterized.named_parameters(
      ('adam_opt', tf.keras.optimizers.Adam),
      ('adagrad_opt', tf.keras.optimizers.Adagrad),
  )
  def test_fed_avg_with_adaptive_client_decreases_loss(self, optimizer):
    federated_data = [[_batch_fn()]]

    client_optimizer_fn = functools.partial(optimizer, epsilon=0.01)
    server_optimizer_fn = tf.keras.optimizers.SGD

    iterative_process = fed_avg_client_opt.build_iterative_process(
        _uncompiled_model_builder,
        client_optimizer_fn=client_optimizer_fn,
        server_optimizer_fn=server_optimizer_fn)

    _, train_outputs = self._run_rounds(iterative_process, federated_data, 5)
    self.assertLess(train_outputs[4]['loss'], train_outputs[0]['loss'])

  @parameterized.named_parameters(
      ('adam_opt', tf.keras.optimizers.Adam),
      ('adagrad_opt', tf.keras.optimizers.Adagrad),
  )
  def test_fed_avg_with_adaptive_client_and_server(self, optimizer):
    federated_data = [[_batch_fn()]]

    client_optimizer_fn = functools.partial(optimizer, epsilon=0.01)
    server_optimizer_fn = functools.partial(optimizer, epsilon=0.01)

    iterative_process = fed_avg_client_opt.build_iterative_process(
        _uncompiled_model_builder,
        client_optimizer_fn=client_optimizer_fn,
        server_optimizer_fn=server_optimizer_fn)
    _, train_outputs = self._run_rounds(iterative_process, federated_data, 5)
    self.assertLess(train_outputs[4]['loss'], train_outputs[0]['loss'])

  @parameterized.named_parameters(
      ('test_n1_m1', 1, 1),
      ('test_n1_m2', 1, 2),
      ('test_n1_m3', 1, 3),
      ('test_n2_m4', 2, 4),
      ('test_n3_m3', 3, 3),
      ('test_n2_m8', 2, 8),
  )
  def test_client_state_aggregate_mean(self, n, m):
    dataset1 = [_batch_fn() for _ in range(n)]
    dataset2 = [_batch_fn() for _ in range(m)]
    federated_data = [dataset1, dataset2]

    client_optimizer_fn = functools.partial(
        tf.keras.optimizers.Adam, epsilon=0.01)
    server_optimizer_fn = tf.keras.optimizers.SGD

    iterative_process = fed_avg_client_opt.build_iterative_process(
        _uncompiled_model_builder,
        client_optimizer_fn=client_optimizer_fn,
        server_optimizer_fn=server_optimizer_fn,
        client_opt_weight_fn=lambda x: 1.0)

    state = iterative_process.initialize()
    state, _ = iterative_process.next(state, federated_data)

    client_opt_iteration = state.client_optimizer_state.iterations
    self.assertEqual(client_opt_iteration, (n+m)//2)

  @parameterized.named_parameters(
      ('test_n1_m1', 1, 2),
      ('test_n1_m2', 2, 4),
      ('test_n1_m3', 3, 8),
  )
  def test_client_state_aggregate_min(self, n, m):
    dataset1 = [_batch_fn() for _ in range(n)]
    dataset2 = [_batch_fn() for _ in range(m)]
    federated_data = [dataset1, dataset2]

    client_optimizer_fn = functools.partial(
        tf.keras.optimizers.Adam, epsilon=0.01)
    server_optimizer_fn = tf.keras.optimizers.SGD

    iterative_process = fed_avg_client_opt.build_iterative_process(
        _uncompiled_model_builder,
        client_optimizer_fn=client_optimizer_fn,
        server_optimizer_fn=server_optimizer_fn,
        client_opt_weight_fn=lambda x: 1.0,
        optimizer_aggregation='min')

    state = iterative_process.initialize()
    state, _ = iterative_process.next(state, federated_data)

    client_opt_iteration = state.client_optimizer_state.iterations
    self.assertEqual(client_opt_iteration, n)

  @parameterized.named_parameters(
      ('test_n1_m1', 1, 2),
      ('test_n1_m2', 2, 4),
      ('test_n1_m3', 3, 8),
  )
  def test_client_state_aggregate_max(self, n, m):
    dataset1 = [_batch_fn() for _ in range(n)]
    dataset2 = [_batch_fn() for _ in range(m)]
    federated_data = [dataset1, dataset2]

    client_optimizer_fn = functools.partial(
        tf.keras.optimizers.Adam, epsilon=0.01)
    server_optimizer_fn = tf.keras.optimizers.SGD

    iterative_process = fed_avg_client_opt.build_iterative_process(
        _uncompiled_model_builder,
        client_optimizer_fn=client_optimizer_fn,
        server_optimizer_fn=server_optimizer_fn,
        client_opt_weight_fn=lambda x: 1.0,
        optimizer_aggregation='max')

    state = iterative_process.initialize()
    state, _ = iterative_process.next(state, federated_data)

    client_opt_iteration = state.client_optimizer_state.iterations
    self.assertEqual(client_opt_iteration, m)

  @parameterized.named_parameters(
      ('test_n1_m1', 1, 2),
      ('test_n1_m2', 2, 4),
      ('test_n1_m3', 3, 8),
  )
  def test_client_state_aggregate_sum(self, n, m):
    dataset1 = [_batch_fn() for _ in range(n)]
    dataset2 = [_batch_fn() for _ in range(m)]
    federated_data = [dataset1, dataset2]

    client_optimizer_fn = functools.partial(
        tf.keras.optimizers.Adam, epsilon=0.01)
    server_optimizer_fn = tf.keras.optimizers.SGD

    iterative_process = fed_avg_client_opt.build_iterative_process(
        _uncompiled_model_builder,
        client_optimizer_fn=client_optimizer_fn,
        server_optimizer_fn=server_optimizer_fn,
        client_opt_weight_fn=lambda x: 1.0,
        optimizer_aggregation='sum')

    state = iterative_process.initialize()
    state, _ = iterative_process.next(state, federated_data)

    client_opt_iteration = state.client_optimizer_state.iterations
    self.assertEqual(client_opt_iteration, n + m)

  def test_state_types(self):
    federated_data = [[_batch_fn()]]

    client_optimizer_fn = functools.partial(
        tf.keras.optimizers.Adam, epsilon=0.01)
    server_optimizer_fn = tf.keras.optimizers.SGD

    iterative_process = fed_avg_client_opt.build_iterative_process(
        _uncompiled_model_builder,
        client_optimizer_fn=client_optimizer_fn,
        server_optimizer_fn=server_optimizer_fn)

    state, _ = self._run_rounds(iterative_process, federated_data, 1)
    self.assertIsInstance(state, fed_avg_client_opt.ServerState)
    self.assertIsInstance(state.model, tff.learning.ModelWeights)

  def test_get_model_weights(self):
    federated_data = [[_batch_fn()]]

    iterative_process = fed_avg_client_opt.build_iterative_process(
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


if __name__ == '__main__':
  tf.test.main()
