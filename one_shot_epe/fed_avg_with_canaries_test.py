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
import unittest

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_federated as tff

from one_shot_epe import canary_insertion
from one_shot_epe import dot_product_utils
from one_shot_epe import fed_avg_with_canaries

_TEST_SEED = 0xBAD5EED


def _create_dataset():
  # Create data satisfying y = x + 1
  x = [[1.0], [2.0], [3.0]]
  y = [[2.0], [3.0], [4.0]]
  return tf.data.Dataset.from_tensor_slices((x, y)).batch(3)


def _get_input_spec():
  return _create_dataset().element_spec


def _model_fn(initializer='zeros'):
  keras_model = tf.keras.Sequential(
      [
          tf.keras.layers.Dense(
              1,
              kernel_initializer=initializer,
              bias_initializer=initializer,
              input_shape=(1,),
          )
      ]
  )
  return tff.learning.models.from_keras_model(
      keras_model=keras_model,
      input_spec=_get_input_spec(),
      loss=tf.keras.losses.MeanAbsoluteError(),
  )


def _weight_tensors_from_model(
    model: tff.learning.models.VariableModel,
) -> tff.learning.models.ModelWeights:
  return tf.nest.map_structure(
      lambda var: var.numpy(),
      tff.learning.models.ModelWeights.from_model(model),
  )


class FedAvgWithCanariesTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('no_simulation', False),
      ('with_simulation', True),
  )
  def test_client_update_real_client(self, simulation):
    client_tf = fed_avg_with_canaries._build_client_update(
        model_fn=_model_fn, use_experimental_simulation_loop=simulation
    )
    optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=1.0)
    initial_weights = _weight_tensors_from_model(_model_fn())
    dataset = _create_dataset()
    client_id = 'real:client_a'
    client_result, model_output = client_tf(
        optimizer, initial_weights, dataset, client_id, _TEST_SEED
    )

    # Loss is 2 + 3 + 4 = 9, and 3 examples.
    self.assertAllClose([9, 3], model_output['loss'])
    self.assertAllEqual(model_output['num_examples'][0], 3)
    self.assertAllEqual(model_output['num_batches'][0], 1)

    # The update should be the partial derivatives of the loss at a=b=0.
    # Since the absolute error loss is l = |ax+b-y|, we have:
    # dl/da = sign(-y)x. (-1-2-3)/3 = -2
    # dl/db = sign(-y) = -1.
    self.assertAllClose(client_result.update[0][0][0], -2)
    self.assertAllClose(client_result.update[1][0], -1)

    self.assertAllEqual(client_result.update_weight, 1.0)

  def test_client_update_canary_client(self):
    client_tf = fed_avg_with_canaries._build_client_update(model_fn=_model_fn)
    optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=1.0)
    initial_weights = _weight_tensors_from_model(_model_fn())
    dataset = _create_dataset()
    client_id = 'canary:0'
    client_result, model_output = client_tf(
        optimizer, initial_weights, dataset, client_id, _TEST_SEED
    )

    update_vec = tf.concat(
        [tf.reshape(t, [-1]) for t in client_result.update], axis=0
    )
    self.assertAllClose(
        tf.linalg.norm(update_vec), fed_avg_with_canaries._CANARY_SCALING_FACTOR
    )
    self.assertAllEqual(client_result.update_weight, 1.0)

    self.assertAllClose([0, 0], model_output['loss'])
    self.assertAllEqual(model_output['num_examples'][0], 0)
    self.assertAllEqual(model_output['num_batches'][0], 0)

  @parameterized.named_parameters(('sim', True), ('no_sim', False))
  @unittest.mock.patch.object(fed_avg_with_canaries, '_iter_reduce')
  @unittest.mock.patch.object(fed_avg_with_canaries, '_dataset_reduce')
  def test_client_tf_dataset_reduce_fn(
      self, simulation, mock_dataset_reduce, mock_iter_reduce
  ):
    client_tf = fed_avg_with_canaries._build_client_update(
        model_fn=_model_fn, use_experimental_simulation_loop=simulation
    )
    optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=1.0)
    initial_weights = _weight_tensors_from_model(_model_fn())
    dataset = _create_dataset()
    client_id = 'real:0'
    client_tf(optimizer, initial_weights, dataset, client_id, _TEST_SEED)
    if simulation:
      mock_dataset_reduce.assert_not_called()
      mock_iter_reduce.assert_called_once()
    else:
      mock_dataset_reduce.assert_called_once()
      mock_iter_reduce.assert_not_called()

  def test_execution_with_optimizer(self):
    optimizer_fn = lambda: tf.keras.optimizers.legacy.SGD(learning_rate=1.0)
    client_work_process = fed_avg_with_canaries._build_canary_client_work(
        _model_fn, optimizer_fn, _TEST_SEED
    )
    client_data = [_create_dataset()] * 2
    client_model_weights = [
        (_weight_tensors_from_model(_model_fn()), 'real:0'),
        (_weight_tensors_from_model(_model_fn()), 'canary:0'),
    ]
    state = client_work_process.initialize()
    output = client_work_process.next(state, client_model_weights, client_data)

    # The update should be the partial derivatives of the loss at a=b=0.
    # Since the absolute error loss is l = |ax+b-y|, we have:
    # dl/da = sign(-y)x. (-1-2-3)/3 = -2
    # dl/db = sign(-y) = -1.
    real_client_result = output.result[0]
    self.assertAllClose(real_client_result.update[0][0][0], -2)
    self.assertAllClose(real_client_result.update[1][0], -1)
    self.assertAllEqual(real_client_result.update_weight, 1.0)

    canary_client_result = output.result[1]
    canary_update_vec = tf.concat(
        [tf.reshape(t, [-1]) for t in canary_client_result.update], axis=0
    )
    self.assertAllClose(
        tf.linalg.norm(canary_update_vec),
        fed_avg_with_canaries._CANARY_SCALING_FACTOR,
    )
    self.assertAllClose(canary_client_result.update_weight, 1.0)

    self.assertCountEqual(output.measurements, ['train'])

  @parameterized.named_parameters(('sim', True), ('no_sim', False))
  @unittest.mock.patch.object(fed_avg_with_canaries, '_iter_reduce')
  @unittest.mock.patch.object(fed_avg_with_canaries, '_dataset_reduce')
  def test_execution_dataset_reduce(
      self, simulation, mock_dataset_reduce, mock_iter_reduce
  ):
    optimizer_fn = lambda: tf.keras.optimizers.legacy.SGD(learning_rate=1.0)
    client_work_process = fed_avg_with_canaries._build_canary_client_work(
        _model_fn,
        optimizer_fn,
        _TEST_SEED,
        use_experimental_simulation_loop=simulation,
    )
    client_data = [_create_dataset()] * 2
    client_model_weights = [
        (_weight_tensors_from_model(_model_fn()), 'real:0'),
        (_weight_tensors_from_model(_model_fn()), 'canary:0'),
    ]
    state = client_work_process.initialize()
    client_work_process.next(state, client_model_weights, client_data)

    if simulation:
      mock_dataset_reduce.assert_not_called()
      mock_iter_reduce.assert_called_once()
    else:
      mock_dataset_reduce.assert_called_once()
      mock_iter_reduce.assert_not_called()

  def test_learning_process_convergence_real_clients(self):
    num_rounds = 30
    num_clients = 3

    def client_optimizer_fn():
      return tf.keras.optimizers.legacy.SGD(learning_rate=0.1)

    lr_decay = tf.keras.optimizers.schedules.PolynomialDecay(
        1.0, num_rounds, 0.2
    )

    def server_optimizer_fn():
      return tf.keras.optimizers.legacy.SGD(learning_rate=lr_decay)

    client_ids = [f'real:{i}' for i in range(num_clients)]
    train_data = tff.simulation.datasets.TestClientData.from_clients_and_tf_fn(
        client_ids, lambda x: _create_dataset()
    )
    agg_factory = tff.aggregators.UnweightedMeanFactory()
    learning_process = fed_avg_with_canaries.build_canary_learning_process(
        model_fn=_model_fn,
        dataset_computation=train_data.dataset_computation,
        canary_seed=_TEST_SEED,
        num_canaries=0,
        num_unseen_canaries=0,
        update_aggregator_factory=agg_factory,
        client_optimizer_fn=client_optimizer_fn,
        server_optimizer_fn=server_optimizer_fn,
    )
    state = learning_process.initialize()
    first_loss, final_loss = None, None
    for i in range(num_rounds):
      selected_clients = client_ids.copy()
      selected_clients.remove(client_ids[i % num_clients])
      output = learning_process.next(state, selected_clients)
      state = output.state
      loss = output.metrics['client_work']['train']['loss']
      if i == 0:
        first_loss = loss
      else:
        final_loss = loss
    weights = learning_process.get_model_weights(state).trainable
    final_model = tf.concat([tf.reshape(t, [-1]) for t in weights], axis=0)
    self.assertAllClose(final_model, [1, 1], atol=2e-1)
    self.assertLess(final_loss, first_loss)

  def test_learning_process_with_canary(self):
    num_canaries = 3
    num_unseen_canaries = 2
    weights = _weight_tensors_from_model(_model_fn()).trainable
    canary_updates = [
        dot_product_utils.packed_canary(weights, i, _TEST_SEED)
        for i in range(num_canaries)
    ]
    optimizer_fn = lambda: tf.keras.optimizers.legacy.SGD(learning_rate=1.0)
    # Use DP factory for clipping. Canary updates are scaled to be huge.
    agg_factory = tff.aggregators.DifferentiallyPrivateFactory.gaussian_fixed(
        noise_multiplier=0.0, clients_per_round=1, clip=1.0
    )

    @tf.function
    def make_dataset(x):
      del x
      return _create_dataset().unbatch()

    train_data = tff.simulation.datasets.TestClientData.from_clients_and_tf_fn(
        ['0', '1'], make_dataset
    )
    train_data = canary_insertion.add_canaries(train_data, num_canaries)
    train_data = train_data.preprocess(lambda d: d.batch(3))
    learning_process = fed_avg_with_canaries.build_canary_learning_process(
        _model_fn,
        train_data.dataset_computation,
        _TEST_SEED,
        num_canaries,
        num_unseen_canaries,
        agg_factory,
        optimizer_fn,
        optimizer_fn,
    )

    # First round with canary:0. Weights should move in the negative direction
    # of canary:0.
    state = learning_process.initialize()
    output = learning_process.next(state, ['canary:0'])
    tf.nest.map_structure(
        lambda w, c: self.assertAllClose(w, -c),
        learning_process.get_model_weights(output.state).trainable,
        canary_updates[0],
    )
    train_metrics = output.metrics['client_work']['train']
    self.assertAllClose(train_metrics['loss'], 0.0)
    self.assertAllClose(train_metrics['num_examples'], 0)
    self.assertAllClose(train_metrics['num_batches'], 0)
    max_cosines = output.state.max_canary_model_delta_cosines
    self.assertLen(max_cosines, num_canaries)
    self.assertAllClose(max_cosines[0], 1.0)
    for i in [1, 2]:
      self.assertAllClose(
          max_cosines[i],
          dot_product_utils.compute_cosine(canary_updates[0], i, _TEST_SEED),
      )
    # Unseen canaries will have arbitrary cosine values.
    max_unseen_cosines = output.state.max_unseen_canary_model_delta_cosines
    self.assertLen(max_unseen_cosines, num_unseen_canaries)
    for i in range(num_unseen_canaries):
      for x in [-1.0, 0.0, 1.0]:
        self.assertNotAllClose(max_unseen_cosines[i], x)

    # Second round with canary:1. Weights are now negative sum of canary:0 from
    # the first round and canary:1 from the second round.
    output = learning_process.next(output.state, ['canary:1'])
    tf.nest.map_structure(
        lambda w, c0, c1: self.assertAllClose(w, -c0 - c1),
        learning_process.get_model_weights(output.state).trainable,
        canary_updates[0],
        canary_updates[1],
    )
    train_metrics = output.metrics['client_work']['train']
    self.assertAllClose(train_metrics['loss'], 0.0)
    self.assertAllClose(train_metrics['num_examples'], 0)
    self.assertAllClose(train_metrics['num_batches'], 0)

    next_max_cosines = output.state.max_canary_model_delta_cosines
    self.assertAllGreaterEqual(next_max_cosines - max_cosines, 0)
    max_cosines = next_max_cosines
    next_max_unseen_cosines = output.state.max_unseen_canary_model_delta_cosines
    self.assertAllGreaterEqual(next_max_unseen_cosines - max_unseen_cosines, 0)
    max_unseen_cosines = next_max_unseen_cosines

    # Third round with canary:0 and canary:2.
    output = learning_process.next(output.state, ['canary:2', 'canary:0'])
    tf.nest.map_structure(
        lambda w, c0, c1, c2: self.assertAllClose(w, -2 * c0 - c1 - c2),
        learning_process.get_model_weights(output.state).trainable,
        canary_updates[0],
        canary_updates[1],
        canary_updates[2],
    )
    train_metrics = output.metrics['client_work']['train']
    self.assertAllClose(train_metrics['loss'], 0.0)
    self.assertAllClose(train_metrics['num_examples'], 0)
    self.assertAllClose(train_metrics['num_batches'], 0)

    next_max_cosines = output.state.max_canary_model_delta_cosines
    self.assertAllGreaterEqual(next_max_cosines - max_cosines, 0.0)
    next_max_unseen_cosines = output.state.max_unseen_canary_model_delta_cosines
    self.assertAllGreaterEqual(next_max_unseen_cosines - max_unseen_cosines, 0)


if __name__ == '__main__':
  tff.backends.native.set_sync_local_cpp_execution_context()
  tf.test.main()
