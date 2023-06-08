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
import asyncio
import collections
import csv
import os

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from multi_epoch_dp_matrix_factorization.dp_ftrl import dp_fedavg
from multi_epoch_dp_matrix_factorization.dp_ftrl import training_loop

_Batch = collections.namedtuple('Batch', ['x', 'y'])


def _build_federated_averaging_process() -> (
    tff.learning.templates.LearningProcess
):
  return dp_fedavg.build_dpftrl_fedavg_process(
      _uncompiled_model_fn,
      client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1),
  )


def _uncompiled_model_fn():
  keras_model = tff.simulation.models.mnist.create_keras_model(
      compile_model=False
  )
  input_spec = _create_input_spec()
  return tff.learning.from_keras_model(
      keras_model=keras_model,
      input_spec=input_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
  )


def _batch_fn():
  batch = _Batch(
      x=np.ones([1, 784], dtype=np.float32), y=np.ones([1, 1], dtype=np.int64)
  )
  return batch


def _create_input_spec():
  return _Batch(
      x=tf.TensorSpec(shape=[None, 784], dtype=tf.float32),
      y=tf.TensorSpec(dtype=tf.int64, shape=[None, 1]),
  )


def _read_from_csv(file_name):
  """Returns a list of fieldnames and a list of metrics from a given CSV."""
  with tf.io.gfile.GFile(file_name, 'r') as csv_file:
    reader = csv.DictReader(csv_file)
    fieldnames = reader.fieldnames
    csv_metrics = list(reader)
  return fieldnames, csv_metrics


class ExperimentRunnerTest(tf.test.TestCase):

  def test_raises_non_iterative_process(self):
    bad_iterative_process = _build_federated_averaging_process().next
    federated_data = [[_batch_fn()]]

    def client_datasets_fn(round_num):
      del round_num
      return federated_data, 0

    def validation_fn(model):
      del model
      return {}

    root_output_dir = self.get_temp_dir()
    with self.assertRaises(TypeError):
      training_loop.run(
          iterative_process=[bad_iterative_process],
          client_datasets_fn=client_datasets_fn,
          validation_fn=validation_fn,
          total_epochs=1,
          total_rounds=10,
          run_name='non_iterative_process',
          root_output_dir=root_output_dir,
      )

  def test_raises_non_callable_client_dataset(self):
    iterative_process = _build_federated_averaging_process()
    client_dataset = [[_batch_fn()]]

    def validation_fn(model):
      del model
      return {}

    root_output_dir = self.get_temp_dir()
    with self.assertRaises(TypeError):
      training_loop.run(
          iterative_process=iterative_process,
          client_datasets_fn=client_dataset,
          validation_fn=validation_fn,
          total_epochs=1,
          total_rounds=10,
          run_name='non_callable_client_dataset',
          root_output_dir=root_output_dir,
      )

  def test_raises_non_callable_evaluate_fn(self):
    iterative_process = _build_federated_averaging_process()
    federated_data = [[_batch_fn()]]

    def client_datasets_fn(round_num):
      del round_num
      return federated_data, 0

    metrics_dict = {}
    root_output_dir = self.get_temp_dir()
    with self.assertRaises(TypeError):
      training_loop.run(
          iterative_process=iterative_process,
          client_datasets_fn=client_datasets_fn,
          validation_fn=metrics_dict,
          total_epochs=1,
          total_rounds=10,
          run_name='non_callable_evaluate',
          root_output_dir=root_output_dir,
      )

  def test_raises_non_str_output_dir(self):
    iterative_process = _build_federated_averaging_process()
    federated_data = [[_batch_fn()]]

    def client_datasets_fn(round_num):
      del round_num
      return federated_data, 0

    def validation_fn(model):
      del model
      return {}

    with self.assertRaises(TypeError):
      training_loop.run(
          iterative_process=iterative_process,
          client_datasets_fn=client_datasets_fn,
          validation_fn=validation_fn,
          total_epochs=1,
          total_rounds=10,
          run_name='non_str_output_dir',
          root_output_dir=1,
      )

  def test_fedavg_training_decreases_loss(self):
    batch = _batch_fn()
    federated_data = [[batch]]
    iterative_process = _build_federated_averaging_process()

    def client_datasets_fn(round_num):
      del round_num
      return federated_data, 0

    def validation_fn(model):
      keras_model = tff.simulation.models.mnist.create_keras_model(
          compile_model=True
      )
      model.assign_weights_to(keras_model)
      return {'loss': keras_model.evaluate(batch.x, batch.y)}

    initial_state = iterative_process.initialize()

    root_output_dir = self.get_temp_dir()
    final_state = training_loop.run(
        iterative_process=iterative_process,
        client_datasets_fn=client_datasets_fn,
        validation_fn=validation_fn,
        total_epochs=1,
        total_rounds=10,
        run_name='fedavg_decreases_loss',
        root_output_dir=root_output_dir,
    )

    self.assertLess(
        validation_fn(iterative_process.get_model_weights(final_state))['loss'],
        validation_fn(iterative_process.get_model_weights(initial_state))[
            'loss'
        ],
    )

  def test_checkpoint_manager_saves_state(self):
    loop = asyncio.get_event_loop()
    run_name = 'checkpoint_manager_saves_state'
    iterative_process = _build_federated_averaging_process()
    federated_data = [[_batch_fn()]]

    client_seed = 5

    def client_datasets_fn(round_num):
      del round_num
      return federated_data, 0

    def validation_fn(model):
      del model
      return {}

    root_output_dir = self.get_temp_dir()
    final_state = training_loop.run(
        iterative_process=iterative_process,
        client_datasets_fn=client_datasets_fn,
        validation_fn=validation_fn,
        total_epochs=1,
        total_rounds=1,
        run_name=run_name,
        root_output_dir=root_output_dir,
        clients_seed=client_seed,
    )

    program_state_manager = tff.program.FileProgramStateManager(
        os.path.join(root_output_dir, 'checkpoints', run_name)
    )
    restored_state, restored_round = loop.run_until_complete(
        program_state_manager.load_latest((final_state, 0))
    )

    self.assertEqual(restored_round, 0)

    keras_model = tff.simulation.models.mnist.create_keras_model(
        compile_model=True
    )
    model_weights = iterative_process.get_model_weights(restored_state[0])
    model_weights.assign_weights_to(keras_model)
    restored_loss = keras_model.test_on_batch(
        federated_data[0][0].x, federated_data[0][0].y
    )
    final_model_weights = iterative_process.get_model_weights(final_state)
    final_model_weights.assign_weights_to(keras_model)
    final_loss = keras_model.test_on_batch(
        federated_data[0][0].x, federated_data[0][0].y
    )
    self.assertEqual(final_loss, restored_loss)
    # We persist the seed we saw or generated to ensure that we can restore the
    # sampling procedure as appropriate.
    self.assertEqual(restored_state[1], client_seed)

  def test_fn_writes_metrics(self):
    run_name = 'test_metrics'
    iterative_process = _build_federated_averaging_process()
    batch = _batch_fn()
    federated_data = [[batch]]

    def client_datasets_fn(round_num):
      del round_num
      return federated_data, 0

    def test_fn(model):
      keras_model = tff.simulation.models.mnist.create_keras_model(
          compile_model=True
      )
      model.assign_weights_to(keras_model)
      return {'loss': keras_model.evaluate(batch.x, batch.y)}

    def validation_fn(model):
      del model
      return {}

    root_output_dir = self.get_temp_dir()
    training_loop.run(
        iterative_process=iterative_process,
        client_datasets_fn=client_datasets_fn,
        validation_fn=validation_fn,
        total_epochs=1,
        total_rounds=1,
        run_name=run_name,
        root_output_dir=root_output_dir,
        rounds_per_eval=10,
        test_fn=test_fn,
    )

    csv_file = os.path.join(
        root_output_dir, 'results', run_name, 'experiment.metrics.csv'
    )
    fieldnames, metrics = _read_from_csv(csv_file)
    self.assertLen(metrics, 2)
    self.assertIn('test/loss', fieldnames)
    self.assertContainsSubset(
        [
            'test/loss',
            'train/model_l2_norm',
            'train/client_work/train/num_batches',
        ],
        fieldnames,
    )


class ClientIDShufflerTest(tf.test.TestCase, parameterized.TestCase):

  def test_raises_epochs_going_backwards(self):
    client_ids = list(range(3))
    shuffler = training_loop.ClientIDShuffler(1, client_ids, seed=1234)
    _, epoch = shuffler(3)  # Round 3 belongs to  epoch 1
    self.assertEqual(epoch, 1)
    # We raise whenever were asked for an epoch smaller than one we've already
    # started yielding samples from in this process.
    with self.assertRaises(ValueError):
      shuffler(2)  # Belongs to epoch 0

  @parameterized.named_parameters(('reshuffle', True), ('', False))
  def test_epoch_0_shuffling(self, reshuffle):
    num_clients = 5
    client_ids = list(range(num_clients))
    shuffler = training_loop.ClientIDShuffler(
        clients_per_round=1,
        client_ids=client_ids,
        seed=123,
        reshuffle_each_epoch=reshuffle,
    )
    # All clients seen in one epoch:
    clients_in_epoch = []
    for i in range(num_clients):
      round_clients = shuffler(i)[0]
      clients_in_epoch.extend(round_clients)
    # We should see all 5 clients, but not in unshuffled order
    # (this test would be flaky without a fixed seed):
    self.assertLen(clients_in_epoch, 5)
    self.assertNotEqual(clients_in_epoch, client_ids)
    self.assertCountEqual(clients_in_epoch, client_ids)

  def test_shuffling(self):
    client_ids = list(range(5))
    shuffler = training_loop.ClientIDShuffler(
        clients_per_round=1,
        client_ids=client_ids,
        seed=123,
        reshuffle_each_epoch=True,
    )
    epoch2clientid = collections.defaultdict(list)
    for round_num in range(2 * len(client_ids)):  # Two epochs.
      clients, epoch = shuffler(round_num)
      epoch2clientid[epoch].extend(clients)

    # We should have shuffled the clients differently for the two
    # epochs because reshuffle_each_epoch=True
    self.assertNotEqual(epoch2clientid[0], epoch2clientid[1])
    self.assertLen(epoch2clientid[0], 5)
    self.assertLen(epoch2clientid[0], 5)

  @parameterized.named_parameters(
      ('reshuffle_10', 10, True),
      ('reshuffle_11', 11, True),
      ('10', 10, False),
      ('11', 11, False),
  )
  def test_deterministic_sequence_generated_with_seed(self, n, reshuffle):
    client_ids = list(range(n))
    clients_per_round = 5
    shuffler1 = training_loop.ClientIDShuffler(
        clients_per_round, client_ids, seed=0, reshuffle_each_epoch=reshuffle
    )
    shuffler2 = training_loop.ClientIDShuffler(
        clients_per_round, client_ids, seed=0, reshuffle_each_epoch=reshuffle
    )
    self.assertEqual(shuffler1.rounds_per_epoch, 2)
    self.assertEqual(shuffler1.rounds_per_epoch, 2)
    for round_num in range(7):
      self.assertEqual(shuffler1(round_num), shuffler2(round_num))

  @parameterized.named_parameters(('n=20', 20), ('n=21', 21), ('n=23', 23))
  def test_fixed_offset(self, n):
    # Should always produce 5 rounds per epoch.
    clients_per_round = 4
    client_ids = list(range(n))
    rounds_per_epoch = 5
    shuffler = training_loop.ClientIDShuffler(
        clients_per_round, client_ids, seed=1234567, reshuffle_each_epoch=False
    )
    self.assertEqual(shuffler.rounds_per_epoch, rounds_per_epoch)

    # Test that we see the same clients for the same round_index
    for i in range(rounds_per_epoch):
      epoch_0_clients = shuffler(i)[0]
      self.assertLen(epoch_0_clients, clients_per_round)
      self.assertEqual(epoch_0_clients, shuffler(i)[0])  # Check idempotent
      self.assertEqual(
          epoch_0_clients, shuffler(round_num=rounds_per_epoch + i)[0]
      )
      self.assertEqual(
          epoch_0_clients, shuffler(round_num=9 * rounds_per_epoch + i)[0]
      )

    # Test that we see rounds_per_epoch*clients_per_round unique
    # client_ids during an epoch
    all_clients_seen = collections.Counter()
    for i in range(rounds_per_epoch):
      clients_for_round = shuffler(i)[0]
      self.assertLen(clients_for_round, clients_per_round)
      all_clients_seen.update(clients_for_round)
    # Check we have seen distinct clients:
    self.assertEqual(
        sum(all_clients_seen.values()), rounds_per_epoch * clients_per_round
    )
    self.assertLen(
        all_clients_seen.keys(), rounds_per_epoch * clients_per_round
    )


if __name__ == '__main__':
  tf.test.main()
