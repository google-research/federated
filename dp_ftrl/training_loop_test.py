# Copyright 2021, Google LLC.
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
"""Tests for shared training loops."""

import collections
import os

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from dp_ftrl import training_loop

_Batch = collections.namedtuple('Batch', ['x', 'y'])


def _build_federated_averaging_process():
  return tff.learning.build_federated_averaging_process(
      _uncompiled_model_fn,
      client_optimizer_fn=tf.keras.optimizers.SGD,
      server_optimizer_fn=tf.keras.optimizers.SGD)


def _uncompiled_model_fn():
  keras_model = tff.simulation.models.mnist.create_keras_model(
      compile_model=False)
  input_spec = _create_input_spec()
  return tff.learning.from_keras_model(
      keras_model=keras_model,
      input_spec=input_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy())


def _batch_fn():
  batch = _Batch(
      x=np.ones([1, 784], dtype=np.float32), y=np.ones([1, 1], dtype=np.int64))
  return batch


def _create_input_spec():
  return _Batch(
      x=tf.TensorSpec(shape=[None, 784], dtype=tf.float32),
      y=tf.TensorSpec(dtype=tf.int64, shape=[None, 1]))


class ExperimentRunnerTest(tf.test.TestCase):

  def test_raises_non_iterative_process(self):
    bad_iterative_process = _build_federated_averaging_process().next
    federated_data = [[_batch_fn()]]

    def client_datasets_fn(round_num, epoch):
      del round_num
      return federated_data, epoch

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
          experiment_name='non_iterative_process',
          root_output_dir=root_output_dir)

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
          experiment_name='non_callable_client_dataset',
          root_output_dir=root_output_dir)

  def test_raises_non_callable_evaluate_fn(self):
    iterative_process = _build_federated_averaging_process()
    federated_data = [[_batch_fn()]]

    def client_datasets_fn(round_num, epoch):
      del round_num
      return federated_data, epoch

    metrics_dict = {}
    root_output_dir = self.get_temp_dir()
    with self.assertRaises(TypeError):
      training_loop.run(
          iterative_process=iterative_process,
          client_datasets_fn=client_datasets_fn,
          validation_fn=metrics_dict,
          total_epochs=1,
          total_rounds=10,
          experiment_name='non_callable_evaluate',
          root_output_dir=root_output_dir)

  def test_raises_non_str_output_dir(self):
    iterative_process = _build_federated_averaging_process()
    federated_data = [[_batch_fn()]]

    def client_datasets_fn(round_num, epoch):
      del round_num
      return federated_data, epoch

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
          experiment_name='non_str_output_dir',
          root_output_dir=1)

  def test_fedavg_training_decreases_loss(self):
    batch = _batch_fn()
    federated_data = [[batch]]
    iterative_process = _build_federated_averaging_process()

    def client_datasets_fn(round_num, epoch):
      del round_num
      return federated_data, epoch

    def validation_fn(model):
      keras_model = tff.simulation.models.mnist.create_keras_model(
          compile_model=True)
      model.assign_weights_to(keras_model)
      return {'loss': keras_model.evaluate(batch.x, batch.y)}

    initial_state = iterative_process.initialize()

    root_output_dir = self.get_temp_dir()
    final_state = training_loop.run(
        iterative_process=iterative_process,
        client_datasets_fn=client_datasets_fn,
        validation_fn=validation_fn,
        total_epochs=1,
        total_rounds=1,
        experiment_name='fedavg_decreases_loss',
        root_output_dir=root_output_dir)
    self.assertLess(
        validation_fn(final_state.model)['loss'],
        validation_fn(initial_state.model)['loss'])

  def test_checkpoint_manager_saves_state(self):
    experiment_name = 'checkpoint_manager_saves_state'
    iterative_process = _build_federated_averaging_process()
    federated_data = [[_batch_fn()]]

    def client_datasets_fn(round_num, epoch):
      del round_num
      return federated_data, epoch

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
        experiment_name=experiment_name,
        root_output_dir=root_output_dir)

    ckpt_manager = tff.simulation.FileCheckpointManager(
        os.path.join(root_output_dir, 'checkpoints', experiment_name))
    restored_state, restored_round = ckpt_manager.load_latest_checkpoint(
        final_state)

    self.assertEqual(restored_round, 0)

    keras_model = tff.simulation.models.mnist.create_keras_model(
        compile_model=True)
    restored_state.model.assign_weights_to(keras_model)
    restored_loss = keras_model.test_on_batch(federated_data[0][0].x,
                                              federated_data[0][0].y)
    final_state.model.assign_weights_to(keras_model)
    final_loss = keras_model.test_on_batch(federated_data[0][0].x,
                                           federated_data[0][0].y)
    self.assertEqual(final_loss, restored_loss)

  def test_fn_writes_metrics(self):
    experiment_name = 'test_metrics'
    iterative_process = _build_federated_averaging_process()
    batch = _batch_fn()
    federated_data = [[batch]]

    def client_datasets_fn(round_num, epoch):
      del round_num
      return federated_data, epoch

    def test_fn(model):
      keras_model = tff.simulation.models.mnist.create_keras_model(
          compile_model=True)
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
        experiment_name=experiment_name,
        root_output_dir=root_output_dir,
        rounds_per_eval=10,
        test_fn=test_fn)

    csv_file = os.path.join(root_output_dir, 'results', experiment_name,
                            'experiment.metrics.csv')
    metrics_manager = tff.simulation.CSVMetricsManager(csv_file)
    fieldnames, metrics = metrics_manager.get_metrics()
    self.assertLen(metrics, 2)
    self.assertIn('test/loss', fieldnames)


class ClientIDShufflerTest(tf.test.TestCase):

  def test_shuffling(self):
    clients_data = tff.simulation.datasets.stackoverflow.get_synthetic()
    client_shuffer = training_loop.ClientIDShuffler(1, clients_data)
    epoch, round_num = 0, 0
    total_epochs = 2
    epoch2clientid = [[] for _ in range(total_epochs)]
    while epoch < total_epochs:
      clients, new_epoch = client_shuffer.sample_client_ids(round_num, epoch)
      epoch2clientid[epoch].extend(clients)
      round_num += 1
      epoch = new_epoch
    self.assertCountEqual(epoch2clientid[0], epoch2clientid[1])

  def test_remainder(self):
    clients_data = tff.simulation.datasets.stackoverflow.get_synthetic()
    client_shuffer1 = training_loop.ClientIDShuffler(
        len(clients_data.client_ids) - 1, clients_data, drop_remainder=True)
    client_shuffer2 = training_loop.ClientIDShuffler(
        len(clients_data.client_ids) - 1, clients_data, drop_remainder=False)
    epoch1, epoch2, round_num = 0, 0, 0
    total_rounds = 2
    while round_num < total_rounds:
      clients1, epoch1 = client_shuffer1.sample_client_ids(round_num, epoch1)
      clients2, epoch2 = client_shuffer2.sample_client_ids(round_num, epoch2)
      round_num += 1
    self.assertEqual(len(clients1), len(clients_data.client_ids) - 1)
    self.assertEqual(len(clients2), 1)
    self.assertEqual(epoch1, 2)
    self.assertEqual(epoch2, 1)


if __name__ == '__main__':
  tf.test.main()
