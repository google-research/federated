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

import collections
import os.path

import tensorflow as tf
import tensorflow_federated as tff

from utils import training_utils


def create_scalar_metrics():
  return collections.OrderedDict([
      ('a', {
          'b': 1.0,
          'c': 2.0,
      }),
  ])


def create_task():
  train_client_spec = tff.simulation.baselines.ClientSpec(
      num_epochs=1, batch_size=10, shuffle_buffer_size=1)
  return tff.simulation.baselines.emnist.create_autoencoder_task(
      train_client_spec, use_synthetic_data=True)


class TrainingUtilsTest(tf.test.TestCase):

  def test_checkpoint_manager_saves_to_correct_dir(self):
    root_output_dir = self.get_temp_dir()
    experiment_name = 'test'
    checkpoint_manager, _ = training_utils.configure_managers(
        root_output_dir, experiment_name, rounds_per_checkpoint=1)
    self.assertIsInstance(checkpoint_manager,
                          tff.simulation.FileCheckpointManager)

    checkpoint_path = os.path.join(root_output_dir, 'checkpoints',
                                   experiment_name)
    test_state = create_scalar_metrics()
    checkpoint_manager.save_checkpoint(test_state, 1)
    self.assertCountEqual(tf.io.gfile.listdir(checkpoint_path), ['ckpt_1'])

  def test_program_state_manager_saves_to_correct_dir(self):
    root_output_dir = self.get_temp_dir()
    experiment_name = 'test'
    program_state_manager, _ = training_utils.configure_output_managers(
        root_output_dir, experiment_name)
    self.assertIsInstance(program_state_manager,
                          tff.program.FileProgramStateManager)

    program_state_dir = os.path.join(root_output_dir, 'program_states',
                                     experiment_name)
    test_state = create_scalar_metrics()
    program_state_manager.set_structure(test_state)
    program_state_manager.save(test_state, 1)
    self.assertCountEqual(
        tf.io.gfile.listdir(program_state_dir), ['program_state_1'])

  def test_csv_manager_saves_to_correct_dir(self):
    root_output_dir = self.get_temp_dir()
    experiment_name = 'test'
    _, metrics_managers = training_utils.configure_output_managers(
        root_output_dir, experiment_name)
    csv_manager, _ = metrics_managers
    self.assertIsInstance(csv_manager, tff.simulation.CSVMetricsManager)

    expected_metrics_file = os.path.join(root_output_dir, 'results',
                                         experiment_name,
                                         'experiment.metrics.csv')
    self.assertEqual(csv_manager._metrics_file, expected_metrics_file)

  def test_tensorboard_manager_saves_to_correct_dir(self):
    root_output_dir = self.get_temp_dir()
    experiment_name = 'test'
    _, metrics_managers = training_utils.configure_output_managers(
        root_output_dir, experiment_name)
    _, tensorboard_manager = metrics_managers
    self.assertIsInstance(tensorboard_manager,
                          tff.simulation.TensorBoardManager)

    summary_dir = os.path.join(root_output_dir, 'logdir', experiment_name)
    tensorboard_manager.save_metrics(create_scalar_metrics(), 0)
    self.assertTrue(tf.io.gfile.exists(summary_dir))
    self.assertLen(tf.io.gfile.listdir(summary_dir), 1)

  def test_write_hparams_to_csv_writes_to_correct_file(self):
    root_output_dir = self.get_temp_dir()
    experiment_name = 'test'
    hparams = {'a': 1, 'b': 'foo'}
    hparam_file = os.path.join(root_output_dir, 'results', experiment_name,
                               'hparams.csv')
    training_utils.write_hparams_to_csv(hparams, root_output_dir,
                                        experiment_name)
    self.assertTrue(tf.io.gfile.exists(hparam_file))

  def test_validation_fn_is_compatible_with_model_weights(self):
    task = create_task()
    validation_fn = training_utils.create_validation_fn(
        task, validation_frequency=5)
    model = task.model_fn()
    model_weights = tff.learning.ModelWeights.from_model(model)
    validation_metrics = validation_fn(model_weights, round_num=1)
    self.assertIsInstance(validation_metrics, dict)

  def test_validation_fn_uses_validation_frequency(self):
    task = create_task()
    validation_fn = training_utils.create_validation_fn(
        task, validation_frequency=5)
    model = task.model_fn()
    model_weights = tff.learning.ModelWeights.from_model(model)
    validation_metrics = validation_fn(model_weights, round_num=1)
    self.assertDictEqual(validation_metrics, {})

    validation_metrics = validation_fn(model_weights, round_num=5)
    self.assertNotEqual(validation_metrics, {})

  def test_validation_fn_uses_num_validation_examples(self):
    task = create_task()
    validation_fn = training_utils.create_validation_fn(
        task, validation_frequency=1, num_validation_examples=3)
    model = task.model_fn()
    model_weights = tff.learning.ModelWeights.from_model(model)
    validation_metrics = validation_fn(model_weights, round_num=1)
    self.assertEqual(validation_metrics['stat']['num_examples'], 3)

  def test_create_test_fn_is_compatible_with_model_weights(self):
    task = create_task()
    test_fn = training_utils.create_test_fn(task)
    model = task.model_fn()
    model_weights = tff.learning.ModelWeights.from_model(model)
    test_metrics = test_fn(model_weights)
    self.assertIsInstance(test_metrics, dict)
    self.assertNotEqual(test_metrics, {})

  def test_create_client_selection_fn_returns_client_datasets(self):
    task = create_task()
    client_selection_fn = training_utils.create_client_selection_fn(
        task, clients_per_round=1)
    client_datasets = client_selection_fn(round_num=17)
    self.assertLen(client_datasets, 1)
    client_dataset = client_datasets[0]
    self.assertIsInstance(client_dataset, tf.data.Dataset)

    expected_dataset = task.datasets.sample_train_clients(num_clients=1)[0]
    self.assertAllClose(
        list(client_dataset.as_numpy_iterator()),
        list(expected_dataset.as_numpy_iterator()))


if __name__ == '__main__':
  tf.test.main()
