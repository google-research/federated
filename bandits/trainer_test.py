# Copyright 2022, Google LLC.
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
"""Tests for trainer."""
import collections
import os
from unittest import mock
from absl.testing import flagsaver
from absl.testing import parameterized

import tensorflow as tf
import tensorflow_federated as tff

from bandits import bandits_utils
from bandits import keras_optimizer_utils
from bandits import trainer
from bandits.tasks import task_utils


keras_optimizer_utils.define_optimizer_flags('client')
keras_optimizer_utils.define_optimizer_flags('server')


def create_scalar_metrics():
  metrics = collections.OrderedDict(
      [
          (
              'a',
              collections.OrderedDict([
                  ('b', 1.0),
                  ('c', 2.0),
              ]),
          ),
      ]
  )
  metrics_type = tff.StructType(
      [
          (
              'a',
              tff.StructType([
                  ('b', tf.float32),
                  ('c', tf.float32),
              ]),
          ),
      ]
  )
  return metrics, metrics_type


class ManagerTest(tf.test.TestCase):

  def test_program_state_manager_saves_to_correct_dir(self):
    root_output_dir = self.get_temp_dir()
    experiment_name = 'test'
    program_state_manager, _ = trainer.configure_managers(
        root_output_dir, experiment_name
    )
    self.assertIsInstance(
        program_state_manager, tff.program.FileProgramStateManager
    )

  def test_csv_manager_saves_to_correct_dir(self):
    root_output_dir = self.get_temp_dir()
    experiment_name = 'test'
    _, metrics_managers = trainer.configure_managers(
        root_output_dir, experiment_name
    )
    _, csv_manager, _ = metrics_managers
    self.assertIsInstance(csv_manager, tff.program.CSVFileReleaseManager)

    expected_metrics_file = os.path.join(
        root_output_dir, 'results', experiment_name, 'experiment.metrics.csv'
    )
    self.assertEqual(csv_manager._file_path, expected_metrics_file)

  def test_tensorboard_manager_saves_to_correct_dir(self):
    root_output_dir = self.get_temp_dir()
    experiment_name = 'test'
    _, metrics_managers = trainer.configure_managers(
        root_output_dir, experiment_name
    )
    _, _, tensorboard_manager = metrics_managers
    self.assertIsInstance(
        tensorboard_manager, tff.program.TensorBoardReleaseManager
    )

    summary_dir = os.path.join(root_output_dir, 'logdir', experiment_name)
    metrics, metrics_type = create_scalar_metrics()
    tensorboard_manager.release(metrics, metrics_type, 0)
    self.assertTrue(tf.io.gfile.exists(summary_dir))
    self.assertLen(tf.io.gfile.listdir(summary_dir), 1)


def _variable_list_from(
    model_weights: tff.learning.models.ModelWeights,
) -> list[tf.Tensor]:
  return model_weights.trainable + model_weights.non_trainable


class TrainerTest(tf.test.TestCase, parameterized.TestCase):

  def test_get_task_types(self):
    self.assertListEqual(list(trainer.TaskType), trainer.get_task_types())

  def test_get_bandits_types(self):
    self.assertListEqual(list(trainer.BanditsType), trainer.get_bandits_types())

  @parameterized.named_parameters(
      (
          'emnist10_bandits_eps',
          trainer.TaskType.EMNIST_DIGITS,
          trainer.BanditsType.EPSILON_GREEDY,
      ),
      (
          'emnist62_bandits_eps',
          trainer.TaskType.EMNIST_CHARS,
          trainer.BanditsType.EPSILON_GREEDY,
      ),
      (
          'emnist10_supervised',
          trainer.TaskType.EMNIST_DIGITS,
          trainer.BanditsType.SUPERVISED_MSE,
      ),
      (
          'emnist62_supervised',
          trainer.TaskType.EMNIST_CHARS,
          trainer.BanditsType.SUPERVISED_MSE,
      ),
      (
          'emnist10linear_bandits_eps',
          trainer.TaskType.EMNIST10_LINEAR,
          trainer.BanditsType.EPSILON_GREEDY,
      ),
      (
          'emnist62linear_bandits_eps',
          trainer.TaskType.EMNIST62_LINEAR,
          trainer.BanditsType.EPSILON_GREEDY,
      ),
      (
          'emnist10linear_supervised',
          trainer.TaskType.EMNIST10_LINEAR,
          trainer.BanditsType.SUPERVISED_MSE,
      ),
      (
          'emnist62linear_supervised',
          trainer.TaskType.EMNIST62_LINEAR,
          trainer.BanditsType.SUPERVISED_MSE,
      ),
      (
          'emnist10linear_sce',
          trainer.TaskType.EMNIST10_LINEAR,
          trainer.BanditsType.SUPERVISED_CE,
      ),
      (
          'emnist62linear_sce',
          trainer.TaskType.EMNIST62_LINEAR,
          trainer.BanditsType.SUPERVISED_CE,
      ),
      (
          'emnist10_bandits_ce_eps',
          trainer.TaskType.EMNIST_DIGITS,
          trainer.BanditsType.EPSILON_GREEDY_CE,
      ),
      (
          'emnist62_bandits_ce_eps',
          trainer.TaskType.EMNIST_CHARS,
          trainer.BanditsType.EPSILON_GREEDY_CE,
      ),
  )
  @flagsaver.flagsaver(
      server_optimizer='sgd',
      server_learning_rate=0.1,
      server_sgd_momentum=0.9,
      client_optimizer='sgd',
      client_learning_rate=0.1,
  )
  def test_train_and_eval(self, task, bandits):
    server_optimizer_fn = keras_optimizer_utils.create_optimizer_fn_from_flags(
        'server'
    )
    client_optimizer_fn = keras_optimizer_utils.create_optimizer_fn_from_flags(
        'client'
    )
    trainer.train_and_eval(
        task,
        bandits=bandits,
        total_rounds=5,
        clients_per_round=1,
        rounds_per_eval=1,
        server_optimizer=server_optimizer_fn,
        client_optimizer=client_optimizer_fn,
        use_synthetic_data=True,
        train_client_batch_size=4,
        test_client_batch_size=8,
        bandits_deployment_frequency=2,
    )

  @parameterized.named_parameters(
      (
          'emnist10_bandits_eps',
          trainer.TaskType.EMNIST_DIGITS,
          trainer.BanditsType.EPSILON_GREEDY,
      ),
      (
          'emnist62_bandits_eps',
          trainer.TaskType.EMNIST_CHARS,
          trainer.BanditsType.EPSILON_GREEDY,
      ),
      (
          'emnist10_supervised',
          trainer.TaskType.EMNIST_DIGITS,
          trainer.BanditsType.SUPERVISED_MSE,
      ),
      (
          'emnist10_bandits_ce_eps',
          trainer.TaskType.EMNIST_DIGITS,
          trainer.BanditsType.EPSILON_GREEDY_CE,
      ),
      (
          'stackoverflow_bandits_eps',
          trainer.TaskType.STACKOVERFLOW_TAG,
          trainer.BanditsType.EPSILON_GREEDY,
      ),
      (
          'stackoverflow_bandits_ce_eps',
          trainer.TaskType.STACKOVERFLOW_TAG,
          trainer.BanditsType.EPSILON_GREEDY_CE,
      ),
      (
          'stackoverflow_supervised',
          trainer.TaskType.STACKOVERFLOW_TAG,
          trainer.BanditsType.SUPERVISED_MSE,
      ),
      (
          'stackoverflow_supervised_ce',
          trainer.TaskType.STACKOVERFLOW_TAG,
          trainer.BanditsType.SUPERVISED_CE,
      ),
  )
  @flagsaver.flagsaver(
      server_optimizer='adam',
      server_learning_rate=0.1,
      server_adam_beta_1=0.9,
      server_adam_beta_2=0.99,
      server_adam_epsilon=1e-5,
      client_optimizer='sgd',
      client_learning_rate=0.1,
  )
  def test_train_and_eval_adam(self, task, bandits):
    server_optimizer_fn = keras_optimizer_utils.create_optimizer_fn_from_flags(
        'server'
    )
    client_optimizer_fn = keras_optimizer_utils.create_optimizer_fn_from_flags(
        'client'
    )
    trainer.train_and_eval(
        task,
        bandits=bandits,
        total_rounds=5,
        clients_per_round=1,
        rounds_per_eval=1,
        server_optimizer=server_optimizer_fn,
        client_optimizer=client_optimizer_fn,
        use_synthetic_data=True,
        train_client_batch_size=4,
        test_client_batch_size=8,
        bandits_deployment_frequency=2,
        stackoverflow_vocab_size=30,
        stackoverflow_tag_size=5,
    )

  @parameterized.named_parameters(
      (
          'emnist10',
          trainer.TaskType.EMNIST_DIGITS,
          trainer.BanditsType.SUPERVISED_MSE,
          trainer.BanditsType.EPSILON_GREEDY,
      ),
      (
          'stackoverflow',
          trainer.TaskType.STACKOVERFLOW_TAG,
          trainer.BanditsType.SUPERVISED_MSE,
          trainer.BanditsType.EPSILON_GREEDY,
      ),
  )
  @flagsaver.flagsaver(
      server_optimizer='adam',
      server_learning_rate=0.1,
      server_adam_beta_1=0.9,
      server_adam_beta_2=0.99,
      server_adam_epsilon=1e-5,
      client_optimizer='sgd',
      client_learning_rate=0.1,
  )
  def test_init_from_pretrain(self, task, bandits1, bandits2):
    # The first stage will use `bandits1` to train an initial model.
    filepath = self.create_tempdir()
    server_optimizer_fn = keras_optimizer_utils.create_optimizer_fn_from_flags(
        'server'
    )
    client_optimizer_fn = keras_optimizer_utils.create_optimizer_fn_from_flags(
        'client'
    )
    initial_state = trainer.train_and_eval(
        task,
        bandits=bandits1,
        total_rounds=3,
        clients_per_round=1,
        rounds_per_eval=10,
        server_optimizer=server_optimizer_fn,
        client_optimizer=client_optimizer_fn,
        use_synthetic_data=True,
        train_client_batch_size=4,
        test_client_batch_size=8,
        bandits_deployment_frequency=1,  # No delay in model deployment.
        stackoverflow_vocab_size=30,
        stackoverflow_tag_size=5,
        export_dir=filepath,
    )
    # Check the loaded initial model.
    state = trainer.train_and_eval(
        task,
        bandits=bandits2,
        total_rounds=0,  # No training rounds.
        clients_per_round=1,
        rounds_per_eval=1,
        server_optimizer=server_optimizer_fn,
        client_optimizer=client_optimizer_fn,
        use_synthetic_data=True,
        train_client_batch_size=4,
        test_client_batch_size=8,
        bandits_deployment_frequency=2,
        stackoverflow_vocab_size=30,
        stackoverflow_tag_size=5,
        initial_model_path=filepath,
    )
    self.assertAllClose(
        _variable_list_from(state.delayed_inference_model),
        _variable_list_from(initial_state.train_state.global_model_weights),
    )
    self.assertAllClose(
        _variable_list_from(state.train_state.global_model_weights),
        _variable_list_from(initial_state.train_state.global_model_weights),
    )
    # The second stage will start from the model trained by the first stage and
    # use `bandits2` for training.
    initial_state = trainer.train_and_eval(
        task,
        bandits=bandits2,
        total_rounds=3,
        clients_per_round=1,
        rounds_per_eval=2,
        server_optimizer=server_optimizer_fn,
        client_optimizer=client_optimizer_fn,
        use_synthetic_data=True,
        train_client_batch_size=4,
        test_client_batch_size=8,
        bandits_deployment_frequency=2,
        stackoverflow_vocab_size=30,
        stackoverflow_tag_size=5,
        initial_model_path=filepath,
    )

  @parameterized.named_parameters(
      (
          'emnist10_bandits_eps_init',
          trainer.TaskType.EMNIST_DIGITS,
          trainer.BanditsType.EPSILON_GREEDY,
          trainer.DistShiftType.INIT,
      ),
  )
  @flagsaver.flagsaver(
      server_optimizer='adam',
      server_learning_rate=0.1,
      server_adam_beta_1=0.9,
      server_adam_beta_2=0.99,
      server_adam_epsilon=1e-5,
      client_optimizer='sgd',
      client_learning_rate=0.1,
  )
  def test_train_and_eval_dist_shift_raise(self, task, bandits, dist_shift):
    server_optimizer_fn = keras_optimizer_utils.create_optimizer_fn_from_flags(
        'server'
    )
    client_optimizer_fn = keras_optimizer_utils.create_optimizer_fn_from_flags(
        'client'
    )
    with self.assertRaises(ValueError):
      trainer.train_and_eval(
          task,
          bandits=bandits,
          distribution_shift=dist_shift,
          total_rounds=5,
          clients_per_round=1,
          rounds_per_eval=1,
          server_optimizer=server_optimizer_fn,
          client_optimizer=client_optimizer_fn,
          use_synthetic_data=True,
          train_client_batch_size=4,
          test_client_batch_size=8,
          bandits_deployment_frequency=2,
          stackoverflow_vocab_size=30,
          stackoverflow_tag_size=5,
      )

  @parameterized.named_parameters(
      (
          'emnist62_bandits_eps_init',
          trainer.TaskType.EMNIST_CHARS,
          trainer.BanditsType.EPSILON_GREEDY,
          trainer.DistShiftType.INIT,
      ),
      (
          'emnist62_bandits_eps_shift',
          trainer.TaskType.EMNIST_CHARS,
          trainer.BanditsType.EPSILON_GREEDY_UNWEIGHT,
          trainer.DistShiftType.BANDITS,
      ),
      (
          'stackoverflow_supervised_init',
          trainer.TaskType.STACKOVERFLOW_TAG,
          trainer.BanditsType.SUPERVISED_MSE,
          trainer.DistShiftType.INIT,
      ),
      (
          'stackoverflow_bandits_eps_shift',
          trainer.TaskType.STACKOVERFLOW_TAG,
          trainer.BanditsType.EPSILON_GREEDY_UNWEIGHT,
          trainer.DistShiftType.BANDITS,
      ),
      (
          'emnist62_bandits_cb_shift',
          trainer.TaskType.EMNIST_CHARS,
          trainer.BanditsType.FALCON,
          trainer.DistShiftType.BANDITS,
      ),
      (
          'stackoverflow_bandits_cb_shift',
          trainer.TaskType.STACKOVERFLOW_TAG,
          trainer.BanditsType.FALCON,
          trainer.DistShiftType.BANDITS,
      ),
      (
          'emnist62_bandits_softmax_shift',
          trainer.TaskType.EMNIST_CHARS,
          trainer.BanditsType.SOFTMAX,
          trainer.DistShiftType.BANDITS,
      ),
      (
          'stackoverflow_bandits_softmax_shift',
          trainer.TaskType.STACKOVERFLOW_TAG,
          trainer.BanditsType.SOFTMAX,
          trainer.DistShiftType.BANDITS,
      ),
  )
  @flagsaver.flagsaver(
      server_optimizer='rmsprop',
      server_learning_rate=0.1,
      client_optimizer='sgd',
      client_learning_rate=0.1,
  )
  def test_train_and_eval_dist_shift(self, task, bandits, dist_shift):
    server_optimizer_fn = keras_optimizer_utils.create_optimizer_fn_from_flags(
        'server'
    )
    client_optimizer_fn = keras_optimizer_utils.create_optimizer_fn_from_flags(
        'client'
    )
    trainer.train_and_eval(
        task,
        bandits=bandits,
        distribution_shift=dist_shift,
        total_rounds=5,
        clients_per_round=1,
        rounds_per_eval=1,
        server_optimizer=server_optimizer_fn,
        client_optimizer=client_optimizer_fn,
        use_synthetic_data=True,
        train_client_batch_size=4,
        test_client_batch_size=8,
        bandits_deployment_frequency=2,
        stackoverflow_vocab_size=30,
        stackoverflow_tag_size=5,
    )

  @parameterized.named_parameters(
      ('eps_greedy', trainer.BanditsType.EPSILON_GREEDY),
      ('falcon', trainer.BanditsType.FALCON),
      ('softmax', trainer.BanditsType.SOFTMAX),
  )
  @mock.patch.object(
      bandits_utils,
      'get_emnist_dist_shift_reward_fn',
      wraps=bandits_utils.get_emnist_dist_shift_reward_fn,
  )
  @flagsaver.flagsaver(
      server_optimizer='adam',
      server_learning_rate=0.1,
      server_adam_beta_1=0.9,
      server_adam_beta_2=0.99,
      server_adam_epsilon=1e-5,
      client_optimizer='sgd',
      client_learning_rate=0.1,
  )
  def test_train_and_eval_emnist62_dist_shift(self, bandits_type, mock_method):
    server_optimizer_fn = keras_optimizer_utils.create_optimizer_fn_from_flags(
        'server'
    )
    client_optimizer_fn = keras_optimizer_utils.create_optimizer_fn_from_flags(
        'client'
    )
    trainer.train_and_eval(
        task=trainer.TaskType.EMNIST_CHARS,
        bandits=bandits_type,
        distribution_shift=trainer.DistShiftType.BANDITS,
        total_rounds=2,
        clients_per_round=1,
        rounds_per_eval=1,
        server_optimizer=server_optimizer_fn,
        client_optimizer=client_optimizer_fn,
        use_synthetic_data=True,
        train_client_batch_size=4,
        test_client_batch_size=8,
        bandits_deployment_frequency=2,
        stackoverflow_vocab_size=30,
        stackoverflow_tag_size=5,
    )
    mock_method.assert_called()

  @parameterized.named_parameters(
      ('eps_greedy', trainer.BanditsType.EPSILON_GREEDY),
      ('falcon', trainer.BanditsType.FALCON),
      ('softmax', trainer.BanditsType.SOFTMAX),
  )
  @mock.patch.object(
      bandits_utils,
      'get_stackoverflow_dist_shift_reward_fn',
      wraps=bandits_utils.get_stackoverflow_dist_shift_reward_fn,
  )
  @flagsaver.flagsaver(
      server_optimizer='adam',
      server_learning_rate=0.1,
      server_adam_beta_1=0.9,
      server_adam_beta_2=0.99,
      server_adam_epsilon=1e-5,
      client_optimizer='sgd',
      client_learning_rate=0.1,
  )
  def test_train_and_eval_stackoverflow_dist_shift(
      self, bandits_type, mock_method
  ):
    server_optimizer_fn = keras_optimizer_utils.create_optimizer_fn_from_flags(
        'server'
    )
    client_optimizer_fn = keras_optimizer_utils.create_optimizer_fn_from_flags(
        'client'
    )
    trainer.train_and_eval(
        task=trainer.TaskType.STACKOVERFLOW_TAG,
        bandits=bandits_type,
        distribution_shift=trainer.DistShiftType.BANDITS,
        total_rounds=2,
        clients_per_round=1,
        rounds_per_eval=1,
        server_optimizer=server_optimizer_fn,
        client_optimizer=client_optimizer_fn,
        use_synthetic_data=True,
        train_client_batch_size=4,
        test_client_batch_size=8,
        bandits_deployment_frequency=2,
        stackoverflow_vocab_size=30,
        stackoverflow_tag_size=5,
    )
    mock_method.assert_called()

  @parameterized.named_parameters(
      ('emnist62', trainer.TaskType.EMNIST_CHARS),
      ('stackoverflow', trainer.TaskType.STACKOVERFLOW_TAG),
  )
  @mock.patch.object(
      task_utils,
      'clientdata_for_select_clients',
      wraps=task_utils.clientdata_for_select_clients,
  )
  @flagsaver.flagsaver(
      server_optimizer='adam',
      server_learning_rate=0.1,
      server_adam_beta_1=0.9,
      server_adam_beta_2=0.99,
      server_adam_epsilon=1e-5,
      client_optimizer='sgd',
      client_learning_rate=0.1,
  )
  def test_train_and_eval_client_shift(self, task, mock_method):
    server_optimizer_fn = keras_optimizer_utils.create_optimizer_fn_from_flags(
        'server'
    )
    client_optimizer_fn = keras_optimizer_utils.create_optimizer_fn_from_flags(
        'client'
    )
    trainer.train_and_eval(
        task=task,
        bandits=trainer.BanditsType.EPSILON_GREEDY,
        distribution_shift=trainer.DistShiftType.BANDITS,
        population_client_selection='0-1',
        total_rounds=2,
        clients_per_round=1,
        rounds_per_eval=1,
        server_optimizer=server_optimizer_fn,
        client_optimizer=client_optimizer_fn,
        use_synthetic_data=True,
        train_client_batch_size=4,
        test_client_batch_size=8,
        bandits_deployment_frequency=2,
        stackoverflow_vocab_size=30,
        stackoverflow_tag_size=5,
    )
    mock_method.assert_called()


class AggregatorTest(tf.test.TestCase, parameterized.TestCase):

  @mock.patch.object(
      tff.aggregators.DifferentiallyPrivateFactory,
      'gaussian_adaptive',
      wraps=tff.aggregators.DifferentiallyPrivateFactory.gaussian_adaptive,
  )
  @flagsaver.flagsaver(
      server_optimizer='adam',
      server_learning_rate=0.1,
      server_adam_beta_1=0.9,
      server_adam_beta_2=0.99,
      server_adam_epsilon=1e-5,
      client_optimizer='sgd',
      client_learning_rate=0.1,
  )
  def test_adaptive_dpsgd_call(self, mock_method):
    server_optimizer_fn = keras_optimizer_utils.create_optimizer_fn_from_flags(
        'server'
    )
    client_optimizer_fn = keras_optimizer_utils.create_optimizer_fn_from_flags(
        'client'
    )
    trainer.train_and_eval(
        task=trainer.TaskType.EMNIST_CHARS,
        bandits=trainer.BanditsType.EPSILON_GREEDY,
        distribution_shift=trainer.DistShiftType.BANDITS,
        population_client_selection='0-1',
        total_rounds=1,
        clients_per_round=1,
        rounds_per_eval=1,
        server_optimizer=server_optimizer_fn,
        client_optimizer=client_optimizer_fn,
        use_synthetic_data=True,
        train_client_batch_size=4,
        test_client_batch_size=8,
        bandits_deployment_frequency=2,
        aggregator_type=trainer.AggregatorType.DPSGD,
        target_unclipped_quantile=0.5,
        clip_norm=0.1,
        noise_multiplier=1e-4,
    )
    mock_method.assert_called()

  @mock.patch.object(
      tff.aggregators.DifferentiallyPrivateFactory,
      'gaussian_fixed',
      wraps=tff.aggregators.DifferentiallyPrivateFactory.gaussian_fixed,
  )
  @flagsaver.flagsaver(
      server_optimizer='adam',
      server_learning_rate=0.1,
      server_adam_beta_1=0.9,
      server_adam_beta_2=0.99,
      server_adam_epsilon=1e-5,
      client_optimizer='sgd',
      client_learning_rate=0.1,
  )
  def test_dpsgd_call(self, mock_method):
    server_optimizer_fn = keras_optimizer_utils.create_optimizer_fn_from_flags(
        'server'
    )
    client_optimizer_fn = keras_optimizer_utils.create_optimizer_fn_from_flags(
        'client'
    )
    trainer.train_and_eval(
        task=trainer.TaskType.EMNIST_CHARS,
        bandits=trainer.BanditsType.EPSILON_GREEDY,
        distribution_shift=trainer.DistShiftType.BANDITS,
        population_client_selection='0-1',
        total_rounds=1,
        clients_per_round=1,
        rounds_per_eval=1,
        server_optimizer=server_optimizer_fn,
        client_optimizer=client_optimizer_fn,
        use_synthetic_data=True,
        train_client_batch_size=4,
        test_client_batch_size=8,
        bandits_deployment_frequency=2,
        aggregator_type=trainer.AggregatorType.DPSGD,
        target_unclipped_quantile=None,
        clip_norm=0.1,
        noise_multiplier=1e-4,
    )
    mock_method.assert_called()

  @mock.patch.object(
      tff.aggregators.DifferentiallyPrivateFactory,
      'tree_aggregation',
      wraps=tff.aggregators.DifferentiallyPrivateFactory.tree_aggregation,
  )
  @flagsaver.flagsaver(
      server_optimizer='adam',
      server_learning_rate=0.1,
      server_adam_beta_1=0.9,
      server_adam_beta_2=0.99,
      server_adam_epsilon=1e-5,
      client_optimizer='sgd',
      client_learning_rate=0.1,
  )
  def test_dpftrl_call(self, mock_method):
    server_optimizer_fn = keras_optimizer_utils.create_optimizer_fn_from_flags(
        'server'
    )
    client_optimizer_fn = keras_optimizer_utils.create_optimizer_fn_from_flags(
        'client'
    )
    trainer.train_and_eval(
        task=trainer.TaskType.EMNIST_CHARS,
        bandits=trainer.BanditsType.EPSILON_GREEDY,
        distribution_shift=trainer.DistShiftType.BANDITS,
        population_client_selection='0-1',
        total_rounds=1,
        clients_per_round=1,
        rounds_per_eval=1,
        server_optimizer=server_optimizer_fn,
        client_optimizer=client_optimizer_fn,
        use_synthetic_data=True,
        train_client_batch_size=4,
        test_client_batch_size=8,
        bandits_deployment_frequency=2,
        aggregator_type=trainer.AggregatorType.DPFTRL,
        target_unclipped_quantile=None,
        clip_norm=0.1,
        noise_multiplier=1e-4,
    )
    mock_method.assert_called()

  @mock.patch.object(
      tff.aggregators.robust,
      'clipping_factory',
      wraps=tff.aggregators.robust.clipping_factory,
  )
  @flagsaver.flagsaver(
      server_optimizer='adam',
      server_learning_rate=0.1,
      server_adam_beta_1=0.9,
      server_adam_beta_2=0.99,
      server_adam_epsilon=1e-5,
      client_optimizer='sgd',
      client_learning_rate=0.1,
  )
  def test_clipping_call(self, mock_method):
    server_optimizer_fn = keras_optimizer_utils.create_optimizer_fn_from_flags(
        'server'
    )
    client_optimizer_fn = keras_optimizer_utils.create_optimizer_fn_from_flags(
        'client'
    )
    trainer.train_and_eval(
        task=trainer.TaskType.EMNIST_CHARS,
        bandits=trainer.BanditsType.EPSILON_GREEDY,
        distribution_shift=trainer.DistShiftType.BANDITS,
        population_client_selection='0-1',
        total_rounds=1,
        clients_per_round=1,
        rounds_per_eval=1,
        server_optimizer=server_optimizer_fn,
        client_optimizer=client_optimizer_fn,
        use_synthetic_data=True,
        train_client_batch_size=4,
        test_client_batch_size=8,
        bandits_deployment_frequency=2,
        target_unclipped_quantile=None,
        clip_norm=0.1,
    )
    mock_method.assert_called()

  @parameterized.named_parameters(
      ('dpsgd', trainer.AggregatorType.DPSGD),
      ('dpftrl', trainer.AggregatorType.DPFTRL),
  )
  @flagsaver.flagsaver(
      server_optimizer='adam',
      server_learning_rate=0.1,
      server_adam_beta_1=0.9,
      server_adam_beta_2=0.99,
      server_adam_epsilon=1e-5,
      client_optimizer='sgd',
      client_learning_rate=0.1,
  )
  def test_clipping_raise(self, aggregator_type):
    server_optimizer_fn = keras_optimizer_utils.create_optimizer_fn_from_flags(
        'server'
    )
    client_optimizer_fn = keras_optimizer_utils.create_optimizer_fn_from_flags(
        'client'
    )
    with self.assertRaisesRegex(ValueError, 'Clipping only without noise'):
      trainer.train_and_eval(
          task=trainer.TaskType.EMNIST_CHARS,
          bandits=trainer.BanditsType.EPSILON_GREEDY,
          distribution_shift=trainer.DistShiftType.BANDITS,
          population_client_selection='0-1',
          total_rounds=1,
          clients_per_round=1,
          rounds_per_eval=1,
          server_optimizer=server_optimizer_fn,
          client_optimizer=client_optimizer_fn,
          use_synthetic_data=True,
          train_client_batch_size=4,
          test_client_batch_size=8,
          bandits_deployment_frequency=2,
          aggregator_type=aggregator_type,
          target_unclipped_quantile=None,
          clip_norm=0.1,
      )

  @parameterized.named_parameters(
      ('adaptive_dpsgd', trainer.AggregatorType.DPSGD, 0.5),
      ('dpsgd', trainer.AggregatorType.DPSGD, None),
      ('dpftrl', trainer.AggregatorType.DPFTRL, None),
  )
  @flagsaver.flagsaver(
      server_optimizer='adam',
      server_learning_rate=0.1,
      server_adam_beta_1=0.9,
      server_adam_beta_2=0.99,
      server_adam_epsilon=1e-5,
      client_optimizer='sgd',
      client_learning_rate=0.1,
  )
  def test_train_and_eval(self, aggregator_type, target_unclipped_quantile):
    server_optimizer_fn = keras_optimizer_utils.create_optimizer_fn_from_flags(
        'server'
    )
    client_optimizer_fn = keras_optimizer_utils.create_optimizer_fn_from_flags(
        'client'
    )
    trainer.train_and_eval(
        task=trainer.TaskType.EMNIST_CHARS,
        bandits=trainer.BanditsType.EPSILON_GREEDY,
        distribution_shift=trainer.DistShiftType.BANDITS,
        population_client_selection='0-1',
        total_rounds=3,
        clients_per_round=1,
        rounds_per_eval=1,
        server_optimizer=server_optimizer_fn,
        client_optimizer=client_optimizer_fn,
        use_synthetic_data=True,
        train_client_batch_size=4,
        test_client_batch_size=8,
        bandits_deployment_frequency=2,
        aggregator_type=aggregator_type,
        target_unclipped_quantile=target_unclipped_quantile,
        clip_norm=0.1,
        noise_multiplier=1e-4,
    )


if __name__ == '__main__':
  tf.test.main()
