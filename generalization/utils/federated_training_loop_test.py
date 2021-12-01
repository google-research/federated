# Copyright 2021, The TensorFlow Federated Authors.
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
import itertools
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import tensorflow_federated as tff

from generalization.utils import federated_training_loop
from generalization.utils import metric_utils


class LoadInitialCheckpointTest(parameterized.TestCase):

  def test_returns_input_state_and_zero_if_checkpoint_is_none(self):
    file_checkpoint_manager = mock.create_autospec(
        tff.simulation.FileCheckpointManager)
    file_checkpoint_manager.load_latest_checkpoint.return_value = (None, 10)
    input_state = 'input_state'
    state, round_num = federated_training_loop._load_initial_checkpoint(
        input_state, file_checkpoint_manager)
    file_checkpoint_manager.load_latest_checkpoint.assert_called_once_with(
        input_state)
    self.assertEqual(input_state, state)
    self.assertEqual(round_num, 0)

  @parameterized.named_parameters(
      ('checkpoint_round_1', 'state', 0),
      ('checkpoint_round_2', {}, 5),
      ('checkpoint_round_3', '0.12', 10),
      ('checkpoint_round_4', 2, 2),
  )
  def test_checkpoint_not_none(self, state, round_num):
    file_checkpoint_manager = mock.create_autospec(
        tff.simulation.FileCheckpointManager)
    file_checkpoint_manager.load_latest_checkpoint.return_value = (state,
                                                                   round_num -
                                                                   1)
    input_state = 'input_state'
    actual_state, actual_round = federated_training_loop._load_initial_checkpoint(
        input_state, file_checkpoint_manager)
    file_checkpoint_manager.load_latest_checkpoint.assert_called_once_with(
        input_state)

    self.assertEqual(actual_state, state)
    self.assertEqual(actual_round, round_num)


class ComputeEvalMetricsTest(parameterized.TestCase):

  def test_eval_function_called_once(self):
    eval_fn = mock.MagicMock()
    prefix = 'prefix'
    input_state = 'state'
    round_num = 0
    federated_training_loop._compute_eval_metrics(input_state, round_num,
                                                  eval_fn, prefix)
    eval_fn.assert_called_once_with(input_state, round_num)

  def test_runs_with_empty_dict(self):
    eval_fn = lambda x, y: {}
    prefix = 'prefix'
    time_key = prefix + metric_utils.TIME_KEY

    actual_metrics = federated_training_loop._compute_eval_metrics(
        'state', 0, eval_fn, prefix)
    self.assertIn(time_key, actual_metrics.keys())
    actual_metrics.pop(time_key)
    expected_metrics = {}
    self.assertDictEqual(actual_metrics, expected_metrics)

  def test_prefixes_keys_with_unpart_string(self):
    metrics = {'metric_1': 0, 'metric_2': 1.0, 'metric_3': 'metric_3'}
    eval_fn = lambda x, y: metrics
    prefix = 'prefix'
    time_key = prefix + metric_utils.TIME_KEY

    actual_metrics = federated_training_loop._compute_eval_metrics(
        'state', 0, eval_fn, prefix)
    self.assertIn(time_key, actual_metrics.keys())
    actual_metrics.pop(time_key)

    expected_metrics = {}
    for (key, value) in metrics.items():
      expected_metrics[prefix + key] = value
    self.assertDictEqual(actual_metrics, expected_metrics)


class BuildOnLoopStartFnTest(parameterized.TestCase):

  @mock.patch.object(federated_training_loop, '_load_initial_checkpoint')
  @mock.patch.object(federated_training_loop, '_compute_eval_metrics')
  def test_calls_with_no_input_args(self, mock_compute_eval_metrics,
                                    mock_initialize):
    on_loop_start_fn = federated_training_loop._create_on_loop_start_fn()
    on_loop_start_input = 'input'
    actual_state, actual_round = on_loop_start_fn(on_loop_start_input)
    mock_initialize.assert_not_called()
    mock_compute_eval_metrics.assert_not_called()

    expected_state = on_loop_start_input
    expected_round = 1
    self.assertEqual(actual_state, expected_state)
    self.assertEqual(actual_round, expected_round)

  @mock.patch.object(federated_training_loop, '_load_initial_checkpoint')
  @mock.patch.object(federated_training_loop, '_compute_eval_metrics')
  def test_calls_with_only_checkpoint_manager_and_zero_checkpoint_round(
      self, mock_compute_eval_metrics, mock_initialize):
    file_checkpoint_manager = mock.create_autospec(
        tff.simulation.FileCheckpointManager)
    expected_state = 'state'
    expected_round = 1
    mock_initialize.return_value = (expected_state, expected_round - 1)
    on_loop_start_fn = federated_training_loop._create_on_loop_start_fn(
        file_checkpoint_manager=file_checkpoint_manager)
    on_loop_start_input = 'input'
    actual_state, actual_round = on_loop_start_fn(on_loop_start_input)
    mock_initialize.assert_called_once_with(on_loop_start_input,
                                            file_checkpoint_manager)
    mock_compute_eval_metrics.assert_not_called()
    file_checkpoint_manager.save_checkpoint.assert_called_once_with(
        expected_state, expected_round - 1)
    self.assertEqual(actual_state, expected_state)
    self.assertEqual(actual_round, expected_round)

  @mock.patch.object(federated_training_loop, '_load_initial_checkpoint')
  @mock.patch.object(federated_training_loop, '_compute_eval_metrics')
  def test_calls_with_only_checkpoint_manager_and_non_zero_checkpoint_round(
      self, mock_compute_eval_metrics, mock_initialize):
    file_checkpoint_manager = mock.create_autospec(
        tff.simulation.FileCheckpointManager)
    expected_state = 'state'
    expected_round = 3
    mock_initialize.return_value = (expected_state, expected_round)
    on_loop_start_fn = federated_training_loop._create_on_loop_start_fn(
        file_checkpoint_manager=file_checkpoint_manager)
    on_loop_start_input = 'input'
    actual_state, actual_round = on_loop_start_fn(on_loop_start_input)
    mock_initialize.assert_called_once_with(on_loop_start_input,
                                            file_checkpoint_manager)
    mock_compute_eval_metrics.assert_not_called()
    file_checkpoint_manager.save_checkpoint.assert_not_called()
    self.assertEqual(actual_state, expected_state)
    self.assertEqual(actual_round, expected_round)

  @mock.patch.object(federated_training_loop, '_load_initial_checkpoint')
  @mock.patch.object(federated_training_loop, '_compute_eval_metrics')
  def test_calls_with_only_metrics_managers(self, mock_compute_eval_metrics,
                                            mock_initialize):
    metric_manager1 = mock.create_autospec(tff.simulation.MetricsManager)
    metric_manager2 = mock.create_autospec(tff.simulation.MetricsManager)
    metrics_managers = [metric_manager1, metric_manager2]
    on_loop_start_fn = federated_training_loop._create_on_loop_start_fn(
        metrics_managers=metrics_managers)
    on_loop_start_input = 'input'
    actual_state, actual_round = on_loop_start_fn(on_loop_start_input)

    mock_initialize.assert_not_called()
    mock_compute_eval_metrics.assert_not_called()
    expected_state = on_loop_start_input
    expected_round = 1
    for metr_mngr in metrics_managers:
      metr_mngr.clear_metrics.assert_called_once_with(expected_round - 1)
      metr_mngr.save_metrics.assert_not_called()
    self.assertEqual(actual_state, expected_state)
    self.assertEqual(actual_round, expected_round)

  @parameterized.named_parameters(
      ('train_train_eval={},train_val={},val={}'.format(*eval_fn_bools),
       *eval_fn_bools)
      for eval_fn_bools in itertools.product([False, True], repeat=3))
  @mock.patch.object(federated_training_loop, '_load_initial_checkpoint')
  @mock.patch.object(federated_training_loop, '_compute_eval_metrics')
  def test_calls_with_only_eval_fns(self, use_part_train_eval_fn,
                                    use_part_val_fn, use_unpart_fn,
                                    mock_compute_eval_metrics, mock_initialize):
    part_train_eval_fn = mock.MagicMock() if use_part_train_eval_fn else None
    part_val_fn = mock.MagicMock() if use_part_val_fn else None
    unpart_fn = mock.MagicMock() if use_unpart_fn else None

    on_loop_start_fn = federated_training_loop._create_on_loop_start_fn(
        part_train_eval_fn=part_train_eval_fn,
        part_val_fn=part_val_fn,
        unpart_fn=unpart_fn)

    on_loop_start_input = 'input'
    actual_state, actual_round = on_loop_start_fn(on_loop_start_input)

    mock_initialize.assert_not_called()
    expected_state = on_loop_start_input
    expected_round = 1

    for eval_fn, prefix in ((part_train_eval_fn,
                             metric_utils.PART_TRAIN_EVAL_METRICS_PREFIX),
                            (part_val_fn, metric_utils.PART_VAL_METRICS_PREFIX),
                            (unpart_fn, metric_utils.UNPART_METRICS_PREFIX)):
      if eval_fn is not None:
        mock_compute_eval_metrics.assert_any_call(expected_state,
                                                  expected_round - 1, eval_fn,
                                                  prefix)

    self.assertEqual(actual_state, expected_state)
    self.assertEqual(actual_round, expected_round)

  @mock.patch.object(federated_training_loop, '_load_initial_checkpoint')
  @mock.patch.object(federated_training_loop, '_compute_eval_metrics')
  def test_calls_with_metrics_managers_and_unpart_fn(self,
                                                     mock_compute_eval_metrics,
                                                     mock_initialize):
    metric_manager1 = mock.create_autospec(tff.simulation.MetricsManager)
    metric_manager2 = mock.create_autospec(tff.simulation.MetricsManager)
    metrics_managers = [metric_manager1, metric_manager2]
    unpart_fn = mock.MagicMock()
    metrics = collections.OrderedDict(metric1=2)
    mock_compute_eval_metrics.return_value = metrics
    on_loop_start_fn = federated_training_loop._create_on_loop_start_fn(
        metrics_managers=metrics_managers, unpart_fn=unpart_fn)
    on_loop_start_input = 'input'
    actual_state, actual_round = on_loop_start_fn(on_loop_start_input)
    mock_initialize.assert_not_called()
    expected_state = on_loop_start_input
    expected_round = 1

    mock_compute_eval_metrics.assert_any_call(
        expected_state, expected_round - 1, unpart_fn,
        metric_utils.UNPART_METRICS_PREFIX)

    for metr_mngr in metrics_managers:
      metr_mngr.clear_metrics.assert_called_once_with(0)
      metr_mngr.save_metrics.assert_called_once_with(metrics, 0)
    self.assertEqual(actual_state, expected_state)
    self.assertEqual(actual_round, expected_round)

  @mock.patch.object(federated_training_loop, '_load_initial_checkpoint')
  @mock.patch.object(federated_training_loop, '_compute_eval_metrics')
  def test_calls_with_non_zero_checkpoint_and_eval_fns(
      self, mock_compute_eval_metrics, mock_initialize):
    file_checkpoint_manager = mock.create_autospec(
        tff.simulation.FileCheckpointManager)
    part_train_eval_fn, part_val_fn, unpart_fn = mock.MagicMock(
    ), mock.MagicMock(), mock.MagicMock()
    expected_state = 'state'
    expected_round = 2
    mock_initialize.return_value = (expected_state, expected_round)
    on_loop_start_fn = federated_training_loop._create_on_loop_start_fn(
        file_checkpoint_manager=file_checkpoint_manager,
        part_train_eval_fn=part_train_eval_fn,
        part_val_fn=part_val_fn,
        unpart_fn=unpart_fn)
    on_loop_start_input = 'input'
    actual_state, actual_round = on_loop_start_fn(on_loop_start_input)
    mock_initialize.assert_called_once_with(on_loop_start_input,
                                            file_checkpoint_manager)
    mock_compute_eval_metrics.assert_not_called()
    file_checkpoint_manager.save_checkpoint.assert_not_called()
    self.assertEqual(actual_state, expected_state)
    self.assertEqual(actual_round, expected_round)


class CreateOnRoundEndTest(absltest.TestCase):

  @mock.patch.object(federated_training_loop, '_compute_eval_metrics')
  def test_calls_with_no_input_args(self, mock_compute_eval_metrics):
    on_round_end_fn = federated_training_loop._create_on_round_end_fn()
    state = 'state'
    round_num = 1
    metrics = {'metric': 1}
    actual_state, actual_metrics = on_round_end_fn(state, round_num, metrics)
    mock_compute_eval_metrics.assert_not_called()
    self.assertEqual(actual_state, state)
    self.assertEqual(actual_metrics, metrics)

  @mock.patch.object(federated_training_loop, '_compute_eval_metrics')
  def test_calls_with_only_checkpoint_manager(self, mock_compute_eval_metrics):
    file_checkpoint_manager = mock.create_autospec(
        tff.simulation.FileCheckpointManager)
    on_round_end_fn = federated_training_loop._create_on_round_end_fn(
        file_checkpoint_manager=file_checkpoint_manager)
    state = 'state'
    round_num = 1
    metrics = {'metric': 1}
    actual_state, actual_metrics = on_round_end_fn(state, round_num, metrics)
    mock_compute_eval_metrics.assert_not_called()
    file_checkpoint_manager.load_latest_checkpoint.assert_not_called()
    file_checkpoint_manager.save_checkpoint.assert_called_once_with(
        state, round_num)
    self.assertEqual(actual_state, state)
    self.assertEqual(actual_metrics, metrics)

  @mock.patch.object(federated_training_loop, '_compute_eval_metrics')
  def test_calls_with_only_metrics_managers(self, mock_compute_eval_metrics):
    mock_metrics_manager1 = mock.create_autospec(tff.simulation.MetricsManager)
    mock_metrics_manager2 = mock.create_autospec(tff.simulation.MetricsManager)
    metrics_managers = [mock_metrics_manager1, mock_metrics_manager2]
    on_round_end_fn = federated_training_loop._create_on_round_end_fn(
        metrics_managers=metrics_managers)
    state = 'state'
    round_num = 1
    metrics = {'metric': 1}
    actual_state, actual_metrics = on_round_end_fn(state, round_num, metrics)
    mock_compute_eval_metrics.assert_not_called()
    for mock_metrics_manager in metrics_managers:
      mock_metrics_manager.clear_metrics.assert_not_called()
      mock_metrics_manager.save_metrics.assert_called_once_with(
          metrics, round_num)
    self.assertEqual(actual_state, state)
    self.assertEqual(actual_metrics, metrics)

  @mock.patch.object(federated_training_loop, '_compute_eval_metrics')
  def test_calls_with_only_unpart_fn(self, mock_compute_eval_metrics):
    unpart_fn = mock.MagicMock()
    mock_compute_eval_metrics.return_value = {'unpart_metric': 2}
    on_round_end_fn = federated_training_loop._create_on_round_end_fn(
        unpart_fn=unpart_fn)
    state = 'state'
    round_num = 1
    metrics = {'metric': 1}
    actual_state, actual_metrics = on_round_end_fn(state, round_num, metrics)
    mock_compute_eval_metrics.assert_called_once_with(
        state, round_num, unpart_fn, metric_utils.UNPART_METRICS_PREFIX)
    self.assertEqual(actual_state, state)
    expected_metrics = {'metric': 1, 'unpart_metric': 2}
    self.assertDictEqual(actual_metrics, expected_metrics)

  @mock.patch.object(federated_training_loop, '_compute_eval_metrics')
  def test_calls_with_unpart_fn_and_metrics_managers(self,
                                                     mock_compute_eval_metrics):
    mock_metrics_manager1 = mock.create_autospec(tff.simulation.MetricsManager)
    mock_metrics_manager2 = mock.create_autospec(tff.simulation.MetricsManager)
    metrics_managers = [mock_metrics_manager1, mock_metrics_manager2]
    unpart_fn = mock.MagicMock()
    mock_compute_eval_metrics.return_value = {'unpart_metric': 2}
    on_round_end_fn = federated_training_loop._create_on_round_end_fn(
        metrics_managers=metrics_managers, unpart_fn=unpart_fn)

    state = 'input_state'
    round_num = 1
    metrics = collections.OrderedDict(metric=1)
    actual_state, actual_metrics = on_round_end_fn(state, round_num, metrics)
    mock_compute_eval_metrics.assert_called_once_with(
        state, round_num, unpart_fn, metric_utils.UNPART_METRICS_PREFIX)
    expected_metrics = {'metric': 1, 'unpart_metric': 2}
    for mock_metrics_manager in metrics_managers:
      mock_metrics_manager.clear_metrics.assert_not_called()
      mock_metrics_manager.save_metrics.assert_called_once_with(
          expected_metrics, round_num)
    self.assertEqual(actual_state, state)
    self.assertEqual(actual_metrics, expected_metrics)


class RunSimulationTest(parameterized.TestCase):

  @mock.patch.object(federated_training_loop, '_run_simulation_with_callbacks')
  @mock.patch.object(federated_training_loop, '_create_on_round_end_fn')
  @mock.patch.object(federated_training_loop, '_create_on_loop_start_fn')
  def test_run_simulation_passes_correctly_with_no_optional_arguments(
      self, mock_create_on_loop_start, mock_create_on_round_end,
      mock_run_simulation_with_callbacks):
    process = mock.create_autospec(tff.templates.IterativeProcess)
    client_selection_fn = lambda x: ()
    total_rounds = 10
    on_loop_start = 'on_loop_start'
    mock_create_on_loop_start.return_value = on_loop_start
    on_round_end = 'on_round_end'
    mock_create_on_round_end.return_value = on_round_end

    federated_training_loop.run_simulation(process, client_selection_fn,
                                           total_rounds)
    mock_create_on_loop_start.assert_called_once_with(None, None, None, None,
                                                      None)
    mock_create_on_round_end.assert_called_once_with(None, None, None, None,
                                                     None)
    mock_run_simulation_with_callbacks.assert_called_once_with(
        process, client_selection_fn, total_rounds, on_loop_start, on_round_end)

  @parameterized.named_parameters(
      (name, *args)
      for name, args in zip([f'case_{idx}' for idx in range(64)],
                            itertools.product([None, 'arg'], repeat=6)))
  @mock.patch.object(federated_training_loop, '_record_test_metrics')
  @mock.patch.object(federated_training_loop, '_run_simulation_with_callbacks')
  @mock.patch.object(federated_training_loop, '_create_on_round_end_fn')
  @mock.patch.object(federated_training_loop, '_create_on_loop_start_fn')
  def test_run_simulation_passes_named_optional_arguments_correctly(
      self, file_checkpoint_manager, metrics_managers, part_train_eval_fn,
      part_val_fn, unpart_fn, test_fn, mock_create_on_loop_start,
      mock_create_on_round_end, mock_run_simulation_with_callbacks,
      mock_record_test_metrics):
    process = mock.create_autospec(tff.templates.IterativeProcess)
    client_selection_fn = lambda x: ()
    total_rounds = 10
    on_loop_start = 'on_loop_start'
    mock_create_on_loop_start.return_value = on_loop_start
    on_round_end = 'on_round_end'
    mock_create_on_round_end.return_value = on_round_end
    final_state = 'final_state'
    mock_run_simulation_with_callbacks.return_value = final_state

    federated_training_loop.run_simulation(
        process,
        client_selection_fn,
        total_rounds,
        part_train_eval_fn=part_train_eval_fn,
        part_val_fn=part_val_fn,
        unpart_fn=unpart_fn,
        test_fn=test_fn,
        file_checkpoint_manager=file_checkpoint_manager,
        metrics_managers=metrics_managers,
    )
    mock_create_on_loop_start.assert_called_once_with(file_checkpoint_manager,
                                                      metrics_managers,
                                                      part_train_eval_fn,
                                                      part_val_fn, unpart_fn)
    mock_create_on_round_end.assert_called_once_with(file_checkpoint_manager,
                                                     metrics_managers,
                                                     part_train_eval_fn,
                                                     part_val_fn, unpart_fn)
    mock_run_simulation_with_callbacks.assert_called_once_with(
        process, client_selection_fn, total_rounds, on_loop_start, on_round_end)
    mock_record_test_metrics.assert_called_once_with(final_state, total_rounds,
                                                     test_fn, metrics_managers)


if __name__ == '__main__':
  absltest.main()
