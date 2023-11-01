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
import csv
import os
import unittest

from absl.testing import parameterized
import numpy as np
from scipy import stats
import tensorflow as tf
import tensorflow_federated as tff

from one_shot_epe import dot_product_utils
from one_shot_epe import train_lib

_TEST_SEED = 0xBAD5EED


def _read_values_from_csv(file_path):
  with tf.io.gfile.GFile(file_path, 'r') as file:
    reader = csv.DictReader(file)
    fieldnames = list(reader.fieldnames)
    values = list(reader)
  return fieldnames, values


class TrainLibTest(
    tf.test.TestCase, parameterized.TestCase, unittest.IsolatedAsyncioTestCase
):

  @parameterized.named_parameters(
      ('0', 0),
      ('0.1', 0.1),
      ('0.5', 0.5),
      ('0.99', 0.99),
  )
  def test_bernoulli_sample(self, sampling_frequency):
    num_clients = 100
    expected_clients_per_round = num_clients * sampling_frequency
    client_ids = [f'{i}' for i in range(num_clients)]
    times_selected = [0] * num_clients

    sample_fn = train_lib.build_bernoulli_sampling_fn(
        client_ids, expected_clients_per_round, _TEST_SEED
    )

    num_trials = 1000
    for i in range(num_trials):
      sample = sample_fn(i)
      for client_id in sample:
        times_selected[int(client_id)] += 1

    p_test_fail = 0.01  # Max probability of falsely failing the test.
    required_confidence = (1 - p_test_fail) ** (1 / num_clients)
    interval = stats.binom.interval(
        required_confidence, num_trials, sampling_frequency
    )

    for i in range(num_clients):
      self.assertAllInRange(times_selected[i], *interval)

  @parameterized.product(
      num_real=[31],
      num_canary=[0, 17],
      total_rounds=[12, 28, 36],
      num_real_epochs=[1, 2, 3],
      canary_repeats=[0, 1, 2, 3],
  )
  def test_build_shuffling_sampling_fn(
      self, num_real, num_canary, total_rounds, num_real_epochs, canary_repeats
  ):
    """Tests build_shuffling_sampling_fn.

    For various combinations and corner cases of real/canary clients, total
    rounds and canary repeats, verifies that each real client is seen exactly
    "num_real_epochs" times and each canary client is seen "canary_repeats"
    times.

    Args:
      num_real: The number of real clients.
      num_canary: The number of canary clients.
      total_rounds: The total number of rounds.
      num_real_epochs: The number of real repeats.
      canary_repeats: The number of canary repeats.
    """

    real_client_ids = [f'real_{i}' for i in range(num_real)]
    canary_client_ids = [f'canary_{i}' for i in range(num_canary)]
    sample_fn, mean_clients_per_round = train_lib.build_shuffling_sampling_fn(
        real_client_ids,
        canary_client_ids,
        total_rounds,
        num_real_epochs,
        canary_repeats,
        _TEST_SEED,
    )

    total_clients_sampled = 0
    real_selected, canary_selected = [], []
    for round_num in range(total_rounds):
      sample = sample_fn(round_num + 1)
      for client_id in sample:
        prefix, suffix = client_id.split('_', 1)
        if prefix == 'real':
          real_selected.append(int(suffix))
        else:
          canary_selected.append(int(suffix))
      self.assertAllInRange(
          len(sample), mean_clients_per_round - 2, mean_clients_per_round + 2
      )
      total_clients_sampled += len(sample)

    true_mean_clients_per_round = total_clients_sampled / total_rounds
    self.assertAllClose(true_mean_clients_per_round, mean_clients_per_round)

    rng = np.random.default_rng(_TEST_SEED)
    real_order = rng.permutation(len(real_client_ids))
    expected_real = list(real_order) * num_real_epochs
    self.assertListEqual(real_selected, expected_real)

    expected_canary = []
    for _ in range(canary_repeats):
      expected_canary.extend(rng.permutation(num_canary))
    self.assertListEqual(canary_selected, expected_canary)

  def test_shuffling_sample_fn_fail_bad_round_number(self):
    num_real = 53
    num_canary = 19
    total_rounds = 17
    real_client_ids = [f'real_{i}' for i in range(num_real)]
    canary_client_ids = [f'canary_{i}' for i in range(num_canary)]
    sample_fn, _ = train_lib.build_shuffling_sampling_fn(
        real_client_ids,
        canary_client_ids,
        total_rounds=total_rounds,
        num_real_epochs=1,
        canary_repeats=3,
        seed=_TEST_SEED,
    )

    with self.assertRaises(ValueError):
      sample_fn(0)

    with self.assertRaises(ValueError):
      sample_fn(total_rounds + 1)

  async def test_compute_and_release_final_model_canary_metrics(self):
    num_canaries = 2
    dim = 3
    canary_seed = _TEST_SEED
    file_path = self.create_tempfile()
    os.remove(file_path)
    release_manager = tff.program.CSVFileReleaseManager(
        file_path=file_path, save_mode=tff.program.CSVSaveMode.WRITE
    )

    max_cosines = np.random.uniform(-1.0, 1.0, num_canaries)
    max_unseen_cosines = np.random.uniform(-1.0, 1.0, num_canaries)

    model_weights = tff.learning.models.ModelWeights(
        tf.random.stateless_normal((dim,), (0, 1)), None
    )
    await train_lib.compute_and_release_final_model_canary_metrics(
        release_manager,
        model_weights,
        canary_seed,
        max_cosines,
        max_unseen_cosines,
    )
    actual_fieldnames, actual_values = _read_values_from_csv(file_path)
    expected_fieldnames = ['key']
    expected_fieldnames += [
        f'final_cosines/canary:{c}' for c in range(num_canaries)
    ]
    expected_fieldnames += [
        f'max_model_delta_cosines/canary:{c}' for c in range(num_canaries)
    ]
    expected_fieldnames += [
        f'max_model_delta_cosines/unseen_canary:{c}'
        for c in range(num_canaries)
    ]
    self.assertCountEqual(actual_fieldnames, expected_fieldnames)

    for c in range(num_canaries):
      expected = -dot_product_utils.compute_cosine(
          model_weights.trainable, c, canary_seed
      )
      actual = float(actual_values[0][f'final_cosines/canary:{c}'])
      self.assertAllClose(actual, expected)

    for c in range(num_canaries):
      self.assertAllClose(
          float(actual_values[0][f'max_model_delta_cosines/canary:{c}']),
          max_cosines[c],
      )
      self.assertAllClose(
          float(actual_values[0][f'max_model_delta_cosines/unseen_canary:{c}']),
          max_unseen_cosines[c],
      )


class CreateManagersTest(parameterized.TestCase):

  def test_create_managers_returns_managers(self):
    root_dir = self.create_tempdir()

    file_program_state_manager, release_managers = train_lib.create_managers(
        root_dir=root_dir, experiment_name='test'
    )

    self.assertIsInstance(
        file_program_state_manager, tff.program.FileProgramStateManager
    )
    self.assertLen(release_managers, 3)
    self.assertIsInstance(
        release_managers[0], tff.program.LoggingReleaseManager
    )
    self.assertIsInstance(
        release_managers[1], tff.program.CSVFileReleaseManager
    )
    self.assertIsInstance(
        release_managers[2], tff.program.TensorBoardReleaseManager
    )

  @unittest.mock.patch.object(tff.program, 'TensorBoardReleaseManager')
  @unittest.mock.patch.object(tff.program, 'CSVFileReleaseManager')
  @unittest.mock.patch.object(tff.program, 'LoggingReleaseManager')
  @unittest.mock.patch.object(tff.program, 'FileProgramStateManager')
  def test_create_managers_creates_managers(
      self,
      mock_file_program_state_manager,
      mock_logging_release_manager,
      mock_csv_file_release_manager,
      mock_tensorboard_release_manager,
  ):
    root_dir = self.create_tempdir()
    experiment_name = 'test'
    csv_save_mode = tff.program.CSVSaveMode.APPEND

    train_lib.create_managers(
        root_dir=root_dir,
        experiment_name=experiment_name,
        csv_save_mode=csv_save_mode,
    )

    program_state_dir = os.path.join(root_dir, 'checkpoints', experiment_name)
    mock_file_program_state_manager.assert_called_with(
        root_dir=program_state_dir, keep_total=1, keep_first=False
    )
    mock_logging_release_manager.assert_called_once_with()
    csv_file_path = os.path.join(
        root_dir, 'results', experiment_name, 'experiment.metrics.csv'
    )
    mock_csv_file_release_manager.assert_called_once_with(
        file_path=csv_file_path,
        save_mode=csv_save_mode,
        key_fieldname='round_num',
    )
    summary_dir = os.path.join(root_dir, 'logdir', experiment_name)
    mock_tensorboard_release_manager.assert_called_once_with(summary_dir)


if __name__ == '__main__':
  tf.test.main()
