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
"""Libraries for auditing FL models with canary insertion."""

import collections
from collections.abc import Callable
import math
import os.path
import time
from typing import Any, Optional

from absl import logging
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from one_shot_epe import dot_product_utils
from utils import utils_impl
from tensorboard.plugins.hparams import api as hp


def build_bernoulli_sampling_fn(
    client_ids: list[str],
    expected_clients_per_round: float,
    seed: int,
) -> Callable[[int], list[str]]:
  """Builds Bernoulli sampling function.

  Args:
    client_ids: A list of client_ids.
    expected_clients_per_round: The expected number of clients per round.
    seed: The random seed. If None, randomness is seeded nondeterministically.

  Returns:
    Sampling function that selects clients i.i.d. with mean
    expected_clients_per_round.
  """
  p_select = expected_clients_per_round / len(client_ids)
  if p_select > 1:
    raise ValueError(
        'Expected clients per round cannot exceed number of client IDs. Found '
        f'{len(client_ids)} client ids with {expected_clients_per_round} '
        'expected_clients_per_round.'
    )
  uniform_sampling_fn = tff.simulation.build_uniform_sampling_fn(
      client_ids, random_seed=seed
  )

  if seed is None:
    seed = time.time_ns()

  def sampling_fn(round_num):
    size = tf.random.stateless_binomial(
        (), (round_num, seed), len(client_ids), p_select
    )
    return uniform_sampling_fn(round_num, size)

  return sampling_fn


def build_shuffling_sampling_fn(
    real_client_ids: list[str],
    canary_client_ids: list[str],
    total_rounds: int,
    num_real_epochs: int,
    canary_repeats: int,
    seed: int,
) -> tuple[Callable[[int], list[str]], float]:
  """Builds shuffling fn with canaries cycled through `canary_repeats` times.

  Args:
    real_client_ids: The list of real client ids.
    canary_client_ids: The list of canary client ids.
    total_rounds: The total number of rounds.
    num_real_epochs: The number of times each real client is seen. Must divide
      total_rounds.
    canary_repeats: The number times each canary is seen.
    seed: The random seed. If None, randomness is seeded nondeterministically.

  Returns:
    Sampling function that shuffles and cycles through real clients a specified
    number of times, and also cycles through canaries a specified number of
    times.
  """
  if not real_client_ids:
    raise ValueError('`real_client_ids` cannot be empty.')
  if total_rounds <= 0:
    raise ValueError(
        f'`total_rounds` must be greater than 0. Found `{total_rounds}`.'
    )
  if num_real_epochs <= 0:
    raise ValueError(
        f'`num_real_epochs` must be greater than 0. Found `{num_real_epochs}`.'
    )
  if canary_repeats < 0:
    raise ValueError(
        f'`canary_repeats` must be nonnegative. Found `{canary_repeats}`.'
    )

  rng = np.random.default_rng(seed)

  num_real = len(real_client_ids)
  real_order = rng.permutation(num_real)
  num_canary = len(canary_client_ids)
  canary_order = [rng.permutation(num_canary) for _ in range(canary_repeats)]

  def sampling_fn(round_num: int) -> list[str]:
    if not 1 <= round_num <= total_rounds:
      raise ValueError(
          f'round_num ({round_num}) must be between 1 and total_rounds'
          f' ({total_rounds}).'
      )
    # tff.simulation.run_training_process uses rounds 1 ... total_rounds.
    # For the math here to work we need 0 ... (total_rounds - 1).
    round_num -= 1

    begin_real = math.ceil(
        round_num * num_real * num_real_epochs / total_rounds
    )
    end_real = math.ceil(
        (round_num + 1) * num_real * num_real_epochs / total_rounds
    )
    real_sample = [
        real_client_ids[real_order[i % num_real]]
        for i in range(begin_real, end_real)
    ]

    begin_canary = math.ceil(
        round_num * num_canary * canary_repeats / total_rounds
    )
    end_canary = math.ceil(
        (round_num + 1) * num_canary * canary_repeats / total_rounds
    )
    canary_sample = [
        canary_client_ids[canary_order[i // num_canary][i % num_canary]]
        for i in range(begin_canary, end_canary)
    ]

    return real_sample + canary_sample

  mean_clients_per_round = (
      len(real_client_ids) * num_real_epochs
      + len(canary_client_ids) * canary_repeats
  ) / total_rounds

  return sampling_fn, mean_clients_per_round


async def compute_and_release_final_model_canary_metrics(
    release_manager: tff.program.CSVFileReleaseManager,
    model_weights: tff.learning.models.ModelWeights,
    canary_seed: int,
    max_canary_model_delta_cosines: Any,
    max_unseen_canary_model_delta_cosines: Any,
) -> None:
  """Computes and writes final model cosines."""
  num_canaries = len(max_canary_model_delta_cosines)
  cosines = dot_product_utils.compute_negative_cosines_with_all_canaries(
      model_weights.trainable,
      num_canaries,
      canary_seed,
  )
  final_metrics = collections.OrderedDict(
      (f'final_cosines/canary:{i}', cosines[i]) for i in range(num_canaries)
  )
  final_metrics.update(
      (f'max_model_delta_cosines/canary:{i}', max_canary_model_delta_cosines[i])
      for i in range(num_canaries)
  )
  max_unseen = max_unseen_canary_model_delta_cosines
  num_unseen_canaries = len(max_unseen)
  final_metrics.update(
      (f'max_model_delta_cosines/unseen_canary:{i}', max_unseen[i])
      for i in range(num_unseen_canaries)
  )
  metrics_type = tff.types.type_conversions.infer_type(final_metrics)
  await release_manager.release(final_metrics, metrics_type, 0)


def create_managers(
    root_dir: str,
    experiment_name: str,
    hparam_dict: Optional[dict[str, Any]] = None,
    csv_save_mode: tff.program.CSVSaveMode = tff.program.CSVSaveMode.WRITE,
) -> tuple[
    tff.program.FileProgramStateManager, list[tff.program.ReleaseManager]
]:
  """Creates a set of managers for running a simulation.

  The managers that are created and how they are configured are intended to be
  used with `tff.simulation.run_training_process` to run a simulation.

  Args:
    root_dir: A string representing the root output directory for the
      simulation.
    experiment_name: A unique identifier for the simulation, used to create
      appropriate subdirectories in `root_dir`.
    hparam_dict: A dictionary of hyperparameters for the run.
    csv_save_mode: A `tff.program.CSVSaveMode` specifying the save mode for the
      `tff.program.CSVFileReleaseManager`.

  Returns:
    A `tff.program.FileProgramStateManager`, and a list of
    `tff.program.ReleaseManager`s consisting of a
    `tff.program.LoggingReleaseManager`, a `tff.program.CSVFileReleaseManager`,
    and a `tff.program.TensorBoardReleaseManager`.
  """
  program_state_dir = os.path.join(root_dir, 'checkpoints', experiment_name)
  program_state_manager = tff.program.FileProgramStateManager(
      root_dir=program_state_dir, keep_total=1, keep_first=False
  )

  logging_release_manager = tff.program.LoggingReleaseManager()

  results_dir = os.path.join(root_dir, 'results', experiment_name)
  csv_file_path = os.path.join(results_dir, 'experiment.metrics.csv')
  csv_file_release_manager = tff.program.CSVFileReleaseManager(
      file_path=csv_file_path,
      save_mode=csv_save_mode,
      key_fieldname='round_num',
  )

  summary_dir = os.path.join(root_dir, 'logdir', experiment_name)
  tensorboard_manager = tff.program.TensorBoardReleaseManager(summary_dir)

  if hparam_dict:
    summary_writer = tf.summary.create_file_writer(summary_dir)
    hparam_dict['metrics_file'] = csv_file_path
    hparams_file = os.path.join(results_dir, 'hparams.csv')
    utils_impl.atomic_write_series_to_csv(hparam_dict, hparams_file)
    with summary_writer.as_default():
      hp.hparams({k: v for k, v in hparam_dict.items() if v is not None})

  logging.info('Writing...')
  logging.info('    program state to: %s', program_state_dir)
  logging.info('    CSV metrics to: %s', csv_file_path)
  logging.info('    TensorBoard summaries to: %s', summary_dir)
  return program_state_manager, [
      logging_release_manager,
      csv_file_release_manager,
      tensorboard_manager,
  ]
