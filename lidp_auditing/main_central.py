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
"""Launch training + auditing experiments."""

from collections.abc import Sequence
import sys

from absl import app
from absl import flags
from absl import logging
import numpy as np
import pandas as pd  # pylint:disable=unused-import
import tensorflow as tf

from lidp_auditing import auditing_trainer
from lidp_auditing import constants
from lidp_auditing import data
from lidp_auditing import models
from lidp_auditing import utils

# General flags
_EXPERIMENT_NAME = flags.DEFINE_string(
    'experiment_name',
    None,
    (
        'The name of this experiment. Will be appended to '
        '--output_dir to separate experiment results.'
    ),
    required=True,
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir', None, 'The name of the output directory.', required=True
)
# The output is saved in the folder {_OUTPUT_DIR}/{_EXPERIMENT_NAME}
_DATASET_NAME = flags.DEFINE_enum(
    'dataset_name',
    None,
    constants.DATASET_NAMES,
    'Which dataset to use for experiments.',
    required=True,
)
_CSV_DATA_PATH = flags.DEFINE_string(
    'csv_data_path', None, 'Path to load the data from (if applicable)'
)
_CANARY_TYPE = flags.DEFINE_enum(
    'canary_type',
    None,
    constants.CANARY_TYPES,
    'Which type of canary to use for experiments.',
    required=True,
)
_NUM_CANARIES = flags.DEFINE_integer(
    'num_canaries',
    None,
    'Number of canaries',
    required=True,
)
_NUM_CANARIES_TEST = flags.DEFINE_integer(
    'num_canaries_test',
    None,
    'Number of test canaries',
)
_TEST_CANARY_ADD_ONE = flags.DEFINE_bool(
    'test_canary_add_one',
    False,
    'If true, use an additional test canary',
)
_SEED = flags.DEFINE_integer('seed', None, 'Seed', required=True)
_NUM_SEEDS_IN_RUN = flags.DEFINE_integer(
    'num_seeds_in_run', 1, 'Number of Seeds to run'
)
_CANARY_DATA_SCALE = flags.DEFINE_float(
    'canary_data_scale', 1.0, 'Scaling factor for static/adaptive data canaries'
)
_CANARY_CLASS = flags.DEFINE_integer('canary_class', None, 'Canary class')
_MIN_DIMENSION = flags.DEFINE_integer(
    'min_dimension',
    0,
    'Smallest dimension to use for data poisoning',
    lower_bound=0,
    upper_bound=int(1e10),
)
_MAX_DIMENSION = flags.DEFINE_integer(
    'max_dimension',
    1000,
    'Largest dimension to use for data poisoning',
    lower_bound=0,
    upper_bound=int(1e10),
)


@flags.multi_flags_validator(
    ['min_dimension', 'max_dimension'],
    message='Minimum is not smaller than maximum',
)
def CheckMinMaxFlags(flags_dict):
  return flags_dict['min_dimension'] <= flags_dict['max_dimension']


_MODEL_TYPE = flags.DEFINE_enum(
    'model_type',
    None,
    constants.MODEL_TYPES,
    'Which model architecture to use.',
    required=True,
)
_HIDDEN_DIM = flags.DEFINE_integer('hidden_dim', 256, 'Hidden dimension')

# Training and privacy flags
_LEARNING_RATE = flags.DEFINE_float(
    'learning_rate', 1e-2, 'Optimizer learning rate', lower_bound=0
)
_BATCH_SIZE_TRAIN = flags.DEFINE_integer(
    'batch_size_train', 100, 'Training batch size', lower_bound=1
)
_BATCH_SIZE_TEST = flags.DEFINE_integer(
    'batch_size_test', 1000, 'Eval batch size', lower_bound=1
)
_NUM_EPOCHS = flags.DEFINE_integer(
    'num_epochs', 50, 'Number of train epochs', lower_bound=1, upper_bound=100
)
_RUN_NONPRIVATE = flags.DEFINE_bool(
    'run_nonprivate', False, 'if true, run without DP noise'
)
_L2_CLIP_NORM = flags.DEFINE_float(
    'l2_clip_norm', None, 'gradient clip for DP-SGD', lower_bound=0
)
_DP_EPSILON = flags.DEFINE_float('dp_epsilon', 10, 'target eps', lower_bound=0)
_DP_DELTA = flags.DEFINE_float(
    'dp_delta', 1e-5, 'target delta', lower_bound=0, upper_bound=1
)
_VALIDATION_MODE = flags.DEFINE_bool(
    'validation_mode', False, 'use a part of the training data for eval'
)


def GetNumTestCanaries() -> int | None:
  """Number of test canaries (using number of train canaries as default)."""
  if _NUM_CANARIES_TEST.value is not None:
    return _NUM_CANARIES_TEST.value
  elif _TEST_CANARY_ADD_ONE.value:
    return _NUM_CANARIES.value + 1
  else:
    return _NUM_CANARIES.value


def GetClipNormFromArgs() -> float:
  """Get clip norm based on tuned hyperparameters."""
  return constants.get_clip_norm(
      _DATASET_NAME.value,
      _MODEL_TYPE.value,
      _DP_EPSILON.value,
      _RUN_NONPRIVATE.value,
  )


def MainForSeed(seed: int) -> None:
  """Run the main computations for a single seed."""
  tf.random.set_seed(seed + 100)  # Fix initial seed

  # Save file name
  dsinfo = (
      f'{_DATASET_NAME.value}_{_CANARY_TYPE.value}_k{_NUM_CANARIES.value}'
      f'_c{_CANARY_DATA_SCALE.value}_dim{_MIN_DIMENSION.value}'
  )
  eps = (
      'inf'
      if _RUN_NONPRIVATE.value or _DP_EPSILON.value >= 1e5
      else _DP_EPSILON.value
  )
  l2_clip_norm = (
      GetClipNormFromArgs()
      if _L2_CLIP_NORM.value is None
      else _L2_CLIP_NORM.value
  )
  optim_info = (
      f'lr{_LEARNING_RATE.value}_norm{l2_clip_norm}'
      f'_eps{eps}_delta{_DP_DELTA.value}'
  )
  suffix_val = '_val' if _VALIDATION_MODE.value else ''
  savefilename = (
      f'{_OUTPUT_DIR.value}/{_EXPERIMENT_NAME.value}/logdir/'
      f'{dsinfo}_{optim_info}_seed{seed}{suffix_val}.csv'
  )
  logging.warning('***Output filename: %s', savefilename)

  # Load data
  num_train, datasets = data.get_datasets(
      _DATASET_NAME.value,
      _CANARY_TYPE.value,
      _NUM_CANARIES.value,
      GetNumTestCanaries(),
      seed,
      _CANARY_DATA_SCALE.value,
      _MIN_DIMENSION.value,
      _MAX_DIMENSION.value,
      _CANARY_CLASS.value,
      csv_data_path=_CSV_DATA_PATH.value,
      validation_mode=_VALIDATION_MODE.value,
  )

  # Setup the models
  model = models.get_model_for_dataset(
      _DATASET_NAME.value, _MODEL_TYPE.value, hidden_units=_HIDDEN_DIM.value
  )
  num_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
  logging.warning('Model setup done. #Params = %d', num_params)

  # DP accounting
  assert num_train % _BATCH_SIZE_TRAIN.value == 0, 'batch size should divide n'
  if _RUN_NONPRIVATE.value or _DP_EPSILON.value >= 1e5:  # no DP
    noise_multiplier = 0.0
    logging.warning('Running with no DP!')
  else:
    noise_multiplier = utils.get_optimal_noise_multiplier_dpsgd(
        num_train,
        _BATCH_SIZE_TRAIN.value,
        _NUM_EPOCHS.value,
        _DP_EPSILON.value,
        _DP_DELTA.value,
    )
  logging.warning('Using noise multiplier %f', noise_multiplier)

  # Run
  logging.warning('Starting training')
  logs = auditing_trainer.run_training_with_canaries(
      datasets,
      model,
      _CANARY_TYPE.value,
      _NUM_EPOCHS.value,
      _BATCH_SIZE_TRAIN.value,
      _BATCH_SIZE_TEST.value,
      _LEARNING_RATE.value,
      l2_clip_norm,
      noise_multiplier,
      seed,
      savefilename,
  )
  logging.warning('Done training')

  # Save
  logs.to_csv(savefilename)
  logging.warning('Saved logs to: %s', savefilename)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  logging.warning('Starting! Command line flags:')
  flag_dict = {
      f.name: f.value for f in flags.FLAGS.get_flags_for_module(__name__)
  }
  logging.warning(flag_dict)
  logging.flush()
  # To print out our logs correctly, disable numpy summarization.
  np.set_printoptions(threshold=sys.maxsize)

  for i in range(_NUM_SEEDS_IN_RUN.value):
    seed = _SEED.value + i
    logging.warning(
        'Starting seed = %d (%d / %d, %f percent)',
        seed,
        i,
        _NUM_SEEDS_IN_RUN.value,
        round(i / _NUM_SEEDS_IN_RUN.value * 100, 2),
    )
    MainForSeed(seed)
    logging.warning(
        'Done seed = %d (index = %d, total = %d)',
        seed,
        i,
        _NUM_SEEDS_IN_RUN.value,
    )


if __name__ == '__main__':
  app.run(main)
