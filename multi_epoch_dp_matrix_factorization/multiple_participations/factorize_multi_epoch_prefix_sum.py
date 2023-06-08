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
"""Binary for factorizing and writing multi-epoch prefix sum matrices."""

import collections
from collections.abc import Sequence
import os

from absl import app
from absl import flags
from absl import logging
from jax import numpy as jnp
import numpy as np
import optax
import pandas as pd
import tensorflow as tf

from multi_epoch_dp_matrix_factorization import matrix_constructors
from multi_epoch_dp_matrix_factorization import matrix_io
from multi_epoch_dp_matrix_factorization.multiple_participations import contrib_matrix_builders
from multi_epoch_dp_matrix_factorization.multiple_participations import lagrange_terms
from multi_epoch_dp_matrix_factorization.multiple_participations import lt_initializers
from multi_epoch_dp_matrix_factorization.multiple_participations import optimization
from utils import training_utils


# Unused import to declare XM flags.

IRRELEVANT_FLAGS = frozenset(iter(flags.FLAGS))

_INIT_MATRIX_DIR = flags.DEFINE_string(
    'init_matrix_dir',
    '',
    'Directory from which to load optional initial matrix',
)
_NUM_EPOCHS = flags.DEFINE_integer(
    'num_epochs', 1, 'Number of epochs for which to optimize'
)
_STEPS_PER_EPOCH = flags.DEFINE_integer(
    'steps_per_epoch', 1, 'Number of steps in each epoch.'
)
_CONSTRAINT_PATTERN = flags.DEFINE_enum(
    'constraint_pattern',
    'all_positive',
    ['all_positive', 'none'],
    'Additional constraints on the factorization.',
)

_NN_LR = flags.DEFINE_float(
    'nn_lr', 0.01, 'Learning rate for nonnegativity optimizer.'
)
_NN_MOMENTUM = flags.DEFINE_float(
    'nn_momentum', 0.95, 'Momentum value for nonnegativity optimizer.'
)
_PREFIX_SUM = 'prefix_sum'
_MOMENTUM_COOLDOWN = 'momentum_with_cooldown'
_MATRIX_TO_FACTOR = flags.DEFINE_enum(
    'matrix_to_factor',
    _PREFIX_SUM,
    [_PREFIX_SUM, _MOMENTUM_COOLDOWN],
    (
        'Which matrix to factor, either the standard prefix sum matrix, '
        'or the matrix with momentum and learning-rate cooldown. The '
        'lr schedule and momentum parameter are currently hard-coded.'
    ),
)

# Operational parameters
_MAX_ITER = flags.DEFINE_integer(
    'max_iterations', 10, 'Maximum number of steps to take.'
)
_STEPS_PER_EVAL = flags.DEFINE_integer(
    'steps_per_eval', 1, 'Number of steps to take between evalautions.'
)
_REL_DUALITY_GAP = flags.DEFINE_float(
    'target_relative_duality_gap',
    1e-3,
    'Relative duality gap to use as stopping criterion.',
)
_ROOT_DIR = flags.DEFINE_string(
    'root_output_dir', '', 'Directory to write matrices and loss values to.'
)
_RUN_NAME = flags.DEFINE_string(
    'run_name',
    '',
    (
        'Unique experiment name. Will be appended to root_output_dir to'
        ' uniquify results.'
    ),
)

HPARAM_FLAGS = [f for f in flags.FLAGS if f not in IRRELEVANT_FLAGS]

FLAGS = flags.FLAGS


def _get_lagrange_terms() -> lagrange_terms.LagrangeTerms:
  """Constructs initial LagrangeTerms based on flags."""
  constraint_pattern = _CONSTRAINT_PATTERN.value
  if constraint_pattern == 'all_positive':
    if _INIT_MATRIX_DIR.value:
      return lt_initializers.init_nonnegative_lagrange_terms_from_path(
          _INIT_MATRIX_DIR.value, _NUM_EPOCHS.value, _STEPS_PER_EPOCH.value
      )
    else:
      return lt_initializers.init_nonnegative_lagrange_terms(
          _NUM_EPOCHS.value, _STEPS_PER_EPOCH.value
      )
  elif constraint_pattern == 'none':
    n = _NUM_EPOCHS.value * _STEPS_PER_EPOCH.value
    contrib_matrix = contrib_matrix_builders.epoch_participation_matrix(
        n, _NUM_EPOCHS.value
    )
    return lagrange_terms.init_lagrange_terms(contrib_matrix)
  else:
    raise ValueError(f'Unknown --constraint_pattern {constraint_pattern}')


def get_lr_schedule(n):
  # Hard-coded for now based on previous experiments
  cooldown_period = n // 4
  cooldown_target = 0.05
  lr = np.ones(n)
  lr[-cooldown_period:] = np.linspace(1.0, cooldown_target, num=cooldown_period)
  return lr


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  hparam_dict = collections.OrderedDict(
      [(name, FLAGS[name].value) for name in HPARAM_FLAGS]
  )

  program_state_manager, (_, csv_manager, tb_manager) = (
      training_utils.create_managers(
          root_dir=_ROOT_DIR.value,
          experiment_name=_RUN_NAME.value,
      )
  )
  training_utils.write_hparams_to_csv(
      hparam_dict=hparam_dict,
      root_output_dir=_ROOT_DIR.value,
      experiment_name=_RUN_NAME.value,
  )

  n = _NUM_EPOCHS.value * _STEPS_PER_EPOCH.value

  flags.FLAGS.matrix_root_path = os.path.join(_ROOT_DIR.value, _RUN_NAME.value)
  if _MATRIX_TO_FACTOR.value == _PREFIX_SUM:
    s_matrix = jnp.tri(n, dtype=jnp.float64)
    # We need to ensure we write with a path matrix_io.get_prefix_sum_w_h can
    # load via aggregator_builder.py
    output_dir = matrix_io.get_matrix_path(n, matrix_io.PREFIX_OPT)
    learning_rates = None
  else:
    momentum = 0.95  # Hard-coded for now:
    learning_rates = get_lr_schedule(n)
    s_matrix = matrix_constructors.momentum_sgd_matrix(
        n, momentum, learning_rates
    )
    # We need to ensure we write with a path that aggregator_builder.py
    # can reconstruct via the lr_momentum_matrix codepath from which
    # momentum can be inferred.
    output_dir = matrix_io.get_momentum_path(n, momentum)

  assert s_matrix.dtype == np.float64

  lt = _get_lagrange_terms()
  lt.assert_valid()

  logging.info('Calling into Lagrange dual problem solver.')
  results = optimization.solve_lagrange_dual_problem(
      s_matrix=s_matrix,
      lt=lt,
      update_langrange_terms_fn=optimization.OptaxUpdate(
          # Larger problems seem to need smaller learning rates in order to
          # not produce non-PD u_total() matrices; see comments
          # on OptaxUpdate
          nonneg_optimizer=optax.sgd(_NN_LR.value, momentum=_NN_MOMENTUM.value),
          lt=lt,
          multiplicative_update=True,
      ),
      max_iterations=_MAX_ITER.value,
      iters_per_eval=_STEPS_PER_EVAL.value,
      target_relative_duality_gap=_REL_DUALITY_GAP.value,
      program_state_manager=program_state_manager,
      metric_release_managers=(csv_manager, tb_manager),
  )

  logging.info('Writing final results to %s', output_dir)
  tf.io.gfile.makedirs(output_dir)
  loss_csv_filename = os.path.join(output_dir, 'losses.csv')

  logging.info('Final loss: %s', results['losses'][-1])
  logging.info('Final dual objective value: %s', results['dual_obj_vals'][-1])
  df_arg = {}
  for col_name in ['losses', 'dual_obj_vals']:
    df_arg[col_name] = results[col_name]
  df = pd.DataFrame(df_arg)

  # TODO(b/241453645): Move formatting as a dataframe into the optimize
  # function itself? Serialize not-per-round entries in results as well?
  df.to_csv(loss_csv_filename)

  matrix_io.verify_and_write(
      results['W'], results['H'], s_matrix, output_dir, lr_sched=learning_rates
  )


if __name__ == '__main__':
  app.run(main)
