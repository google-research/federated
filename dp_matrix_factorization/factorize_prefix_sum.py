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
"""Binary to factor prefix-sum matrix with minimal reconstruction error."""

import collections
import os
from typing import Any, Mapping, Sequence

from absl import app
from absl import flags
from absl import logging
from jax import numpy as jnp
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff

from dp_matrix_factorization import initializers
from dp_matrix_factorization import loops

FLAGS = flags.FLAGS

flags.DEFINE_enum('method', 'fixed_point_iteration',
                  ['fixed_point_iteration', 'gradient_descent'],
                  'Method to use for computing the optimum.')

# Parameters used to define initialization of H, implicitly defining the space
# over which to search for gradient descent.
flags.DEFINE_enum(
    'strategy', 'binary_tree', initializers.SUPPORTED_STRATEGIES,
    'Strategy to use for initialization of H, therefore defining the search '
    'space.')

# Strategy-specific flags; depending on strategy value above, some may be
# unused.
flags.DEFINE_integer('log_2_observations', 5,
                     'Log-base-two of the number of elements to aggregate.')
flags.DEFINE_integer(
    'num_extra_intermediate_dimensions', 0,
    'Number of dimensions to add to binary tree matrix representation of data.')
flags.DEFINE_string(
    'h_tensor_proto_path', '',
    'Path to existing h tensor proto to use for doubling initialization.')

flags.DEFINE_float(
    'target_relative_duality_gap', 0.001,
    'Continue fixed point iterations until '
    '`max_fixed_point_iterations` is reached, or '
    '`suboptimality/optimal_loss_lower_bound <= target_relative_duality_gap.`')

flags.DEFINE_integer(
    'max_iterations', 1000,
    'Maximum iterations to use to compute fixed point of phi.')

# Optimization parameters
flags.DEFINE_integer('n_iters', 100,
                     'Number of gradient-descent iterations to train for.')
flags.DEFINE_float('learning_rate', 0.1,
                   'Learning rate to use for gradient descent.')
flags.DEFINE_bool(
    'streaming', True,
    'Whether the matrix factorization should respect streaming constraints, '
    'or can be dense.')

# Operational parameters
flags.DEFINE_string('root_output_dir', '',
                    'Directory to write matrices and loss values to.')
flags.DEFINE_string(
    'experiment_name', '',
    'Name for this experiment. Will be appended to root_output_dir to separate results.'
)
flags.DEFINE_integer('steps_per_checkpoint', 10,
                     'Number of steps to wait between writing checkpoints.')


def _get_parameters_for_init_strategy(strategy: str) -> Mapping[str, Any]:
  """Packages flags into dictionary for consumption by initializers library."""
  if strategy == 'binary_tree' or strategy == 'identity' or strategy == 'random_binary_tree_structure':
    # Each of these only needs appropriate dimensionality to self-initialize.
    return collections.OrderedDict(log_2_leaves=FLAGS.log_2_observations)
  elif strategy == 'extended_binary_tree':
    return collections.OrderedDict(
        log_2_leaves=FLAGS.log_2_observations,
        num_extra_rows=FLAGS.num_extra_intermediate_dimensions)
  elif strategy == 'double_h_solution':
    if not FLAGS.h_tensor_proto_path:
      raise ValueError(
          'If using double_h_solution strategy, existing H tensor must be provided as a proto filepath.'
      )
    h_matrix_proto_contents = tf.io.read_file(FLAGS.h_tensor_proto_path)
    h_matrix = tf.io.parse_tensor(h_matrix_proto_contents, tf.float64)
    return collections.OrderedDict(h_to_double=h_matrix)
  else:
    raise ValueError(f'Unknown initialization strategy: {FLAGS.strategy}. '
                     f'Expected one of {initializers.SUPPORTED_STRATEGIES}.')


def log_and_write_results(results, output_dir, size):
  """Logs results to stdout and writes to output_dir."""

  loss_csv_filename = os.path.join(output_dir, 'losses.csv')

  if results.get('loss_sequence') is not None:
    logging.info('Final loss: %s', results['loss_sequence'][-1])
    df = pd.DataFrame(results['loss_sequence'], columns=['loss'])
  elif results.get('loss') is not None:
    logging.info('Final loss: %s', results['loss'])
    df = pd.DataFrame([results['loss']], columns=['loss'])
  else:
    df = pd.DataFrame()

  if results.get('n_iters') is not None:
    df['n_iters'] = [results['n_iters']]

  df.to_csv(loss_csv_filename)

  h_matrix_proto = tf.io.serialize_tensor(tf.constant(results['H']))
  w_matrix_proto = tf.io.serialize_tensor(tf.constant(results['W']))
  logging.info('H matrix: %s', results['H'])
  logging.info('W matrix: %s', results['W'])

  size_str = f'size={size:d}'
  tensor_path = os.path.join(output_dir, 'prefix_opt', size_str)
  h_proto_filename = os.path.join(tensor_path, 'h_matrix_tensor_pb')
  w_proto_filename = os.path.join(tensor_path, 'w_matrix_tensor_pb')
  tf.io.write_file(h_proto_filename, h_matrix_proto)
  tf.io.write_file(w_proto_filename, w_matrix_proto)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if not FLAGS.root_output_dir:
    raise ValueError('Must provide output directory.')

  output_dir = os.path.join(FLAGS.root_output_dir, FLAGS.experiment_name)
  ckpt_dir = os.path.join(output_dir, 'checkpoints')
  ckpt_manager = tff.program.FileProgramStateManager(ckpt_dir)
  # Notice we put all TB summaries in the same directory, separated by name.
  summary_dir = os.path.join(FLAGS.root_output_dir, 'logdir',
                             FLAGS.experiment_name)
  tb_manager = tff.program.TensorBoardReleaseManager(summary_dir)

  try:
    tf.io.gfile.makedirs(output_dir)
  except tf.errors.OpError:
    logging.info('Skipping creation of directory %s, already exists',
                 output_dir)

  dimensionality = 2**FLAGS.log_2_observations
  s_matrix = np.tril(np.ones(shape=(dimensionality, dimensionality)))

  if FLAGS.method == 'fixed_point_iteration':
    results = loops.compute_h_fixed_point_iteration(
        s_matrix=jnp.array(s_matrix),
        target_relative_duality_gap=FLAGS.target_relative_duality_gap,
        max_fixed_point_iterations=FLAGS.max_iterations,
    )

  elif FLAGS.method == 'gradient_descent':
    params = _get_parameters_for_init_strategy(FLAGS.strategy)

    initial_h = initializers.get_initial_h(FLAGS.strategy, params)

    results = loops.learn_h_sgd(
        initial_h=initial_h,
        s_matrix=tf.constant(s_matrix),
        streaming=FLAGS.streaming,
        n_iters=FLAGS.n_iters,
        optimizer=tf.keras.optimizers.SGD(FLAGS.learning_rate),
        program_state_manager=ckpt_manager,
        rounds_per_saving_program_state=FLAGS.steps_per_checkpoint,
        metrics_managers=[tb_manager],
    )
  else:
    raise ValueError(f'Unknown method for computing H: {FLAGS.method}')

  log_and_write_results(results, output_dir, size=dimensionality)


if __name__ == '__main__':
  app.run(main)
