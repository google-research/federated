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
"""Runs federated training with compression."""

import functools
import pprint
from typing import Sequence

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_federated as tff

from compressed_communication import builder
from compressed_communication import configs
from utils import task_utils
from utils import training_utils
from utils import utils_impl
from utils.optimizers import optimizer_utils

with utils_impl.record_hparam_flags() as optimizer_flags:
  optimizer_utils.define_optimizer_flags('client')
  optimizer_utils.define_optimizer_flags('server')

with utils_impl.record_hparam_flags() as shared_flags:
  # Train configuration
  flags.DEFINE_integer(
      'train_epochs', 1,
      'Number of dataset epochs for each client at each training round.')
  flags.DEFINE_integer('train_batch_size', 32, 'Batch size on train clients.')
  flags.DEFINE_integer(
      'max_train_elements', None, 'Max number of examples used by each client '
      'when training. If set to `None`, all examples are used. This should '
      'only be set to a positive value for Stack Overflow tasks.')
  flags.DEFINE_integer('clients_per_train_round', 10,
                       'How many clients to sample at each training round.')

  # Evaluation configuration
  flags.DEFINE_integer('eval_batch_size', 64, 'Batch size for evaluation.')
  flags.DEFINE_integer(
      'clients_per_eval_round', None,
      'How many clients to sample at each evaluation round. If set to a '
      'positive number, we perform a federated evaluation periodically every '
      '`FLAGS.rounds_per_evaluation` round. Otherwise, we perform centralized '
      'evaluation periodically. In either case, a centralized evaluation is '
      'also computed once training has completed.')

  # Aggregator flags
  flags.DEFINE_enum('aggregator', 'quantize_entropy_code', configs.AGGREGATORS,
                    'What aggregator to use.')
  flags.DEFINE_float('step_size', 0.5, 'Quantization step size.')
  flags.DEFINE_enum('rounding_type', 'uniform', configs.ROUNDING_TYPES,
                    'What type of quantization to apply.')
  flags.DEFINE_enum('normalization_type', 'constant',
                    configs.NORMALIZATION_TYPES, 'What normalization to apply.')
  flags.DEFINE_enum('schedule', 'fixed', configs.QUANTIZATION_SCHEDULES,
                    'How to change step_size.')
  flags.DEFINE_float(
      'schedule_hparam', 0., 'Extra parameter for defining the'
      'schedule. `0.` if using `fixed` schedule.')
  flags.DEFINE_float('min_step_size', 0.01,
                     'Minimum value to decay step_size to.')
  flags.DEFINE_float(
      'step_size_sampling_width', 1.15, 'What distribution width of step_size '
      'clients can vote on to minimize distortion + lambda * rate and update '
      'the step_size used in the next round.')
  flags.DEFINE_float('qsgd_num_steps', None, 'Numer of steps to quantize '
                     'values to using QSGD.')
  flags.DEFINE_float(
      'three_lc_sparsity_factor', None, 'Factor that controls the '
      'sparsity level when using 3LC.')
  flags.DEFINE_float('top_k_fraction_to_select', None, 'Fraction of values to '
                     'select when using TopK.')
  flags.DEFINE_enum('rotation', 'identity', configs.ROTATION_TYPES, 'What '
                    'rotation transformation to perform, if any.')
  flags.DEFINE_bool('concatenate', True, 'Whether to concatenate tensors.')
  flags.DEFINE_bool('clipping', True, 'Whether to use adaptive clipping.')
  flags.DEFINE_bool('zeroing', False, 'Whether to use adaptive zeroing.')
  flags.DEFINE_bool(
      'weighted_averaging', True, 'Whether to use example-weighted averaging '
      'of client updates (True) or use uniform averaging (False).')
  flags.DEFINE_bool('group_layers', False, 'Whether to group layers.')

  # Training loop configuration
  flags.DEFINE_string(
      'experiment_name', None, 'The name of this experiment. Will be append to '
      '--root_output_dir to separate experiment results.')
  flags.mark_flag_as_required('experiment_name')
  flags.DEFINE_string('root_output_dir', '/tmp/compressed_fl/',
                      'Root directory for writing experiment output.')
  flags.DEFINE_integer('total_rounds', 1500, 'Number of total training rounds.')
  flags.DEFINE_integer(
      'rounds_per_eval', 10,
      'How often to evaluate the model on a sample of evaluation clients.')
  flags.DEFINE_integer('rounds_per_checkpoint', 25,
                       'How often to checkpoint the global model.')

  # Random seeds for reproducibility
  flags.DEFINE_integer(
      'random_seed', 0, 'An integer random seed governing randomness in client '
      'sampling in the simulation.')

with utils_impl.record_hparam_flags() as task_flags:
  task_utils.define_task_flags()

FLAGS = flags.FLAGS


def _write_hparam_flags():
  """Creates an ordered dictionary of hyperparameter flags and writes to CSV."""
  hparam_dict = utils_impl.lookup_flag_values(shared_flags)

  # Update with optimizer flags corresponding to the chosen optimizers.
  opt_flag_dict = utils_impl.lookup_flag_values(optimizer_flags)
  opt_flag_dict = optimizer_utils.remove_unused_flags('client', opt_flag_dict)
  opt_flag_dict = optimizer_utils.remove_unused_flags('server', opt_flag_dict)
  hparam_dict.update(opt_flag_dict)

  # Update with task flags
  task_flag_dict = utils_impl.lookup_flag_values(task_flags)
  hparam_dict.update(task_flag_dict)
  training_utils.write_hparams_to_csv(hparam_dict, FLAGS.root_output_dir,
                                      FLAGS.experiment_name)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  client_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('client')
  server_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('server')

  train_client_spec = tff.simulation.baselines.ClientSpec(
      num_epochs=FLAGS.train_epochs,
      batch_size=FLAGS.train_batch_size,
      max_elements=FLAGS.max_train_elements)
  eval_client_spec = tff.simulation.baselines.ClientSpec(
      num_epochs=1, batch_size=FLAGS.eval_batch_size)
  task = task_utils.create_task_from_flags(train_client_spec, eval_client_spec)

  if FLAGS.qsgd_num_steps and FLAGS.aggregator != 'qsgd':
    raise ValueError('Expected `qsgd_num_steps` to be None for `aggregator`='
                     f'{FLAGS.aggregator}, found {FLAGS.qsgd_num_steps}.')
  if FLAGS.three_lc_sparsity_factor and FLAGS.aggregator != 'three_lc':
    raise ValueError('Expected `three_lc_sparsity_factor` to be None for '
                     f'`aggregator`={FLAGS.aggregator}, found '
                     f'{FLAGS.three_lc_sparsity_factor}.')
  if FLAGS.top_k_fraction_to_select and FLAGS.aggregator != 'top_k':
    raise ValueError('Expected `top_k_fraction_to_select` to be None for '
                     f'`aggregator`={FLAGS.aggregator}, found '
                     f'{FLAGS.top_k_fraction_to_select}.')

  if FLAGS.aggregator == 'no_compression':
    aggregation_factory = builder.build_no_compression_aggregator(
        rotation=FLAGS.rotation,
        concatenate=FLAGS.concatenate,
        zeroing=FLAGS.zeroing,
        clipping=FLAGS.clipping,
        weighted=FLAGS.weighted_averaging)
  elif FLAGS.aggregator == 'quantize_entropy_code':
    aggregation_factory = builder.build_quantization_encode_aggregator(
        step_size=FLAGS.step_size,
        rounding_type=FLAGS.rounding_type,
        normalization_type=FLAGS.normalization_type,
        step_size_sched=FLAGS.schedule,
        step_size_sched_hparam=FLAGS.schedule_hparam,
        min_step_size=FLAGS.min_step_size,
        rotation=FLAGS.rotation,
        concatenate=FLAGS.concatenate,
        zeroing=FLAGS.zeroing,
        clipping=FLAGS.clipping,
        weighted=FLAGS.weighted_averaging)
  elif FLAGS.aggregator == 'vote_step_size':
    aggregation_factory = builder.build_vote_step_size_aggregator(
        step_size=FLAGS.step_size,
        rounding_type=FLAGS.rounding_type,
        sampling_width=FLAGS.step_size_sampling_width,
        rotation=FLAGS.rotation,
        concatenate=FLAGS.concatenate,
        zeroing=FLAGS.zeroing,
        clipping=FLAGS.clipping,
        weighted=FLAGS.weighted_averaging)
  elif FLAGS.aggregator == 'entropy_cross_entropy':
    aggregation_factory = builder.build_entropy_cross_entropy_aggregator(
        step_size=FLAGS.step_size,
        rounding_type=FLAGS.rounding_type,
        rotation=FLAGS.rotation,
        concatenate=FLAGS.concatenate,
        zeroing=FLAGS.zeroing,
        clipping=FLAGS.clipping,
        weighted=FLAGS.weighted_averaging,
        group_layers=FLAGS.group_layers,
        task=FLAGS.task)
  elif FLAGS.aggregator == 'histogram':
    aggregation_factory = builder.build_histogram_aggregator(
        rotation=FLAGS.rotation,
        concatenate=FLAGS.concatenate,
        zeroing=FLAGS.zeroing,
        clipping=FLAGS.clipping,
        weighted=FLAGS.weighted_averaging)
  elif FLAGS.aggregator == 'rotation_ablation':
    aggregation_factory = builder.build_rotation_ablation_aggregator(
        step_size=FLAGS.step_size,
        rounding_type=FLAGS.rounding_type,
        rotation=FLAGS.rotation,
        concatenate=FLAGS.concatenate,
        zeroing=FLAGS.zeroing,
        clipping=FLAGS.clipping,
        weighted=FLAGS.weighted_averaging)
  elif FLAGS.aggregator == 'drive':
    aggregation_factory = builder.build_drive_aggregator(
        rotation=FLAGS.rotation,
        concatenate=FLAGS.concatenate,
        zeroing=FLAGS.zeroing,
        clipping=FLAGS.clipping,
        weighted=FLAGS.weighted_averaging)
  elif FLAGS.aggregator == 'one_bit_sgd':
    aggregation_factory = builder.build_one_bit_sgd_aggregator(
        rotation=FLAGS.rotation,
        concatenate=FLAGS.concatenate,
        zeroing=FLAGS.zeroing,
        clipping=FLAGS.clipping,
        weighted=FLAGS.weighted_averaging)
  elif FLAGS.aggregator == 'qsgd':
    if FLAGS.qsgd_num_steps is None:
      raise ValueError('`qsgd_num_steps` must be defined for `qsgd` '
                       'aggregator.')
    aggregation_factory = builder.build_qsgd_aggregator(
        num_steps=FLAGS.qsgd_num_steps,
        rotation=FLAGS.rotation,
        concatenate=FLAGS.concatenate,
        zeroing=FLAGS.zeroing,
        clipping=FLAGS.clipping,
        weighted=FLAGS.weighted_averaging)
  elif FLAGS.aggregator == 'terngrad':
    aggregation_factory = builder.build_terngrad_aggregator(
        rotation=FLAGS.rotation,
        concatenate=FLAGS.concatenate,
        zeroing=FLAGS.zeroing,
        clipping=FLAGS.clipping,
        weighted=FLAGS.weighted_averaging)
  elif FLAGS.aggregator == 'three_lc':
    if FLAGS.three_lc_sparsity_factor is None:
      raise ValueError('`three_lc_sparsity_factor` must be defined for '
                       '`three_lc` aggregator.')
    aggregation_factory = builder.build_three_lc_aggregator(
        sparsity_factor=FLAGS.three_lc_sparsity_factor,
        rotation=FLAGS.rotation,
        concatenate=FLAGS.concatenate,
        zeroing=FLAGS.zeroing,
        clipping=FLAGS.clipping,
        weighted=FLAGS.weighted_averaging)
  elif FLAGS.aggregator == 'top_k':
    if FLAGS.top_k_fraction_to_select is None:
      raise ValueError('`top_k_fraction_to_select` must be defined for '
                       '`top_k` aggregator.')
    aggregation_factory = builder.build_top_k_aggregator(
        fraction_to_select=FLAGS.top_k_fraction_to_select,
        rotation=FLAGS.rotation,
        concatenate=FLAGS.concatenate,
        zeroing=FLAGS.zeroing,
        clipping=FLAGS.clipping,
        weighted=FLAGS.weighted_averaging)
  else:
    raise ValueError(f'Provided value for `aggregator`, {FLAGS.aggregator}, '
                     'is not supported.')

  if FLAGS.weighted_averaging:
    client_weighting = tff.learning.ClientWeighting.NUM_EXAMPLES
  else:
    client_weighting = tff.learning.ClientWeighting.UNIFORM

  iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
      model_fn=task.model_fn,
      client_optimizer_fn=client_optimizer_fn,
      server_optimizer_fn=server_optimizer_fn,
      client_weighting=client_weighting,
      model_aggregator=aggregation_factory)
  train_data = task.datasets.train_data.preprocess(
      task.datasets.train_preprocess_fn)
  training_process = (
      tff.simulation.compose_dataset_computation_with_iterative_process(
          train_data.dataset_computation, iterative_process))

  training_selection_fn = functools.partial(
      tff.simulation.build_uniform_sampling_fn(
          train_data.client_ids, random_seed=FLAGS.random_seed),
      size=FLAGS.clients_per_train_round)

  federated_eval = tff.learning.build_federated_evaluation(task.model_fn)

  # Get evaluation data
  raw_eval_data = task.datasets.test_data
  eval_preprocess_fn = task.datasets.eval_preprocess_fn
  central_raw_eval_data = raw_eval_data.create_tf_dataset_from_all_clients()
  central_eval_data = eval_preprocess_fn(central_raw_eval_data)

  if FLAGS.clients_per_eval_round is not None:
    if FLAGS.clients_per_eval_round <= 0:
      raise ValueError('The clients_per_eval_round flag must be `None` or a '
                       'positive integer, found {}.'.format(
                           FLAGS.clients_per_eval_round))

    @tff.tf_computation(tf.string)
    def build_eval_dataset_from_client_id(client_id):
      raw_client_data = raw_eval_data.dataset_computation(client_id)
      return eval_preprocess_fn(raw_client_data)

    composed_eval_fn = tff.simulation.compose_dataset_computation_with_computation(
        build_eval_dataset_from_client_id, federated_eval)

    evaluation_selection_fn = functools.partial(
        tff.simulation.build_uniform_sampling_fn(
            sample_range=raw_eval_data.client_ids,
            replace=False,
            random_seed=FLAGS.random_seed),
        size=FLAGS.clients_per_eval_round)

    def evaluation_fn(state, evaluation_data):
      return composed_eval_fn(state.model, evaluation_data)

  else:
    central_eval_data = central_eval_data.cache()
    evaluation_selection_fn = lambda _: [central_eval_data]

    def evaluation_fn(state, evaluation_data):
      return federated_eval(state.model, evaluation_data)

  program_state_manager, metrics_managers = training_utils.create_managers(
      FLAGS.root_output_dir, FLAGS.experiment_name)
  _write_hparam_flags()
  state = tff.simulation.run_training_process(
      training_process=training_process,
      training_selection_fn=training_selection_fn,
      total_rounds=FLAGS.total_rounds,
      evaluation_fn=evaluation_fn,
      evaluation_selection_fn=evaluation_selection_fn,
      rounds_per_evaluation=FLAGS.rounds_per_eval,
      program_state_manager=program_state_manager,
      rounds_per_saving_program_state=FLAGS.rounds_per_checkpoint,
      metrics_managers=metrics_managers)

  test_metrics = federated_eval(state.model, [central_eval_data])
  logging.info('Test metrics:\n %s', pprint.pformat(test_metrics))
  for metrics_manager in metrics_managers:
    metrics_manager.release(test_metrics, FLAGS.total_rounds + 1)


if __name__ == '__main__':
  app.run(main)
