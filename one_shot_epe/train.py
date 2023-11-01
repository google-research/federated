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
"""Trains a federated model in simulation with canary clients."""

import asyncio
import collections
import os.path
import time

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_federated as tff

from one_shot_epe import canary_insertion
from one_shot_epe import fed_avg_with_canaries
from one_shot_epe import train_lib
from utils import task_utils
from utils.optimizers import optimizer_utils


IRRELEVANT_FLAGS = frozenset(iter(flags.FLAGS))

# Optimizer
optimizer_utils.define_optimizer_flags('client')
optimizer_utils.define_optimizer_flags('server')

# Training
_CLIENT_EPOCHS_PER_ROUND = flags.DEFINE_integer(
    'client_epochs_per_round',
    1,
    'Number of epochs in the client to take per round.',
)
_CLIENT_BATCH_SIZE = flags.DEFINE_integer(
    'client_batch_size', 16, 'Batch size on the clients.'
)
_SAMPLING = flags.DEFINE_enum(
    'sampling', 'shuffle', ['bernoulli', 'shuffle'], 'Sampling method.'
)
_CLIENTS_PER_ROUND = flags.DEFINE_integer(
    'clients_per_round',
    100,
    'How many clients to sample per round if using Berboulli sampling.',
)
_CLIENT_DATASETS_RANDOM_SEED = flags.DEFINE_integer(
    'client_datasets_random_seed', None, 'Random seed for client sampling.'
)
_MAX_EXAMPLES_PER_CLIENT = flags.DEFINE_integer(
    'max_examples_per_client',
    None,
    (
        'Maximum number of examples for each training client. If set to None, '
        'all available examples are used.'
    ),
)
_EVAL_BATCH_SIZE = flags.DEFINE_integer(
    'eval_batch_size', 100, 'Batch size for evaluation.'
)
_TRAIN_EPOCHS = flags.DEFINE_integer(
    'train_epochs', 5, 'Number of training epochs.'
)
_TOTAL_ROUNDS = flags.DEFINE_integer(
    'total_rounds', 500, 'Number of total training rounds.'
)
_ROUNDS_PER_EVALUATION = flags.DEFINE_integer(
    'rounds_per_evaluation',
    50,
    'How often to evaluate the global model on the validation dataset.',
)
_ROUNDS_PER_CHECKPOINT = flags.DEFINE_integer(
    'rounds_per_checkpoint', 50, 'How often to checkpoint the global model.'
)
_NUM_VALIDATION_EXAMPLES = flags.DEFINE_integer(
    'num_validation_examples', 10000, 'The number of validationexamples to use.'
)
_NUM_TEST_EXAMPLES = flags.DEFINE_integer(
    'num_test_examples',
    -1,
    (
        'The number of test'
        'examples to use. If set to -1, all available examples are used.'
    ),
)
_CANARY_REPEATS = flags.DEFINE_integer(
    'canary_repeats',
    1,
    'Number of times to repeat each canary if using shuffling.',
)

# Output
_ROOT_OUTPUT_DIR = flags.DEFINE_string(
    'root_output_dir', None, 'Root directory for writing experiment output.'
)
_RUN_NAME = flags.DEFINE_string(
    'run_name',
    None,
    (
        'The name of this run. Will be append to '
        '--root_output_dir to separate results of different runs.'
    ),
)

# Differential privacy
_NOISE_MULTIPLIER = flags.DEFINE_float(
    'noise_multiplier', 0.0, 'Noise multiplier.'
)
_TARGET_UNCLIPPED_QUANTILE = flags.DEFINE_float(
    'target_unclipped_quantile', 0.5, 'Target unclipped quantile.'
)
_CLIPPING_NORM = flags.DEFINE_float(
    'clipping_norm', 0, 'Clipping norm. If 0, use adaptive clipping.'
)

# Canaries
_NUM_CANARIES = flags.DEFINE_integer(
    'num_canaries', 1000, 'Number of canary clients.'
)
_NUM_UNSEEN_CANARIES = flags.DEFINE_integer(
    'num_unseen_canaries', 1000, 'Number of unseen canary clients.'
)
_CANARY_SEED = flags.DEFINE_integer(
    'canary_seed', None, 'Seed to generate canary updates.'
)

# Task
task_utils.define_task_flags()

_USE_EXPERIMENTAL_SIMULATION_LOOP = flags.DEFINE_bool(
    'use_experimental_simulation_loop',
    None,
    'Whether to use experimental simulation loop.',
)

# Debug
_USE_SYNTHETIC_DATA = flags.DEFINE_bool(
    'use_synthetic_data',
    False,
    (
        'Whether to use synthetic data. This should '
        'only be set to True for debugging purposes.'
    ),
)

HPARAM_FLAGS = [f for f in flags.FLAGS if f not in IRRELEVANT_FLAGS]


def _train() -> None:
  """Trains federated model with canaries."""
  logging.info('Show FLAGS for debugging:')
  for f in HPARAM_FLAGS:
    logging.info('%s=%s', f, flags.FLAGS[f].value)

  hparam_dict = collections.OrderedDict(
      [(name, flags.FLAGS[name].value) for name in HPARAM_FLAGS]
  )

  client_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('client')
  server_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('server')
  train_client_spec = tff.simulation.baselines.ClientSpec(
      num_epochs=_CLIENT_EPOCHS_PER_ROUND.value,
      batch_size=_CLIENT_BATCH_SIZE.value,
      max_elements=_MAX_EXAMPLES_PER_CLIENT.value,
  )
  eval_client_spec = tff.simulation.baselines.ClientSpec(
      num_epochs=1, batch_size=_EVAL_BATCH_SIZE.value
  )
  task = task_utils.create_task_from_flags(
      train_client_spec=train_client_spec,
      eval_client_spec=eval_client_spec,
      use_synthetic_data=_USE_SYNTHETIC_DATA.value,
  )

  logging.info('Trainable weights:')
  trainable_model_size = 0
  for weight in task.model_fn().trainable_variables:
    logging.info('name: %s  shape: %s', weight.name, weight.shape)
    trainable_model_size += weight.shape.num_elements()
  logging.info('Total trainable model size: %s', trainable_model_size)

  client_datasets_random_seed = _CLIENT_DATASETS_RANDOM_SEED.value
  if client_datasets_random_seed is None:
    logging.warn('client_datasets_random_seed not specified. Using fixed seed.')
    client_datasets_random_seed = 0

  train_data = canary_insertion.add_canaries(
      task.datasets.train_data,
      _NUM_CANARIES.value,
  )

  @tff.tf_computation(tf.string)
  def build_train_dataset_from_client_id(client_id):
    raw_client_data = train_data.dataset_computation(client_id)
    return task.datasets.train_preprocess_fn(raw_client_data)

  if _SAMPLING.value == 'bernoulli':
    training_selection_fn = train_lib.build_bernoulli_sampling_fn(
        train_data.client_ids,
        _CLIENTS_PER_ROUND.value,
        client_datasets_random_seed,
    )
    mean_clients_per_round = _CLIENTS_PER_ROUND.value
  else:  # _SAMPLING.value == 'shuffle'
    if _CANARY_REPEATS.value is None:
      raise ValueError('`canary_repeats` must be specified for shuffling.')
    real_client_ids = [
        id for id in train_data.client_ids if id.startswith('real:')
    ]
    canary_client_ids = [
        id for id in train_data.client_ids if id.startswith('canary:')
    ]
    training_selection_fn, mean_clients_per_round = (
        train_lib.build_shuffling_sampling_fn(
            real_client_ids,
            canary_client_ids,
            _TOTAL_ROUNDS.value,
            _TRAIN_EPOCHS.value,
            _CANARY_REPEATS.value,
            client_datasets_random_seed,
        )
    )
    logging.info('Mean clients per round: %s', mean_clients_per_round)

  federated_eval = tff.learning.build_federated_evaluation(
      task.model_fn,
      use_experimental_simulation_loop=_USE_EXPERIMENTAL_SIMULATION_LOOP.value,
  )

  def evaluation_fn(state, evaluation_data):
    return federated_eval(state.global_model_weights, evaluation_data)

  all_test_data = task.datasets.test_data.create_tf_dataset_from_all_clients(
      seed=0
  )
  validation_data = task.datasets.eval_preprocess_fn(
      all_test_data.take(_NUM_VALIDATION_EXAMPLES.value)
  )
  evaluation_selection_fn = lambda round_num: [validation_data]

  if _CLIPPING_NORM.value != 0:
    aggregation_factory = (
        tff.aggregators.DifferentiallyPrivateFactory.gaussian_fixed(
            noise_multiplier=_NOISE_MULTIPLIER.value,
            clients_per_round=mean_clients_per_round,
            clip=_CLIPPING_NORM.value,
        )
    )
  else:
    aggregation_factory = (
        tff.aggregators.DifferentiallyPrivateFactory.gaussian_adaptive(
            noise_multiplier=_NOISE_MULTIPLIER.value,
            clients_per_round=mean_clients_per_round,
            target_unclipped_quantile=_TARGET_UNCLIPPED_QUANTILE.value,
        )
    )

  if _CANARY_SEED.value is None:
    canary_seed = time.time_ns()
  else:
    canary_seed = _CANARY_SEED.value

  num_unseen_canaries_for_training = _NUM_UNSEEN_CANARIES.value

  training_process = fed_avg_with_canaries.build_canary_learning_process(
      model_fn=task.model_fn,
      dataset_computation=build_train_dataset_from_client_id,
      canary_seed=canary_seed,
      num_canaries=_NUM_CANARIES.value,
      num_unseen_canaries=num_unseen_canaries_for_training,
      update_aggregator_factory=aggregation_factory,
      client_optimizer_fn=client_optimizer_fn,
      server_optimizer_fn=server_optimizer_fn,
      use_experimental_simulation_loop=_USE_EXPERIMENTAL_SIMULATION_LOOP.value,
  )
  program_state_manager, metrics_managers = train_lib.create_managers(
      _ROOT_OUTPUT_DIR.value, _RUN_NAME.value, hparam_dict
  )
  final_state = tff.simulation.run_training_process(
      training_process=training_process,
      training_selection_fn=training_selection_fn,
      total_rounds=_TOTAL_ROUNDS.value,
      evaluation_fn=evaluation_fn,
      evaluation_selection_fn=evaluation_selection_fn,
      rounds_per_evaluation=_ROUNDS_PER_EVALUATION.value,
      program_state_manager=program_state_manager,
      rounds_per_saving_program_state=_ROUNDS_PER_CHECKPOINT.value,
      metrics_managers=metrics_managers,
  )

  loop = asyncio.get_event_loop()
  if _NUM_CANARIES.value > 0 or _NUM_UNSEEN_CANARIES.value > 0:
    final_model_cosine_path = os.path.join(
        _ROOT_OUTPUT_DIR.value,
        'cosines',
        _RUN_NAME.value,
        'final_model_cosines.csv',
    )
    logging.info(
        'Writing final model canary cosines to: %s', final_model_cosine_path
    )
    release_manager = tff.program.CSVFileReleaseManager(final_model_cosine_path)
    final_model_weights = training_process.get_model_weights(final_state)
    loop.run_until_complete(
        train_lib.compute_and_release_final_model_canary_metrics(
            release_manager,
            final_model_weights,
            final_state.client_work['canary_seed'],
            final_state.max_canary_model_delta_cosines,
            final_state.max_unseen_canary_model_delta_cosines,
        )
    )

  if _NUM_TEST_EXAMPLES.value > 0 or _NUM_TEST_EXAMPLES.value == -1:
    logging.info('Evaluating on test set.')
    test_data = [
        task.datasets.eval_preprocess_fn(
            all_test_data.skip(_NUM_VALIDATION_EXAMPLES.value).take(
                _NUM_TEST_EXAMPLES.value
            )
        )
    ]
    final_model_test_path = os.path.join(
        _ROOT_OUTPUT_DIR.value,
        'results',
        _RUN_NAME.value,
        'final_test_metrics.csv',
    )
    logging.info('Writing final test metrics to: %s', final_model_test_path)
    release_manager = tff.program.CSVFileReleaseManager(final_model_test_path)
    test_evaluation_time_start = time.time()
    test_metrics = evaluation_fn(final_state, test_data)
    test_metrics['test_secs'] = time.time() - test_evaluation_time_start
    metrics_type = tff.types.type_conversions.infer_type(test_metrics)
    loop.run_until_complete(
        release_manager.release(test_metrics, metrics_type, 0)
    )

  loop.run_until_complete(program_state_manager.remove_all())


def main(argv: list[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  _train()


if __name__ == '__main__':
  app.run(main)
