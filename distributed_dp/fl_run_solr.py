# Copyright 2021, Google LLC. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runs federated training with differential privacy on StackOverflow LR."""

import functools
import math
import pprint

from absl import app
from absl import flags
from absl import logging
import tensorflow_federated as tff

from distributed_dp import fl_utils
from utils import training_utils
from utils import utils_impl
from utils.optimizers import optimizer_utils

with utils_impl.record_hparam_flags() as optimizer_flags:
  # Defining optimizer flags
  optimizer_utils.define_optimizer_flags('client')
  optimizer_utils.define_optimizer_flags('server')

with utils_impl.record_hparam_flags() as shared_flags:
  # Federated training hyperparameters
  flags.DEFINE_string('model', None, '(Placeholder flag)')
  flags.DEFINE_integer('client_epochs_per_round', 1,
                       'Number of epochs in the client to take per round.')
  flags.DEFINE_integer('client_batch_size', 100, 'Batch size on the clients.')
  flags.DEFINE_integer('clients_per_round', 60,
                       'How many clients to sample per round.')
  flags.DEFINE_integer('client_datasets_random_seed', 1,
                       'Random seed for client sampling.')
  flags.DEFINE_boolean(
      'uniform_weighting', True,
      'Whether to weigh clients uniformly. If false, clients '
      'are weighted by the number of samples.')

  # Training loop configuration
  flags.DEFINE_integer('total_rounds', 1500, 'Number of total training rounds.')
  flags.DEFINE_string(
      'experiment_name', None, 'The name of this experiment. Will be append to '
      '--root_output_dir to separate experiment results.')
  flags.DEFINE_string('root_output_dir', '/tmp/ddg_so_lr/',
                      'Root directory for writing experiment output.')
  flags.DEFINE_integer(
      'rounds_per_eval', 1,
      'How often to evaluate the global model on the validation dataset.')
  flags.DEFINE_integer('rounds_per_checkpoint', 50,
                       'How often to checkpoint the global model.')

with utils_impl.record_hparam_flags() as compression_flags:
  flags.DEFINE_integer('num_bits', 16, 'Number of bits for quantization.')
  flags.DEFINE_float('beta', math.exp(-0.5), 'Beta for stochastic rounding.')
  flags.DEFINE_integer('k_stddevs', 4,
                       'Number of stddevs to bound the signal range.')

with utils_impl.record_hparam_flags() as dp_flags:
  flags.DEFINE_float(
      'epsilon', 5.0, 'Epsilon for the DP mechanism. '
      'No DP used if this is None.')
  flags.DEFINE_float('delta', None, 'Delta for the DP mechanism. ')
  flags.DEFINE_float('l2_norm_clip', 2.0, 'Initial L2 norm clip.')

  dp_mechanisms = ['gaussian', 'ddgauss']
  flags.DEFINE_enum('dp_mechanism', 'ddgauss', dp_mechanisms,
                    'Which DP mechanism to use.')

FLAGS = flags.FLAGS

NUM_TRAIN_CLIENTS = 342477

# Hard coded constants governing the behavior of the tag prediction task.
WORD_VOCAB_SIZE = 10000
TAG_VOCAB_SIZE = 500
MAX_ELEMENTS_PER_TRAIN_CLIENT = 1000
NUM_VALIDATION_EXAMPLES = 10000


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  client_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('client')
  server_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('server')

  compression_dict = utils_impl.lookup_flag_values(compression_flags)
  dp_dict = utils_impl.lookup_flag_values(dp_flags)

  train_client_spec = tff.simulation.baselines.ClientSpec(
      num_epochs=FLAGS.client_epochs_per_round,
      batch_size=FLAGS.client_batch_size,
      max_elements=MAX_ELEMENTS_PER_TRAIN_CLIENT)
  task = tff.simulation.baselines.stackoverflow.create_tag_prediction_task(
      train_client_spec,
      word_vocab_size=WORD_VOCAB_SIZE,
      tag_vocab_size=TAG_VOCAB_SIZE)

  model_trainable_variables = task.model_fn().trainable_variables

  # Most logic for deciding what to run is here.
  aggregation_factory = fl_utils.build_aggregator(
      compression_flags=compression_dict,
      dp_flags=dp_dict,
      num_clients=NUM_TRAIN_CLIENTS,
      num_clients_per_round=FLAGS.clients_per_round,
      num_rounds=FLAGS.total_rounds,
      client_template=model_trainable_variables)

  iterative_process = tff.learning.build_federated_averaging_process(
      model_fn=task.model_fn,
      server_optimizer_fn=server_optimizer_fn,
      client_weighting=tff.learning.ClientWeighting.UNIFORM,
      client_optimizer_fn=client_optimizer_fn,
      model_update_aggregation_factory=aggregation_factory)
  train_data = task.datasets.train_data.preprocess(
      task.datasets.train_preprocess_fn)
  training_process = (
      tff.simulation.compose_dataset_computation_with_iterative_process(
          train_data.dataset_computation, iterative_process))

  client_selection_fn = functools.partial(
      tff.simulation.build_uniform_sampling_fn(
          train_data.client_ids, random_seed=FLAGS.client_datasets_random_seed),
      size=FLAGS.clients_per_round)
  validation_fn = training_utils.create_validation_fn(
      task,
      validation_frequency=FLAGS.rounds_per_eval,
      num_validation_examples=NUM_VALIDATION_EXAMPLES)

  def validation_fn_from_state(state, round_num):
    return validation_fn(state.model, round_num)

  checkpoint_manager, metrics_managers = training_utils.configure_managers(
      FLAGS.root_output_dir,
      FLAGS.experiment_name,
      rounds_per_checkpoint=FLAGS.rounds_per_checkpoint)

  state = tff.simulation.run_simulation(
      process=training_process,
      client_selection_fn=client_selection_fn,
      total_rounds=FLAGS.total_rounds,
      validation_fn=validation_fn_from_state,
      file_checkpoint_manager=checkpoint_manager,
      metrics_managers=metrics_managers)

  test_fn = training_utils.create_test_fn(task)
  test_metrics = test_fn(state.model)
  logging.info('Test metrics:\n %s', pprint.pformat(test_metrics))
  for metrics_manager in metrics_managers:
    metrics_manager.save_metrics(test_metrics, FLAGS.total_rounds + 1)


if __name__ == '__main__':
  app.run(main)
