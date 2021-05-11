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
"""Trains and evaluates an EMNIST classification model."""

import math
from typing import Callable

from absl import app
from absl import flags
import tensorflow_federated as tff

from distributed_discrete_gaussian import fl_utils
from optimization.emnist import federated_emnist
from optimization.shared import training_specs
from utils import training_loop
from utils import utils_impl

with utils_impl.record_hparam_flags():
  # Experiment hyperparameters
  flags.DEFINE_enum('model', '1m_cnn', ['cnn', '2nn', '1m_cnn'],
                    'Which model to use.')
  flags.DEFINE_integer('client_batch_size', 20, 'Batch size on the clients.')
  flags.DEFINE_integer('clients_per_round', 100, 'Number of clients per round.')
  flags.DEFINE_integer(
      'client_epochs_per_round', 1,
      'Number of client (inner optimizer) epochs per federated round.')
  flags.DEFINE_boolean(
      'uniform_weighting', True,
      'Whether to weigh clients uniformly. If false, clients '
      'are weighted by the number of samples.')
  flags.DEFINE_boolean('only_digits', False,
                       'Whether to use only digits for the EMNIST dataset.')
  flags.DEFINE_integer('client_datasets_random_seed', 42,
                       'Random seed for client sampling.')

  # Optimizer configuration (this defines one or more flags per optimizer).
  utils_impl.define_optimizer_flags('server')
  utils_impl.define_optimizer_flags('client')

with utils_impl.record_new_flags() as training_loop_flags:
  flags.DEFINE_integer('total_rounds', 1500, 'Number of total training rounds.')
  flags.DEFINE_string(
      'experiment_name', None, 'The name of this experiment. Will be append to '
      '--root_output_dir to separate experiment results.')
  flags.DEFINE_string('root_output_dir', '/tmp/ddg_emnist/',
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
      'epsilon', 10.0, 'Epsilon for the DP mechanism. '
      'No DP used if this is None.')
  flags.DEFINE_float('delta', None, 'Delta for the DP mechanism. ')
  flags.DEFINE_float('l2_norm_clip', 0.03, 'Initial L2 norm clip.')

  dp_mechanisms = ['gaussian', 'ddgauss']
  flags.DEFINE_enum('dp_mechanism', 'ddgauss', dp_mechanisms,
                    'Which DP mechanism to use.')

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  client_optimizer_fn = lambda: utils_impl.create_optimizer_from_flags('client')
  server_optimizer_fn = lambda: utils_impl.create_optimizer_from_flags('server')

  compression_dict = utils_impl.lookup_flag_values(compression_flags)
  dp_dict = utils_impl.lookup_flag_values(dp_flags)

  def iterative_process_builder(
      model_fn: Callable[[], tff.learning.Model],
  ) -> tff.templates.IterativeProcess:
    """Creates an iterative process using a given TFF `model_fn`."""

    model_trainable_variables = model_fn().trainable_variables

    # Most logic for deciding what to run is here.
    aggregation_factory = fl_utils.build_aggregator(
        compression_flags=compression_dict,
        dp_flags=dp_dict,
        num_clients=3400,
        num_clients_per_round=FLAGS.clients_per_round,
        num_rounds=FLAGS.total_rounds,
        client_template=model_trainable_variables)

    return tff.learning.build_federated_averaging_process(
        model_fn=model_fn,
        server_optimizer_fn=server_optimizer_fn,
        client_weighting=tff.learning.ClientWeighting.UNIFORM,
        client_optimizer_fn=client_optimizer_fn,
        model_update_aggregation_factory=aggregation_factory)

  task_spec = training_specs.TaskSpec(
      iterative_process_builder=iterative_process_builder,
      client_epochs_per_round=FLAGS.client_epochs_per_round,
      client_batch_size=FLAGS.client_batch_size,
      clients_per_round=FLAGS.clients_per_round,
      client_datasets_random_seed=FLAGS.client_datasets_random_seed)

  runner_spec = federated_emnist.configure_training(task_spec, FLAGS.model)

  training_loop.run(
      iterative_process=runner_spec.iterative_process,
      client_datasets_fn=runner_spec.client_datasets_fn,
      validation_fn=runner_spec.validation_fn,
      test_fn=runner_spec.test_fn,
      total_rounds=FLAGS.total_rounds,
      experiment_name=FLAGS.experiment_name,
      root_output_dir=FLAGS.root_output_dir,
      rounds_per_eval=FLAGS.rounds_per_eval,
      rounds_per_checkpoint=FLAGS.rounds_per_checkpoint)


if __name__ == '__main__':
  app.run(main)
