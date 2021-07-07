# Copyright 2020, Google LLC.
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
"""Runs federated training on various tasks using FedPA.

Specifically, we create (according to flags) an iterative processes that allows
for client and server learning rate schedules, as well as various client and
server optimization methods. For more details on the learning rate scheduling
and optimization methods, see `optimization/shared/optimizer_utils.py`.
For details on the iterative process, see
`posterior_inference/shared/fed_pa_schedule.py`.
"""

import pprint

from absl import app
from absl import flags
from absl import logging
import tensorflow_federated as tff

from posterior_averaging import fed_pa_schedule
from utils import task_utils
from utils import training_utils
from utils import utils_impl
from utils.optimizers import optimizer_utils

with utils_impl.record_hparam_flags() as optimizer_flags:
  # Defining optimizer flags
  optimizer_utils.define_optimizer_flags('client')
  optimizer_utils.define_optimizer_flags('server')
  optimizer_utils.define_lr_schedule_flags('client')
  optimizer_utils.define_lr_schedule_flags('server')

with utils_impl.record_hparam_flags() as shared_flags:
  # Federated training hyperparameters
  flags.DEFINE_integer('client_epochs_per_round', 1,
                       'Number of epochs in the client to take per round.')
  flags.DEFINE_integer('client_batch_size', 20, 'Batch size on the clients.')
  flags.DEFINE_integer('clients_per_round', 10,
                       'How many clients to sample per round.')
  flags.DEFINE_integer('client_datasets_random_seed', 1,
                       'Random seed for client sampling.')
  flags.DEFINE_integer(
      'max_elements_per_client', None, 'Maximum number of '
      'elements for each training client. If set to None, all '
      'available examples are used.')

  # Training loop configuration
  flags.DEFINE_string(
      'experiment_name', None, 'The name of this experiment. Will be append to '
      '--root_output_dir to separate experiment results.')
  flags.mark_flag_as_required('experiment_name')
  flags.DEFINE_string('root_output_dir', '/tmp/fed_pa/',
                      'Root directory for writing experiment output.')
  flags.DEFINE_integer('total_rounds', 200, 'Number of total training rounds.')
  flags.DEFINE_integer(
      'rounds_per_eval', 1,
      'How often to evaluate the global model on the validation dataset.')
  flags.DEFINE_integer(
      'num_validation_examples', -1, 'The number of validation'
      'examples to use. If set to -1, all available examples '
      'are used.')
  flags.DEFINE_integer('rounds_per_checkpoint', 50,
                       'How often to checkpoint the global model.')

with utils_impl.record_hparam_flags() as fed_pa_flags:
  flags.DEFINE_integer(
      'client_mixin_epochs_per_round', 1,
      'The number of client epochs per federated round used for MCMC mixing.')
  flags.DEFINE_enum(
      'client_mixin_check_scheme', 'fixed_epochs', ['fixed_epochs'],
      'The name of the scheme used for checking whether MCMC has mixed-in:\n'
      '- fixed_epochs: assumes chains mix-in after a fixed number of epochs.')
  flags.DEFINE_integer(
      'client_mixin_check_start_round', 100,
      'The round number starting which we start checking for MCMC mixin. '
      'During all rounds before that we assume that clients are not mixed-in.')
  flags.DEFINE_enum('client_update_delta_scheme', 'posterior_avg',
                    ['simple_avg', 'posterior_avg'],
                    'The name of the scheme used to update weight deltas.')
  flags.DEFINE_float(
      'client_shrinkage_rho', 0.1,
      'The hyperparameter of the shrinkage estimator of the posterior '
      'covariance matrix.')
  flags.DEFINE_boolean(
      'mask_zeros_in_client_updates', False,
      'Indicates whether to average client deltas with zero masking.')

with utils_impl.record_hparam_flags() as task_flags:
  task_utils.define_task_flags()

FLAGS = flags.FLAGS


def _write_hparam_flags():
  """Creates an ordered dictionary of hyperparameter flags and writes to CSV."""
  hparam_dict = utils_impl.lookup_flag_values(shared_flags)
  hparam_dict.update(utils_impl.lookup_flag_values(fed_pa_flags))

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


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  client_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('client')
  server_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('server')
  client_lr_schedule = optimizer_utils.create_lr_schedule_from_flags('client')
  server_lr_schedule = optimizer_utils.create_lr_schedule_from_flags('server')

  client_mixedin_schedule_fn = fed_pa_schedule.create_mixin_check_fn(
      name=FLAGS.client_mixin_check_scheme,
      num_mixin_epochs=FLAGS.client_mixin_epochs_per_round,
      start_round=FLAGS.client_mixin_check_start_round)
  client_update_delta_fn = fed_pa_schedule.create_update_delta_fn(
      name=FLAGS.client_update_delta_scheme, rho=FLAGS.client_shrinkage_rho)

  train_client_spec = tff.simulation.baselines.ClientSpec(
      num_epochs=1,
      batch_size=FLAGS.client_batch_size,
      max_elements=FLAGS.max_elements_per_client)
  task = task_utils.create_task_from_flags(train_client_spec)

  process = fed_pa_schedule.build_fed_pa_process(
      model_fn=task.model_fn,
      client_update_epochs=FLAGS.client_epochs_per_round,
      client_optimizer_fn=client_optimizer_fn,
      client_lr=client_lr_schedule,
      server_optimizer_fn=server_optimizer_fn,
      server_lr=server_lr_schedule,
      client_mixedin_schedule_fn=client_mixedin_schedule_fn,
      client_update_delta_fn=client_update_delta_fn,
      mask_zeros_in_client_updates=FLAGS.mask_zeros_in_client_updates)
  client_selection_fn = training_utils.create_client_selection_fn(
      task,
      clients_per_round=FLAGS.clients_per_round,
      random_seed=FLAGS.client_datasets_random_seed)
  validation_fn = training_utils.create_validation_fn(
      task,
      validation_frequency=FLAGS.rounds_per_eval,
      num_validation_examples=FLAGS.num_validation_examples)

  def validation_fn_from_state(state, round_num):
    return validation_fn(state.model, round_num)

  checkpoint_manager, metrics_managers = training_utils.configure_managers(
      FLAGS.root_output_dir,
      FLAGS.experiment_name,
      rounds_per_checkpoint=FLAGS.rounds_per_checkpoint)
  _write_hparam_flags()
  state = tff.simulation.run_simulation(
      process=process,
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
