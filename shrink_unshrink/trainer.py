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
"""Runs federated training on various tasks using a generalized form of FedAvg.

Specifically, we create (according to flags) an iterative processes that allows
for client and server learning rate schedules, as well as various client and
server optimization methods. For more details on the learning rate scheduling
and optimization methods, see `shared/optimizer_utils.py`. For details on the
iterative process, see `shared/fed_avg_schedule.py`.
"""
import functools
import pprint

from absl import app
from absl import flags
from absl import logging
import tensorflow_federated as tff

from shrink_unshrink import models
from shrink_unshrink import shrink_unshrink_tff
from shrink_unshrink import simple_fedavg_tf
from shrink_unshrink import simple_fedavg_tff
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
  flags.DEFINE_string('root_output_dir', '/tmp/fed_opt2/',
                      'Root directory for writing experiment output.')
  flags.DEFINE_integer('total_rounds', 200, 'Number of total training rounds.')
  flags.DEFINE_integer(
      'rounds_per_eval', 100,
      'How often to evaluate the global model on the validation dataset.')
  flags.DEFINE_integer(
      'num_validation_examples', 4, 'The number of validation'
      'examples to use. If set to -1, all available examples '
      'are used.')
  flags.DEFINE_integer('rounds_per_checkpoint', 50,
                       'How often to checkpoint the global model.')
  flags.DEFINE_integer('small_embedding_size', 72,
                       'The embedding size of the lstm.')
  flags.DEFINE_integer('small_lstm_size', 503, 'The size of the lstm layer.')

  flags.DEFINE_enum(
      name='shrink_unshrink_type',
      default='identity',
      enum_values=['identity', 'layerwise', 'client_layerwise'],
      help='what type of shrink_unshrink to do')
  flags.DEFINE_enum(
      name='build_projection_matrix_type',
      default='normal',
      enum_values=['normal', 'dropout', 'qr'],
      help='what type of shrink_unshrink to do')

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


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  logging.info('beginning main')
  client_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('client')
  server_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('server')

  train_client_spec = tff.simulation.baselines.ClientSpec(
      num_epochs=FLAGS.client_epochs_per_round,
      batch_size=FLAGS.client_batch_size,
      max_elements=FLAGS.max_elements_per_client)
  task = task_utils.create_task_from_flags(train_client_spec)

  big_rnn_model_fn, small_rnn_model_fn = models.make_big_and_small_stackoverflow_model_fn(
      task,
      big_embedding_size=96,
      big_lstm_size=670,
      small_embedding_size=FLAGS.small_embedding_size,
      small_lstm_size=FLAGS.small_lstm_size)

  if FLAGS.shrink_unshrink_type == 'identity':
    logging.info('using identity shrink')
    make_shrink = shrink_unshrink_tff.make_identity_shrink
    make_unshrink = shrink_unshrink_tff.make_identity_unshrink
    server_model_fn = big_rnn_model_fn
    client_model_fn = big_rnn_model_fn
  elif FLAGS.shrink_unshrink_type == 'layerwise':
    logging.info('using layerwise shrink')
    make_shrink = shrink_unshrink_tff.make_layerwise_projection_shrink
    make_unshrink = shrink_unshrink_tff.make_layerwise_projection_unshrink
    server_model_fn = big_rnn_model_fn
    client_model_fn = small_rnn_model_fn
  elif FLAGS.shrink_unshrink_type == 'client_layerwise':
    logging.info('using client_layerwise shrink')
    make_shrink = shrink_unshrink_tff.make_client_specific_layerwise_projection_shrink
    make_unshrink = shrink_unshrink_tff.make_client_specific_layerwise_projection_unshrink
    server_model_fn = big_rnn_model_fn
    client_model_fn = small_rnn_model_fn
  else:
    raise ValueError('invalid shrink unshrink passed')

  print('creating iterative process')
  # allows for modifications to lstm layers
  # left_mask = [-1, 0, 2, -1, 2, -1, 0, -1]
  # right_mask = [0, 1, 1, 1, 0, 0, -1, -1]

  # does not allow for modifications to lstm layers
  left_mask = [-1, 0, -1, -1, 2, -1, 0, -1]
  right_mask = [0, -1, -1, -1, 0, 0, -1, -1]
  if FLAGS.build_projection_matrix_type == 'normal':
    logging.info('using normal projection matrix')
    build_projection_matrix = simple_fedavg_tf.build_normal_projection_matrix
  elif FLAGS.build_projection_matrix_type == 'dropout':
    logging.info('using dropout projection matrix')
    build_projection_matrix = simple_fedavg_tf.build_dropout_projection_matrix
  elif FLAGS.build_projection_matrix_type == 'qr':
    logging.info('using qr projection matrix')
    build_projection_matrix = simple_fedavg_tf.build_qr_projection_matrix
  else:
    raise ValueError('invalid build_projection_matrix_type passed')
  shrink_unshrink_info = simple_fedavg_tf.LayerwiseProjectionShrinkUnshrinkInfoV2(
      left_mask=left_mask,
      right_mask=right_mask,
      build_projection_matrix=build_projection_matrix)

  iterative_process = simple_fedavg_tff.build_federated_shrink_unshrink_process(
      server_model_fn=server_model_fn,
      client_model_fn=client_model_fn,
      make_shrink=make_shrink,
      make_unshrink=make_unshrink,
      shrink_unshrink_info=shrink_unshrink_info,
      client_optimizer_fn=client_optimizer_fn,
      server_optimizer_fn=server_optimizer_fn)

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
      num_validation_examples=FLAGS.num_validation_examples)

  def validation_fn_from_state(state, round_num):
    return validation_fn(state.model_weights, round_num)

  checkpoint_manager, metrics_managers = training_utils.configure_managers(
      FLAGS.root_output_dir,
      FLAGS.experiment_name,
      rounds_per_checkpoint=FLAGS.rounds_per_checkpoint,
      csv_metrics_manager_save_mode=tff.simulation.SaveMode.WRITE)
  _write_hparam_flags()
  logging.info('about to run simulation')
  state = tff.simulation.run_simulation(
      process=training_process,
      client_selection_fn=client_selection_fn,
      total_rounds=FLAGS.total_rounds,
      validation_fn=validation_fn_from_state,
      file_checkpoint_manager=checkpoint_manager,
      metrics_managers=metrics_managers)
  test_fn = training_utils.create_test_fn(task)
  # test_metrics = test_fn(state.model)
  test_metrics = test_fn(state.model_weights)
  logging.info('Test metrics:\n %s', pprint.pformat(test_metrics))
  for metrics_manager in metrics_managers:
    metrics_manager.save_metrics(test_metrics, FLAGS.total_rounds + 1)


if __name__ == '__main__':
  app.run(main)
