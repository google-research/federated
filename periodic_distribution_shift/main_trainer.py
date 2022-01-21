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
"""Runs federated training on various periodic distribution shift tasks."""

from absl import app
from absl import flags
import tensorflow as tf
import tensorflow_federated as tff

from periodic_distribution_shift import fedavg_temporal_kmeans
from periodic_distribution_shift import train_loop_kmeans
from periodic_distribution_shift import validation_utils
from periodic_distribution_shift.datasets import client_sampling
from periodic_distribution_shift.tasks import task_utils
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
  flags.DEFINE_string('root_output_dir', '/tmp/test/',
                      'Root directory for writing experiment output.')
  flags.DEFINE_integer('total_rounds', 200, 'Number of total training rounds.')
  flags.DEFINE_integer(
      'rounds_per_eval', 1,
      'How often to evaluate the global model on the validation dataset.')
  flags.DEFINE_integer('period', 128, 'Period of the distribution shift.')
  flags.DEFINE_integer('kmeans_k', 2, 'Number of cluster centers.')
  flags.DEFINE_integer('feature_dim', 128, 'Number of feature dimensions.')
  flags.DEFINE_string(
      'shift_fn', 'linear',
      'Which interpolation function to use. Can choose from '
      '`linear` or `cosine`.')
  flags.DEFINE_float(
      'shift_p', 1., 'The exponent added to the interpolation function. '
      'Used to simulate the bias of the data distribution, '
      'caused by difference in training speed on different '
      'subpopulations.')
  flags.DEFINE_boolean('post_kmeans', False,
                       'Whether to use vanilla kmeans without any prior.')
  flags.DEFINE_boolean('aggregated_kmeans', False,
                       'Whether to use aggregated kmeans')
  flags.DEFINE_float(
      'geo_lr', 0.2, 'The step size for the geometric updates on the '
      'distance scalar.')
  flags.DEFINE_boolean(
      'rescale_eval', False, 'Whether to apply the distance scalar during '
      'evaluation.')
  flags.DEFINE_integer('num_tests', 5, 'Number of tests to run.')
  flags.DEFINE_float('clip_norm', -1., 'Maximum norm for gradient clipping.')
  flags.DEFINE_string('prior_fn', 'linear',
                      'Prior guess of the distribution shift.')
  flags.DEFINE_boolean(
      'zero_mid', False,
      'Whether to set the step size of geometric update to 0 '
      'in the middle of each period.')
  flags.DEFINE_float(
      'label_smooth_w', 0., 'Weight of label smoothing regularization on the '
      'unselected branch. Only effective when '
      '`aggregated_kmeans = True`.')
  flags.DEFINE_float(
      'label_smooth_eps', 0.2,
      'Epsilon of the label smoothing for the unselected branch.'
      'The value should be within 0 to 1, where 1 enforces the '
      'prediction to be uniform on all labels, and 0 falls back '
      'to cross entropy loss on one-hot label.')

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


def train_and_eval():
  """Train and evaluate periodic distribution shift tasks."""
  client_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('client')
  server_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('server')
  client_lr_schedule = optimizer_utils.create_lr_schedule_from_flags('client')
  server_lr_schedule = optimizer_utils.create_lr_schedule_from_flags('server')

  train_client_spec = tff.simulation.baselines.ClientSpec(
      num_epochs=FLAGS.client_epochs_per_round,
      batch_size=FLAGS.client_batch_size,
      max_elements=FLAGS.max_elements_per_client)
  task = task_utils.create_task_from_flags(train_client_spec)

  iterative_process = fedavg_temporal_kmeans.build_fed_avg_process(
      model_fn=task.model_fn,
      client_optimizer_fn=client_optimizer_fn,
      client_lr=client_lr_schedule,
      server_optimizer_fn=server_optimizer_fn,
      server_lr=server_lr_schedule,
      client_weight_fn=None,
      kmeans_k=FLAGS.kmeans_k,
      feature_dim=FLAGS.feature_dim,
      aggregated_kmeans=FLAGS.aggregated_kmeans,
      clip_norm=FLAGS.clip_norm)

  client_sampling_fn = client_sampling.build_time_varying_dataset_fn(
      train_client_spec=train_client_spec,
      clients_per_round=FLAGS.clients_per_round,
      period=FLAGS.period,
      shift_fn=FLAGS.shift_fn,
      shift_p=FLAGS.shift_p,
      task_name=FLAGS.task,
      random_seed=FLAGS.client_datasets_random_seed)

  validation_fn = validation_utils.create_general_validation_fn(
      task,
      validation_frequency=FLAGS.rounds_per_eval,
      kmeans_eval=True,
      k_total=FLAGS.kmeans_k,
      feature_dim=FLAGS.feature_dim)

  def validation_fn_from_state(state, round_num):
    if FLAGS.rescale_eval:
      dist_scalar = state.dist_scalar
    else:
      dist_scalar = tf.constant(1., tf.float32)
    return validation_fn(
        state.model,
        state.kmeans_centers,
        dist_scalar=dist_scalar,
        round_num=round_num)

  # TODO(b/193904908): checkpoint will be saved every round. Make it
  # configurable.
  checkpoint_manager, metrics_managers = training_utils.create_managers(
      FLAGS.root_output_dir,
      FLAGS.experiment_name)
  _write_hparam_flags()

  test_fn = validation_utils.create_general_test_fn(
      task,
      kmeans_eval=True,
      k_total=FLAGS.kmeans_k,
      feature_dim=FLAGS.feature_dim)

  train_loop_kmeans.run_simulation_with_kmeans(
      train_process=iterative_process,
      period=FLAGS.period,
      client_selection_fn=client_sampling_fn,
      total_rounds=FLAGS.total_rounds,
      file_checkpoint_manager=checkpoint_manager,
      metrics_managers=metrics_managers,
      validation_fn=validation_fn_from_state,
      test_fn=test_fn,
      test_freq=FLAGS.rounds_per_eval,
      num_tests=FLAGS.num_tests,
      server_optimizer_fn=server_optimizer_fn,
      server_lr_schedule=server_lr_schedule,
      model_fn=task.model_fn,
      aggregated_kmeans=FLAGS.aggregated_kmeans,
      geo_lr=FLAGS.geo_lr,
      prior_fn=FLAGS.prior_fn,
      zero_mid=FLAGS.zero_mid,
  )


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  # Multi-GPU configuration
  client_devices = tf.config.list_logical_devices('GPU')
  server_device = tf.config.list_logical_devices('CPU')[0]
  tff.backends.native.set_local_python_execution_context(
      max_fanout=2 * FLAGS.clients_per_round,
      server_tf_device=server_device,
      client_tf_devices=client_devices,
      clients_per_thread=FLAGS.clients_per_thread)

  train_and_eval()


if __name__ == '__main__':
  app.run(main)
