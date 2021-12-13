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
"""Runs federated training on various tasks using a generalized form of FedAvg.

Specifically, we create (according to flags) an iterative processes that allows
for client and server learning rate schedules, as well as various client and
server optimization methods. For more details on the learning rate scheduling
and optimization methods, see `shared/optimizer_utils.py`. For details on the
iterative process, see `utils/fed_avg_schedule.py`.
"""

import collections
from typing import Callable

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_federated as tff

from generalization.tasks import cifar100_image
from generalization.tasks import emnist_character
from generalization.tasks import shakespeare_character
from generalization.tasks import stackoverflow_word
from generalization.tasks import training_specs
from generalization.utils import fed_avg_schedule
from generalization.utils import federated_training_loop
from generalization.utils import metric_utils
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
  flags.DEFINE_integer(
      'train_clients_per_round', 10,
      'How many training clients to sample per round during training.')

  # Training loop configuration
  flags.DEFINE_string(
      'experiment_name', None, 'The name of this experiment. Will be append to '
      '--root_output_dir to separate experiment results.')
  flags.mark_flag_as_required('experiment_name')
  flags.DEFINE_string('root_output_dir', '/tmp/fed_opt/',
                      'Root directory for writing experiment output.')
  flags.DEFINE_integer('total_rounds', 200, 'Number of total training rounds.')
  flags.DEFINE_integer(
      'rounds_per_eval', 1,
      'How often to evaluate the global model on the validation dataset.')
  flags.DEFINE_integer('rounds_per_checkpoint', 50,
                       'How often to checkpoint the global model.')

  flags.DEFINE_string(
      'sql_database', None,
      'An optional str indicating the data source. If set to None, the TFF '
      'original data source will be used. Otherwise the program will load '
      'SQL-based ClientData from `sql_database`.')
  flags.DEFINE_float(
      'unpart_clients_proportion',
      None,
      'An optional floating number in (0.0, 1.0) representing the proportion '
      'of un-participating clients among the total clients. '
      'If sql_database is not None, or if sql_database is None but the TFF '
      'original federated dataset source does *not* provide a vertical split, '
      'then `unpart_clients_proportion` must not be None. In this case, a '
      'random set of clients will be drawn from the total sets of clients. '
      'If sql_database is None, and the TFF original federated dataset source '
      'provides a vertical split, then `unpart_clients_proportion` must be '
      'None, and the original vertical split will be used.',
      lower_bound=0.0,
      upper_bound=1.0)
  flags.DEFINE_integer(
      'train_val_ratio_intra_client',
      None,
      'An optional integer representing the ratio of ratio of train-validation '
      'split for each client. '
      'If sql_database is not None, or if sql_database is None but the TFF '
      'original federated dataset does *not* provide a horizontal split, '
      'then `train_val_ratio_intra_client` must not be None. In this case, for '
      'each client, the validation dataset contains 1/(train_val_ratio+1) of '
      'total samples, round up if fractional. The training dataset contains '
      'the rest of samples. '
      'If sql_database is None, and the TFF original federated dataset '
      'provides a horizontal split, then then `train_val_ratio_intra_client` '
      'must be None, and the TFF original horizontal split will be used.',
      lower_bound=1)
  flags.DEFINE_float(
      'part_clients_subsampling_rate', 1.0,
      'A floating number in (0.0, 1.0] representing the actual proportion of '
      'candidate participating clients. If < 1.0, a random subset of clients '
      'will be drawn from the "candidate participating clients" that become '
      'the actual participating clients. This attribute is mostly intended for '
      'the ablation study on the effect of participation rate.')
  flags.DEFINE_boolean(
      'include_unpart_train_for_val', False,
      'Whether to include the training dataset of unparticipated clients for '
      'validation. Please refere to training_specs.py for the detailed doc.')
  flags.DEFINE_integer(
      'max_elements_per_client',
      None,
      'An optional integer controlling the maximum number of examples to take '
      'per client. If none, keep all elements for each client. This is intended '
      'primarily to contend with the small set of clients with tens of '
      'thousands of examples. This truncation is applied after all the previous '
      'splits, and effective for all the three-way split.',
      lower_bound=1)

  # Evaluation configuration
  flags.DEFINE_integer(
      'part_clients_per_eval', None,
      'An optional integer representing the number of clients taken from'
      'training dataset for evaluation per evaluation round, used for both '
      'training and valiadtion. '
      'If `None`, all training clients will be used.')
  flags.DEFINE_integer(
      'unpart_clients_per_eval', None,
      'An optional integer representing the number of clients taken from'
      'validation dataset. If `None`, all validation clients will be used.')
  flags.DEFINE_integer(
      'test_clients_for_eval', None,
      'An optional integer representing the number of clients taken from'
      'test dataset. If `None`, all validation clients will be used.')
  flags.DEFINE_boolean(
      'resample_eval_clients', False,
      'Whether resample validation clients every evaluation round')
  flags.DEFINE_integer(
      'eval_client_batch_size', 16,
      'An integer representing the batch size used on validation and test clients.'
  )

  flags.DEFINE_integer(
      'shared_random_seed', 1,
      'An optional integer used to seed the pseudo-random number generator. '
      'The seeds are shared across the following functions: '
      '1) Sampling training client for each training round. '
      '2) Sampling training, validation and test clients for evaluation rounds. '
      'If `None`, no seed is used. '
      'Note that specifying `shared_random_seed` does not result in the same '
      'clients being sampled every round in a given experiment.')
# Task specific flags are defined in the corresponding task definition module.
TASK_FLAGS = collections.OrderedDict(
    cifar100_image=cifar100_image.cifar100_image_flags,
    emnist_character=emnist_character.emnist_character_flags,
    stackoverflow_word=stackoverflow_word.stackoverflow_word_flags,
    shakespeare_character=shakespeare_character.shakespeare_character_flags,
)

_SUPPORTED_TASKS = list(TASK_FLAGS.keys())

with utils_impl.record_hparam_flags() as task_flags:
  flags.DEFINE_enum('task', None, _SUPPORTED_TASKS,
                    'Which task to perform training on.')

FLAGS = flags.FLAGS


def _get_task_kwargs_from_flags():
  """Get task-specific kwargs from FLAGS."""
  task_prefix_len_dict = collections.OrderedDict(
      cifar100_image=len(cifar100_image.FLAG_PREFIX),
      emnist_character=len(emnist_character.FLAG_PREFIX),
      shakespeare_character=len(shakespeare_character.FLAG_PREFIX),
      stackoverflow_word=len(stackoverflow_word.FLAG_PREFIX),
  )
  task_name = FLAGS.task
  prefix_len = task_prefix_len_dict[task_name]
  task_flag_dict = utils_impl.lookup_flag_values(TASK_FLAGS[task_name])
  return {key[prefix_len:]: value for key, value in task_flag_dict.items()}


def _get_hparam_dict_from_flags():
  """Creates an ordered dictionary of hyperparameter flags."""

  hparam_dict = utils_impl.lookup_flag_values(shared_flags)

  # Update with optimizer flags corresponding to the chosen optimizers.
  opt_flag_dict = utils_impl.lookup_flag_values(optimizer_flags)
  opt_flag_dict = optimizer_utils.remove_unused_flags('client', opt_flag_dict)
  opt_flag_dict = optimizer_utils.remove_unused_flags('server', opt_flag_dict)
  hparam_dict.update(opt_flag_dict)

  task_name = FLAGS.task
  if task_name in TASK_FLAGS:
    task_hparam_dict = utils_impl.lookup_flag_values(TASK_FLAGS[task_name])
    hparam_dict.update(task_hparam_dict)

  return hparam_dict


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  client_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('client')
  server_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('server')

  client_lr_schedule = optimizer_utils.create_lr_schedule_from_flags('client')
  server_lr_schedule = optimizer_utils.create_lr_schedule_from_flags('server')

  def iterative_process_builder(
      model_fn: Callable[[],
                         tff.learning.Model]) -> tff.templates.IterativeProcess:
    """Creates an iterative process using a given TFF `model_fn`.

    Args:
      model_fn: A no-arg function returning a `tff.learning.Model`.

    Returns:
      A `tff.templates.IterativeProcess`.
    """
    if FLAGS.task == 'stackoverflow_word':

      def client_weight_fn(local_outputs):
        return tf.cast(tf.squeeze(local_outputs['num_tokens']), tf.float32)
    else:
      client_weight_fn = None

    return fed_avg_schedule.build_fed_avg_process(
        model_fn=model_fn,
        client_optimizer_fn=client_optimizer_fn,
        client_lr=client_lr_schedule,
        server_optimizer_fn=server_optimizer_fn,
        server_lr=server_lr_schedule,
        client_weight_fn=client_weight_fn)

  task_spec = training_specs.TaskSpecFederated(
      iterative_process_builder=iterative_process_builder,
      client_epochs_per_round=FLAGS.client_epochs_per_round,
      client_batch_size=FLAGS.client_batch_size,
      train_clients_per_round=FLAGS.train_clients_per_round,
      rounds_per_eval=FLAGS.rounds_per_eval,
      # The remaining attributes are for base class TaskSpec.
      sql_database=FLAGS.sql_database,
      unpart_clients_proportion=FLAGS.unpart_clients_proportion,
      train_val_ratio_intra_client=FLAGS.train_val_ratio_intra_client,
      part_clients_subsampling_rate=FLAGS.part_clients_subsampling_rate,
      include_unpart_train_for_val=FLAGS.include_unpart_train_for_val,
      max_elements_per_client=FLAGS.max_elements_per_client,
      part_clients_per_eval=FLAGS.part_clients_per_eval,
      unpart_clients_per_eval=FLAGS.unpart_clients_per_eval,
      test_clients_for_eval=FLAGS.test_clients_for_eval,
      resample_eval_clients=FLAGS.resample_eval_clients,
      eval_client_batch_size=FLAGS.eval_client_batch_size,
      shared_random_seed=FLAGS.shared_random_seed)

  task_config_fn_dict = collections.OrderedDict(
      cifar100_image=cifar100_image.configure_training_federated,
      emnist_character=emnist_character.configure_training_federated,
      shakespeare_character=shakespeare_character.configure_training_federated,
      stackoverflow_word=stackoverflow_word.configure_training_federated,
  )

  config_fn = task_config_fn_dict[FLAGS.task]
  task_kwargs = _get_task_kwargs_from_flags()

  logging.info('Starting configuring task.')
  runner_spec = config_fn(task_spec, **task_kwargs)
  logging.info('Finished configuring task.')

  metric_utils.write_hparams(_get_hparam_dict_from_flags(),
                             FLAGS.root_output_dir, FLAGS.experiment_name)

  checkpoint_manager, metrics_managers = metric_utils.configure_default_managers(
      FLAGS.root_output_dir,
      FLAGS.experiment_name,
      rounds_per_checkpoint=FLAGS.rounds_per_checkpoint)

  logging.info('Starting `federated_training_loop.run_simulation`.')
  federated_training_loop.run_simulation(
      process=runner_spec.iterative_process,
      client_selection_fn=runner_spec.client_datasets_fn,
      total_rounds=FLAGS.total_rounds,
      part_train_eval_fn=runner_spec.part_train_eval_fn,
      part_val_fn=runner_spec.part_val_fn,
      unpart_fn=runner_spec.unpart_fn,
      test_fn=runner_spec.test_fn,
      file_checkpoint_manager=checkpoint_manager,
      metrics_managers=metrics_managers,
  )


if __name__ == '__main__':
  app.run(main)
