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
"""Runs centralized training on various tasks with different optimizers.

The tasks, optimizers, and hyperparameters are all governed via flags. For more
details on the supported optimizers, see `shared/optimizer_utils.py`. For more
details on how the training loop is performed, see
`utils/centralized_training_loop.py`.
"""

import collections

from absl import app
from absl import flags
from absl import logging

from generalization.tasks import cifar100_image
from generalization.tasks import emnist_character
from generalization.tasks import shakespeare_character
from generalization.tasks import stackoverflow_word
from generalization.tasks import training_specs
from generalization.utils import centralized_training_loop
from generalization.utils import metric_utils
from utils import utils_impl
from utils.optimizers import optimizer_utils

with utils_impl.record_new_flags() as shared_flags:
  # Generic centralized training flags
  optimizer_utils.define_optimizer_flags('centralized')
  flags.DEFINE_string(
      'experiment_name', None,
      'Name of the experiment. Part of the name of the output directory.')
  flags.mark_flag_as_required('experiment_name')
  flags.DEFINE_string(
      'root_output_dir', '/tmp/centralized_opt',
      'The top-level output directory experiment runs. --experiment_name will '
      'be appended, and the directory will contain tensorboard logs, metrics '
      'written as CSVs, and a CSV of hyperparameter choices.')
  flags.DEFINE_integer('num_epochs', 50, 'Number of epochs to train.')
  flags.DEFINE_integer('epochs_per_checkpoint', 1,
                       'How often to checkpoint the model.')
  flags.DEFINE_integer('batch_size', 32, 'Size of batches for training.')
  flags.DEFINE_integer('centralized_shuffle_buffer_size', 10000,
                       'Shuffling buffer size for centralized training.')
  flags.DEFINE_integer(
      'steps_per_epoch', None, 'An optional integer specifying the total number'
      'of steps (batches of samples) per repoch. If not provided, the epoch'
      'will run until the input dataset is exhausted.')
  flags.DEFINE_integer('decay_epochs', None, 'Number of epochs before decaying '
                       'the learning rate.')
  flags.DEFINE_float('lr_decay', None, 'How much to decay the learning rate by'
                     ' at each stage.')

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
      'training dataset per evaluation round. If `None`, all training clients will be used.'
  )
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
      '1) Sampling training, validation and test clients for evaluation rounds. '
      'If `None`, no seed is used. '
      'Note that specifying `shared_random_seed` does not result in the same '
      'clients being sampled every round.')

# Task specific flags are defined in the corresponding task definition module.
TASK_FLAGS = collections.OrderedDict(
    cifar100_image=cifar100_image.cifar100_image_flags,
    emnist_character=emnist_character.emnist_character_flags,
    shakespeare_character=shakespeare_character.shakespeare_character_flags,
    stackoverflow_word=stackoverflow_word.stackoverflow_word_flags,
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

  # Update with task-specific flags.
  task_name = FLAGS.task
  if task_name in TASK_FLAGS:
    task_hparam_dict = utils_impl.lookup_flag_values(TASK_FLAGS[task_name])
    hparam_dict.update(task_hparam_dict)

  return hparam_dict


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  optimizer = optimizer_utils.create_optimizer_fn_from_flags('centralized')()

  task_spec = training_specs.TaskSpecCentralized(
      # The following 3 attributes are for TaskSpecCentralized.
      optimizer=optimizer,
      batch_size=FLAGS.batch_size,
      centralized_shuffle_buffer_size=FLAGS.centralized_shuffle_buffer_size,
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
      cifar100_image=cifar100_image.configure_training_centralized,
      emnist_character=emnist_character.configure_training_centralized,
      shakespeare_character=shakespeare_character
      .configure_training_centralized,
      stackoverflow_word=stackoverflow_word.configure_training_centralized,
  )

  config_fn = task_config_fn_dict[FLAGS.task]

  logging.info('Starting configuring task.')
  runner_spec = config_fn(task_spec, **(_get_task_kwargs_from_flags()))
  logging.info('Finished configuring task.')

  # TODO(b/194421866): Consider to incorporate write_hparams to end-to-end test.
  metric_utils.write_hparams(
      hparam_dict=_get_hparam_dict_from_flags(),
      root_output_dir=FLAGS.root_output_dir,
      experiment_name=FLAGS.experiment_name)

  checkpoint_callback, metrics_callbacks = metric_utils.configure_default_callbacks(
      root_output_dir=FLAGS.root_output_dir,
      experiment_name=FLAGS.experiment_name,
      epochs_per_checkpoint=FLAGS.epochs_per_checkpoint)

  logging.info('Starting `centralized_training_loop.run`.')
  centralized_training_loop.run(
      keras_model=runner_spec.keras_model,
      train_dataset=runner_spec.train_dataset,
      num_epochs=FLAGS.num_epochs,
      steps_per_epoch=FLAGS.steps_per_epoch,
      decay_epochs=FLAGS.decay_epochs,
      lr_decay=FLAGS.lr_decay,
      part_train_eval_fn=runner_spec.part_train_eval_fn,
      part_val_fn=runner_spec.part_val_fn,
      unpart_fn=runner_spec.unpart_fn,
      test_fn=runner_spec.test_fn,
      checkpoint_callback=checkpoint_callback,
      metrics_callbacks=metrics_callbacks)


if __name__ == '__main__':
  app.run(main)
