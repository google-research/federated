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
"""Runs federated training on various tasks using Federated Reconstruction."""

import collections
import functools
import os.path
from typing import Any, Callable, List, Optional, Tuple

from absl import app
from absl import flags
import tensorflow as tf
import tensorflow_federated as tff

from optimization.shared import optimizer_utils
from reconstruction import evaluation_computation
from reconstruction import reconstruction_model
from reconstruction import reconstruction_utils
from reconstruction import training_process
from reconstruction.movielens import federated_movielens
from reconstruction.shared import federated_evaluation
from reconstruction.shared import federated_trainer_utils
from reconstruction.stackoverflow import federated_stackoverflow
from utils import utils_impl

_SUPPORTED_TASKS = [
    'stackoverflow_nwp', 'movielens_mf', 'stackoverflow_nwp_finetune'
]

with utils_impl.record_hparam_flags() as optimizer_flags:
  # Define optimizer flags.
  # define_optimizer_flags defines flags prefixed by its argument.
  # For each prefix, two flags get created: <prefix>_optimizer, and
  # <prefix>_learning_rate.
  optimizer_utils.define_optimizer_flags('client')
  optimizer_utils.define_optimizer_flags('server')
  # Ignored when the task is `stackoverflow_nwp_finetune`.
  optimizer_utils.define_optimizer_flags('reconstruction')
  # Only used when the task is `stackoverflow_nwp_finetune`. Ignored otherwise.
  optimizer_utils.define_optimizer_flags('finetune')

with utils_impl.record_hparam_flags() as shared_flags:
  # Federated training hyperparameters.
  # The flags below are passed to `run_federated`, and are common to all the
  # tasks.
  flags.DEFINE_integer('client_batch_size', 20, 'Batch size on the clients.')
  flags.DEFINE_integer(
      'clients_per_round', 100,
      'How many clients to sample per round, for both train '
      'and evaluation.')
  flags.DEFINE_integer('total_rounds', 200, 'Number of total training rounds.')
  flags.DEFINE_boolean(
      'global_variables_only', False, 'If True, the model '
      'contains all model variables as global variables. This can be useful '
      'for baselines involving aggregating all variables. Must be True for '
      '`stackoverflow_nwp_finetune` task.')

  # Training loop configuration.
  flags.DEFINE_string(
      'experiment_name', None,
      'The name of this experiment. Will be append to --root_output_dir to '
      'separate experiment results.')
  flags.DEFINE_string('root_output_dir', '/tmp/fed_recon',
                      'Root directory for writing experiment output.')
  flags.DEFINE_integer(
      'rounds_per_eval', 1,
      'How often to evaluate the global model on the validation dataset.')
  flags.DEFINE_integer('rounds_per_checkpoint', 50,
                       'How often to checkpoint the global model.')

flags.mark_flag_as_required('experiment_name')
flags.mark_flag_as_required('root_output_dir')

with utils_impl.record_hparam_flags() as task_flags:
  # Task specification.
  flags.DEFINE_enum('task', None, _SUPPORTED_TASKS,
                    'Which task to perform federated training on.')

# Stack Overflow NWP flags.
with utils_impl.record_hparam_flags() as so_nwp_flags:
  flags.DEFINE_integer('so_nwp_vocab_size', 10000, 'Size of vocab to use.')
  flags.DEFINE_integer('so_nwp_num_oov_buckets', 1,
                       'Number of out of vocabulary buckets.')
  flags.DEFINE_integer('so_nwp_sequence_length', 20,
                       'Max sequence length to use.')
  flags.DEFINE_integer('so_nwp_max_elements_per_user', 1000, 'Max number of '
                       'training sentences to use per user.')
  flags.DEFINE_integer('so_nwp_embedding_size', 96,
                       'Dimension of word embedding to use.')
  flags.DEFINE_integer('so_nwp_latent_size', 670,
                       'Dimension of latent size to use in recurrent cell')
  flags.DEFINE_integer('so_nwp_num_layers', 1,
                       'Number of stacked recurrent layers to use.')
  flags.DEFINE_enum('so_nwp_split_dataset_strategy',
                    federated_trainer_utils.SPLIT_STRATEGY_AGGREGATED, [
                        federated_trainer_utils.SPLIT_STRATEGY_SKIP,
                        federated_trainer_utils.SPLIT_STRATEGY_AGGREGATED
                    ], 'Local variable reconstruction strategy.')
  flags.DEFINE_integer(
      'so_nwp_split_dataset_proportion', 2,
      'Parameter controlling how frequently an example is '
      'picked. Its exact usage depends on the '
      '`so_nwp_split_dataset_strategy` used. If `skip`, then '
      'every `so_nwp_split_dataset_proportion` example is used for '
      'reconstruction; if `aggregated`, then the proportion '
      'of the first 1 / `so_nwp_split_dataset_proportion` '
      'examples is used for reconstruction.')
  flags.DEFINE_boolean(
      'so_nwp_compose_dataset_computation', False,
      'Whether to compose dataset computation with training and evaluation '
      'computations. If True, may speed up experiments by parallelizing '
      'dataset computations in multimachine setups. Not currently supported '
      'in OSS. ')

# MovieLens matrix factorization flags. See `federated_movielens` for more
# details.
with utils_impl.record_hparam_flags() as ml_mf_flags:
  flags.DEFINE_boolean(
      'ml_mf_split_by_user', True,
      'Whether to split MovieLens data into train/val/test by '
      'user ID or by timestamp.')
  flags.DEFINE_float('ml_mf_split_train_fraction', 0.8,
                     'The fraction of the data to use for the train set.')
  flags.DEFINE_float('ml_mf_split_val_fraction', 0.1,
                     'The fraction of the data to use for the val set.')
  flags.DEFINE_boolean(
      'ml_mf_normalize_ratings', False,
      'Whether to normalize ratings to be in [-1, 1] via a '
      'linear scaling.')
  flags.DEFINE_integer(
      'ml_mf_max_examples_per_user', 300,
      'If not None, limit the number of examples per user to '
      'this many examples.')
  flags.DEFINE_integer('ml_mf_num_items', 3706,
                       'Number of items in the preferences matrix.')
  flags.DEFINE_integer(
      'ml_mf_num_latent_factors', 50,
      'Dimensionality of the learned user/item embeddings '
      'used to factorize the preferences matrix.')
  flags.DEFINE_boolean(
      'ml_mf_add_biases', False,
      'If True, add three bias terms: (1) user-specific bias, '
      '(2) item-specific bias, and (3) global bias.')
  flags.DEFINE_float(
      'ml_mf_l2_regularization', 0.0,
      'The constant used to scale L2 regularization on all '
      'weights.')
  flags.DEFINE_float(
      'ml_mf_spreadout_lambda', 0.0,
      'Scaling constant for spreadout regularization on item '
      'embeddings.')
  flags.DEFINE_float(
      'ml_mf_accuracy_threshold', 0.5,
      'Threshold to use to determine whether a prediction is '
      'considered correct for metrics.')
  flags.DEFINE_string(
      'ml_mf_dataset_path',
      'http://files.grouplens.org/datasets/movielens/ml-1m.zip',
      'URL or local path to MovieLens data.')

with utils_impl.record_hparam_flags() as recon_flags:
  # Reconstruction parameters.
  # For a `stackoverflow_nwp_finetune` task, three flags are ignored / un-used:
  # `split_data`, `evaluate_reconstruction`, and `jointly_train_variables`.
  # Other recon_flags are ignored in the federated training process and are only
  # used in the federated evaluation. Specifically, the reconstrunction set is
  # used as a fine-tuning set, and the post-reconstruction set is used for
  # evaluating the fine-tuned model.
  flags.DEFINE_integer(
      'recon_epochs_max', 1,
      'The integer maximum number of iterations over the dataset to make '
      'during reconstruction.')
  flags.DEFINE_boolean(
      'recon_epochs_constant', True,
      'If True, use `recon_epochs_max` as the constant number of '
      'iterations to make during reconstruction. If False, the number of '
      'iterations is min(round_num, recon_epochs_max).')
  flags.DEFINE_integer(
      'recon_steps_max', None,
      'If not None, the integer maximum number of steps (batches) to iterate '
      'through during reconstruction. This maximum number of steps is across '
      'all reconstruction iterations, i.e. it is applied after '
      '`recon_epochs_max` and `recon_epochs_constant`. '
      'If None, this has no effect.')
  flags.DEFINE_integer(
      'post_recon_epochs', 1,
      'The integer constant number of iterations to make over client data '
      'after reconstruction.')
  flags.DEFINE_integer(
      'post_recon_steps_max', None,
      'If not None, the integer maximum number of steps (batches) to iterate '
      'through after reconstruction. If None, this has no effect.')
  flags.DEFINE_boolean(
      'split_dataset', False,
      'If True, splits `client_dataset` in half for each user, using '
      'even-indexed entries in reconstruction and odd-indexed entries after '
      'reconstruction. If False, `client_dataset` is used for both '
      'reconstruction and post-reconstruction, with the above arguments '
      'applied. If True, splitting requires that multiple iterations through '
      'the dataset yield the same ordering. Note that this affects training '
      'only (splitting happens regardless during evaluation). Ignored when the '
      'task is `stackoverflow_nwp_finetune`.')
  flags.DEFINE_boolean(
      'evaluate_reconstruction', False,
      'If True, metrics (including loss) are computed on batches during '
      'reconstruction and post-reconstruction. If False, metrics are computed '
      'on batches only post-reconstruction, when global weights are being '
      'updated. Ignored when the task is `stackoverflow_nwp_finetune`.')
  flags.DEFINE_boolean(
      'jointly_train_variables', False,
      'Whether to train local variables after the reconstruction stage. '
      'Ignored when the task is `stackoverflow_nwp_finetune`.')

with utils_impl.record_hparam_flags() as finetune_flags:
  flags.DEFINE_integer(
      'client_epochs_per_round', None,
      'Number of epochs in the client to take per round. '
      'Used only when `global_variables_only` is True.')

with utils_impl.record_hparam_flags() as dp_flags:
  # Flags for training with differential privacy.
  flags.DEFINE_float(
      'dp_noise_multiplier', None,
      'Noise multiplier for the Gaussian mechanism for DP on model updates. '
      'If None, differential privacy is not used.')
  flags.DEFINE_boolean('dp_zeroing', True,
                       'Whether to enable adaptive zeroing for model updates.')

with utils_impl.record_hparam_flags() as run_flags:
  flags.DEFINE_integer(
      'run_number', None, 'This is an unused flag that can be used to log the '
      'current experiment ID number or run an experiment '
      'multiple times.')

FLAGS = flags.FLAGS

TASK_FLAGS = collections.OrderedDict(
    stackoverflow_nwp=so_nwp_flags,
    movielens_mf=ml_mf_flags,
    stackoverflow_nwp_finetune=so_nwp_flags)

TASK_FLAG_PREFIXES = collections.OrderedDict(
    stackoverflow_nwp='so_nwp',
    movielens_mf='ml_mf',
    stackoverflow_nwp_finetune='so_nwp')


def _write_hparam_flags():
  """Creates an ordered dictionary of hyperparameter flags and writes to CSV."""
  hparam_dict = utils_impl.lookup_flag_values(shared_flags)

  # Update with optimizer flags corresponding to the chosen optimizers.
  opt_flag_dict = utils_impl.lookup_flag_values(optimizer_flags)
  opt_flag_dict = optimizer_utils.remove_unused_flags('client', opt_flag_dict)
  opt_flag_dict = optimizer_utils.remove_unused_flags('server', opt_flag_dict)
  if FLAGS.task == 'stackoverflow_nwp_finetune':
    opt_flag_dict = optimizer_utils.remove_unused_flags('finetune',
                                                        opt_flag_dict)
  else:
    opt_flag_dict = optimizer_utils.remove_unused_flags('reconstruction',
                                                        opt_flag_dict)
  hparam_dict.update(opt_flag_dict)

  # Update with task-specific flags.
  task_name = FLAGS.task
  if task_name in TASK_FLAGS:
    task_hparam_dict = utils_impl.lookup_flag_values(TASK_FLAGS[task_name])
    hparam_dict.update(task_hparam_dict)

  # Update with finetune flags
  if FLAGS.task == 'stackoverflow_nwp_finetune':
    finetune_hparam_dict = utils_impl.lookup_flag_values(finetune_flags)
    hparam_dict.update(finetune_hparam_dict)

  # Update with reconstruction flags.
  recon_hparam_dict = utils_impl.lookup_flag_values(recon_flags)
  hparam_dict.update(recon_hparam_dict)

  # Update with DP flags.
  dp_hparam_dict = utils_impl.lookup_flag_values(dp_flags)
  hparam_dict.update(dp_hparam_dict)

  # Update with run flags.
  run_hparam_dict = utils_impl.lookup_flag_values(run_flags)
  hparam_dict.update(run_hparam_dict)

  results_dir = os.path.join(FLAGS.root_output_dir, 'results',
                             FLAGS.experiment_name)
  utils_impl.create_directory_if_not_exists(results_dir)
  hparam_file = os.path.join(results_dir, 'hparams.csv')
  utils_impl.atomic_write_series_to_csv(hparam_dict, hparam_file)


def _get_task_args():
  """Returns an ordered dictionary of task-specific arguments.

  This method returns a dict of (arg_name, arg_value) pairs, where the
  arg_name has had the task name removed as a prefix (if it exists), as well
  as any leading `-` or `_` characters. This can then be passed to a
  task-specific function that expects arguments in this format.

  Returns:
    An ordered dictionary of (arg_name, arg_value) pairs.
  """
  task_name = FLAGS.task
  task_args = collections.OrderedDict()

  if task_name in TASK_FLAGS:
    task_flag_list = TASK_FLAGS[task_name]
    task_flag_dict = utils_impl.lookup_flag_values(task_flag_list)
    task_flag_prefix = TASK_FLAG_PREFIXES[task_name]
    for (key, value) in task_flag_dict.items():
      assert key.startswith(task_flag_prefix)
      key = key[len(task_flag_prefix):].lstrip('_-')
      task_args[key] = value
  return task_args


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  server_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('server')
  client_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('client')
  if FLAGS.task == 'stackoverflow_nwp_finetune':
    if not FLAGS.global_variables_only:
      raise ValueError('`FLAGS.global_variables_only` must be True for '
                       'a `stackoverflow_nwp_finetune` task.')
    if not FLAGS.client_epochs_per_round:
      raise ValueError('`FLAGS.client_epochs_per_round` must be set for '
                       'a `stackoverflow_nwp_finetune` task.')
    finetune_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags(
        'finetune')
  else:
    reconstruction_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags(
        'reconstruction')

  def iterative_process_builder(
      model_fn: Callable[[], reconstruction_model.ReconstructionModel],
      loss_fn: Callable[[], List[tf.keras.losses.Loss]],
      metrics_fn: Optional[Callable[[], List[tf.keras.metrics.Metric]]] = None,
      client_weight_fn: Optional[Callable[[Any], tf.Tensor]] = None,
      dataset_split_fn_builder: Callable[
          ..., reconstruction_utils.DatasetSplitFn] = reconstruction_utils
      .build_dataset_split_fn,
  ) -> tff.templates.IterativeProcess:
    """Creates an iterative process using a given TFF `model_fn`.

    For a `stackoverflow_nwp_finetune` task, the `model_fn` must return a model
    that has only global variables, and the argument `dataset_split_fn_builder`
    is ignored. The returned iterative process is basically the same as the one
    created by the standard `tff.learning.build_federated_averaging_process`.

    For other tasks, the returned iterative process performs the federated
    reconstruction algorithm defined by
    `training_process.build_federated_reconstruction_process`.

    Args:
      model_fn: A no-arg function returning a
        `reconstruction_model.ReconstructionModel`. The returned model must have
        only global variables for a `stackoverflow_nwp_finetune` task.
      loss_fn: A no-arg function returning a list of `tf.keras.losses.Loss`.
      metrics_fn: A no-arg function returning a list of
        `tf.keras.metrics.Metric`.
      client_weight_fn: Optional function that takes the local model's output,
        and returns a tensor that provides the weight in the federated average
        of model deltas. If not provided, the default is the total number of
        examples processed on device. If DP is used, this argument is ignored,
        and uniform client weighting is used.
      dataset_split_fn_builder: `DatasetSplitFn` builder. Returns a method used
        to split the examples into a reconstruction, and post-reconstruction
        set. Ignored for a `stackoverflow_nwp_finetune` task.

    Raises:
      ValueError: if `model_fn` returns a model with local variables for a
        `stackoverflow_nwp_finetune` task.

    Returns:
      A `tff.templates.IterativeProcess`.
    """

    # Get aggregation factory for DP, if needed.
    aggregation_factory = None
    client_weighting = client_weight_fn
    if FLAGS.dp_noise_multiplier is not None:
      aggregation_factory = tff.learning.dp_aggregator(
          noise_multiplier=FLAGS.dp_noise_multiplier,
          clients_per_round=float(FLAGS.clients_per_round),
          zeroing=FLAGS.dp_zeroing)
      # DP is only implemented for uniform weighting.
      client_weighting = lambda _: 1.0

    if FLAGS.task == 'stackoverflow_nwp_finetune':

      if not reconstruction_utils.has_only_global_variables(model_fn()):
        raise ValueError(
            '`model_fn` should return a model with only global variables. ')

      def fake_dataset_split_fn(
          client_dataset: tf.data.Dataset,
          round_num: tf.Tensor) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        del round_num
        return client_dataset.repeat(0), client_dataset.repeat(
            FLAGS.client_epochs_per_round)

      return training_process.build_federated_reconstruction_process(
          model_fn=model_fn,
          loss_fn=loss_fn,
          metrics_fn=metrics_fn,
          server_optimizer_fn=lambda: server_optimizer_fn(FLAGS.
                                                          server_learning_rate),
          client_optimizer_fn=lambda: client_optimizer_fn(FLAGS.
                                                          client_learning_rate),
          dataset_split_fn=fake_dataset_split_fn,
          client_weight_fn=client_weighting,
          aggregation_factory=aggregation_factory)

    return training_process.build_federated_reconstruction_process(
        model_fn=model_fn,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        server_optimizer_fn=lambda: server_optimizer_fn(FLAGS.
                                                        server_learning_rate),
        client_optimizer_fn=lambda: client_optimizer_fn(FLAGS.
                                                        client_learning_rate),
        reconstruction_optimizer_fn=functools.partial(
            reconstruction_optimizer_fn, FLAGS.reconstruction_learning_rate),
        dataset_split_fn=dataset_split_fn_builder(
            recon_epochs_max=FLAGS.recon_epochs_max,
            recon_epochs_constant=FLAGS.recon_epochs_constant,
            recon_steps_max=FLAGS.recon_steps_max,
            post_recon_epochs=FLAGS.post_recon_epochs,
            post_recon_steps_max=FLAGS.post_recon_steps_max,
            split_dataset=FLAGS.split_dataset),
        evaluate_reconstruction=FLAGS.evaluate_reconstruction,
        jointly_train_variables=FLAGS.jointly_train_variables,
        client_weight_fn=client_weighting,
        aggregation_factory=aggregation_factory)

  def evaluation_computation_builder(
      model_fn: Callable[[], reconstruction_model.ReconstructionModel],
      loss_fn: Callable[[], tf.losses.Loss],
      metrics_fn: Callable[[], List[tf.metrics.Metric]],
      dataset_split_fn_builder: Callable[
          ..., reconstruction_utils.DatasetSplitFn] = reconstruction_utils
      .build_dataset_split_fn,
  ) -> tff.Computation:
    """Creates a `tff.Computation` for federated evaluation.

    For a `stackoverflow_nwp_finetune` task, the returned `tff.Computation` is
    created by `federated_evaluation.build_federated_finetune_evaluation`. For
    other tasks, the returned `tff.Computation` is given by
    `evaluation_computation.build_federated_reconstruction_evaluation`.

    Args:
      model_fn: A no-arg function that returns a `ReconstructionModel`. The
        returned model must have only global variables for a
        `stackoverflow_nwp_finetune` task. This method must *not* capture
        Tensorflow tensors or variables and use them. Must be constructed
        entirely from scratch on each invocation, returning the same model each
        call will result in an error.
      loss_fn: A no-arg function returning a `tf.keras.losses.Loss` to use to
        evaluate the model. The final loss metric is the example-weighted mean
        loss across batches (and across clients).
      metrics_fn: A no-arg function returning a list of
        `tf.keras.metrics.Metric`s to use to evaluate the model. The final
        metrics are the example-weighted mean metrics across batches (and across
        clients).
      dataset_split_fn_builder: `DatasetSplitFn` builder. Returns a method used
        to split the examples into a reconstruction set (which is used as a
        fine-tuning set for a `stackoverflow_nwp_finetune` task), and an
        evaluation set.

    Returns:
      A `tff.Computation` for federated evaluation.
    """

    # For a `stackoverflow_nwp_finetune` task, the first dataset returned by
    # `dataset_split_fn` is used for fine-tuning global variables. For other
    # tasks, the first dataset is used for reconstructing local variables.
    dataset_split_fn = dataset_split_fn_builder(
        recon_epochs_max=FLAGS.recon_epochs_max,
        recon_epochs_constant=FLAGS.recon_epochs_constant,
        recon_steps_max=FLAGS.recon_steps_max,
        post_recon_epochs=FLAGS.post_recon_epochs,
        post_recon_steps_max=FLAGS.post_recon_steps_max,
        # Getting meaningful evaluation metrics requires splitting the data.
        split_dataset=True)

    if FLAGS.task == 'stackoverflow_nwp_finetune':
      return federated_evaluation.build_federated_finetune_evaluation(
          model_fn=model_fn,
          loss_fn=loss_fn,
          metrics_fn=metrics_fn,
          finetune_optimizer_fn=functools.partial(finetune_optimizer_fn,
                                                  FLAGS.finetune_learning_rate),
          dataset_split_fn=dataset_split_fn)

    return evaluation_computation.build_federated_reconstruction_evaluation(
        model_fn=model_fn,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        reconstruction_optimizer_fn=functools.partial(
            reconstruction_optimizer_fn, FLAGS.reconstruction_learning_rate),
        dataset_split_fn=dataset_split_fn)

  # Shared args, useful to support more tasks.
  shared_args = utils_impl.lookup_flag_values(shared_flags)
  shared_args['iterative_process_builder'] = iterative_process_builder
  shared_args['evaluation_computation_builder'] = evaluation_computation_builder

  task_args = _get_task_args()
  _write_hparam_flags()

  if FLAGS.task in ['stackoverflow_nwp', 'stackoverflow_nwp_finetune']:
    run_federated_fn = federated_stackoverflow.run_federated
  elif FLAGS.task == 'movielens_mf':
    run_federated_fn = federated_movielens.run_federated
  else:
    raise ValueError(
        '--task flag {} is not supported, must be one of {}.'.format(
            FLAGS.task, _SUPPORTED_TASKS))

  run_federated_fn(**shared_args, **task_args)


if __name__ == '__main__':
  app.run(main)
