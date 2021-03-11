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
"""Runs federated training with differential privacy on various tasks."""

import os.path
from typing import Callable

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf
import tensorflow_federated as tff

from optimization.cifar100 import federated_cifar100
from optimization.emnist import federated_emnist
from optimization.emnist_ae import federated_emnist_ae
from optimization.shakespeare import federated_shakespeare
from optimization.shared import optimizer_utils
from optimization.shared import training_specs
from optimization.stackoverflow import federated_stackoverflow
from optimization.stackoverflow_lr import federated_stackoverflow_lr
from utils import training_loop
from utils import utils_impl

_SUPPORTED_TASKS = [
    'cifar100', 'emnist_cr', 'emnist_ae', 'shakespeare', 'stackoverflow_nwp',
    'stackoverflow_lr'
]

with utils_impl.record_hparam_flags() as optimizer_flags:
  # Defining optimizer flags
  optimizer_utils.define_optimizer_flags('client')
  optimizer_utils.define_optimizer_flags('server')

with utils_impl.record_hparam_flags() as shared_flags:
  # Federated training hyperparameters
  flags.DEFINE_integer('client_epochs_per_round', 1,
                       'Number of epochs in the client to take per round.')
  flags.DEFINE_integer('client_batch_size', 20, 'Batch size on the clients.')
  flags.DEFINE_integer('clients_per_round', 10,
                       'How many clients to sample per round.')
  flags.DEFINE_integer('client_datasets_random_seed', 1,
                       'Random seed for client sampling.')

  # Training loop configuration
  flags.DEFINE_integer('total_rounds', 200, 'Number of total training rounds.')
  flags.DEFINE_string(
      'experiment_name', None, 'The name of this experiment. Will be append to '
      '--root_output_dir to separate experiment results.')
  flags.DEFINE_string('root_output_dir', '/tmp/fed_opt/',
                      'Root directory for writing experiment output.')
  flags.DEFINE_integer(
      'rounds_per_eval', 1,
      'How often to evaluate the global model on the validation dataset.')
  flags.DEFINE_integer('rounds_per_checkpoint', 50,
                       'How often to checkpoint the global model.')

with utils_impl.record_hparam_flags() as dp_flags:
  # Differential privacy flags
  flags.DEFINE_float(
      'clip', None, 'Clip value for fixed clipping or initial clip for '
      'adaptive clipping. If None, no clipping is used.')
  flags.DEFINE_float('noise_multiplier', None,
                     'Noise multiplier. If None, non-DP aggregator is used.')
  flags.DEFINE_float(
      'adaptive_clip_learning_rate', None, 'Adaptive clip learning rate. If '
      'None, clip adaptation is not used.')
  flags.DEFINE_float('target_unclipped_quantile', 0.5,
                     'Target unclipped quantile.')
  flags.DEFINE_boolean('uniform_weighting', False,
                       'Whether to weigh clients uniformly.')

# Task specification
flags.DEFINE_enum('task', None, _SUPPORTED_TASKS,
                  'Which task to perform federated training on.')

FLAGS = flags.FLAGS


def _write_hparam_flags():
  """Returns an ordered dictionary of pertinent hyperparameter flags."""
  hparam_dict = utils_impl.lookup_flag_values(shared_flags)

  # Update with optimizer flags corresponding to the chosen optimizers.
  opt_flag_dict = utils_impl.lookup_flag_values(optimizer_flags)
  opt_flag_dict = optimizer_utils.remove_unused_flags('client', opt_flag_dict)
  opt_flag_dict = optimizer_utils.remove_unused_flags('server', opt_flag_dict)
  hparam_dict.update(opt_flag_dict)

  results_dir = os.path.join(FLAGS.root_output_dir, 'results',
                             FLAGS.experiment_name)
  utils_impl.create_directory_if_not_exists(results_dir)
  hparam_file = os.path.join(results_dir, 'hparams.csv')
  utils_impl.atomic_write_series_to_csv(hparam_dict, hparam_file)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  client_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('client')
  server_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('server')

  def iterative_process_builder(
      model_fn: Callable[[], tff.learning.Model],
  ) -> tff.templates.IterativeProcess:
    """Creates an iterative process using a given TFF `model_fn`.

    Args:
      model_fn: A no-arg function returning a `tff.learning.Model`.

    Returns:
      A `tff.templates.IterativeProcess`.
    """

    logging.info('Trainable weights:')
    for weight in model_fn().weights.trainable:
      logging.info('name: %s  shape: %s', weight.name, weight.shape)

    if FLAGS.uniform_weighting:
      client_weighting = tff.learning.ClientWeighting.UNIFORM
    elif FLAGS.task == 'shakespeare' or FLAGS.task == 'stackoverflow_nwp':

      def client_weighting(local_outputs):
        return tf.cast(tf.squeeze(local_outputs['num_tokens']), tf.float32)
    else:
      client_weighting = None

    if FLAGS.noise_multiplier is None:
      if FLAGS.uniform_weighting:
        aggregation_factory = tff.aggregators.UnweightedMeanFactory()
      else:
        aggregation_factory = tff.aggregators.MeanFactory()
      if FLAGS.clip is not None:
        if FLAGS.clip <= 0:
          raise ValueError('clip must be positive if clipping is enabled.')
        if FLAGS.adaptive_clip_learning_rate is None:
          clip = FLAGS.clip
        else:
          if FLAGS.adaptive_clip_learning_rate <= 0:
            raise ValueError('adaptive_clip_learning_rate must be positive if '
                             'adaptive clipping is enabled.')
          clip = tff.aggregators.PrivateQuantileEstimationProcess.no_noise(
              initial_estimate=FLAGS.clip,
              target_quantile=FLAGS.target_unclipped_quantile,
              learning_rate=FLAGS.adaptive_clip_learning_rate)
        aggregation_factory = tff.aggregators.clipping_factory(
            clip, aggregation_factory)
    else:
      if not FLAGS.uniform_weighting:
        raise ValueError(
            'Differential privacy is only implemented for uniform weighting.')
      if FLAGS.noise_multiplier <= 0:
        raise ValueError('noise_multiplier must be positive if DP is enabled.')
      if FLAGS.clip is None or FLAGS.clip <= 0:
        raise ValueError('clip must be positive if DP is enabled.')
      if FLAGS.adaptive_clip_learning_rate is None:
        aggregation_factory = tff.aggregators.DifferentiallyPrivateFactory.gaussian_fixed(
            noise_multiplier=FLAGS.noise_multiplier,
            clients_per_round=FLAGS.clients_per_round,
            clip=FLAGS.clip)
      else:
        if FLAGS.adaptive_clip_learning_rate <= 0:
          raise ValueError('adaptive_clip_learning_rate must be positive if '
                           'adaptive clipping is enabled.')
        aggregation_factory = tff.aggregators.DifferentiallyPrivateFactory.gaussian_adaptive(
            noise_multiplier=FLAGS.noise_multiplier,
            clients_per_round=FLAGS.clients_per_round,
            initial_l2_norm_clip=FLAGS.clip,
            target_unclipped_quantile=FLAGS.target_unclipped_quantile,
            learning_rate=FLAGS.adaptive_clip_learning_rate)

    return tff.learning.build_federated_averaging_process(
        model_fn=model_fn,
        server_optimizer_fn=server_optimizer_fn,
        client_weighting=client_weighting,
        client_optimizer_fn=client_optimizer_fn,
        model_update_aggregation_factory=aggregation_factory)

  task_spec = training_specs.TaskSpec(
      iterative_process_builder=iterative_process_builder,
      client_epochs_per_round=FLAGS.client_epochs_per_round,
      client_batch_size=FLAGS.client_batch_size,
      clients_per_round=FLAGS.clients_per_round,
      client_datasets_random_seed=FLAGS.client_datasets_random_seed)

  if FLAGS.task == 'cifar100':
    runner_spec = federated_cifar100.configure_training(task_spec)
  elif FLAGS.task == 'emnist_cr':
    runner_spec = federated_emnist.configure_training(task_spec)
  elif FLAGS.task == 'emnist_ae':
    runner_spec = federated_emnist_ae.configure_training(task_spec)
  elif FLAGS.task == 'shakespeare':
    runner_spec = federated_shakespeare.configure_training(task_spec)
  elif FLAGS.task == 'stackoverflow_nwp':
    runner_spec = federated_stackoverflow.configure_training(task_spec)
  elif FLAGS.task == 'stackoverflow_lr':
    runner_spec = federated_stackoverflow_lr.configure_training(task_spec)
  else:
    raise ValueError(
        '--task flag {} is not supported, must be one of {}.'.format(
            FLAGS.task, _SUPPORTED_TASKS))

  _write_hparam_flags()

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
