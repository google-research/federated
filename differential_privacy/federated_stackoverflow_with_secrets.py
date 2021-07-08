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

import pprint
from typing import Callable

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_federated as tff

from differential_privacy import stackoverflow_nwp_with_secrets
from optimization.tasks import training_specs
from utils import training_utils
from utils import utils_impl
from utils.optimizers import optimizer_utils

with utils_impl.record_hparam_flags() as optimizer_flags:
  # Defining optimizer flags
  optimizer_utils.define_optimizer_flags('client')
  optimizer_utils.define_optimizer_flags('server')

with utils_impl.record_hparam_flags() as shared_flags:
  # Federated training hyperparameters
  flags.DEFINE_integer('client_epochs_per_round', 1,
                       'Number of epochs in the client to take per round.')
  flags.DEFINE_integer('client_batch_size', 16, 'Batch size on the clients.')
  flags.DEFINE_integer('clients_per_round', 100,
                       'How many clients to sample per round.')
  flags.DEFINE_integer('client_datasets_random_seed', 1,
                       'Random seed for client sampling.')

  # Training loop configuration
  flags.DEFINE_integer('total_rounds', 1500, 'Number of total training rounds.')
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

with utils_impl.record_hparam_flags() as secrets_flags:
  # Flags for secret configuration.
  flags.DEFINE_integer('secret_len', 5, 'Number of tokens per secret.')
  flags.DEFINE_list(
      'secret_groups_num_clients', None, 'Comma separated list of integers '
      'representing the number of clients who receive that secret. Must be in '
      'one-to-one correspondence with values in '
      '`secret_groups_substitution_probs`.')
  flags.DEFINE_list(
      'secret_groups_substitution_probs', None, 'Comma separated list of floats'
      ' representing the probability that a sentence will be replaced by the '
      'secret if the client has a secret. Must be in one-to-one correspondence '
      'with values in `secret_groups_num_clients`.')
  flags.DEFINE_integer(
      'num_secrets_per_group', 5, 'Number of secrets for each secret group. '
      'So if there is a secret group "(1, 0.2)" and num_secrets_per_group is '
      '5, then there will be five different secrets that are inserted into one '
      'client each, with sentence substitution probability 0.2.')
  flags.DEFINE_integer(
      'num_reference_secrets', 65536, 'Number of reference secrets for '
      'computing exposure by extrapolation.')

FLAGS = flags.FLAGS


def parse_secret_group_info():
  """Parse secret group info strings."""
  secret_groups_num_clients = []
  for num_clients_str in FLAGS.secret_groups_num_clients:
    try:
      secret_groups_num_clients.append(int(num_clients_str))
    except ValueError:
      raise ValueError(
          f'Error parsing secret_groups_num_clients value "{num_clients_str}" '
          f'as  int.')

  secret_groups_substitution_probs = []
  for substitution_probs_str in FLAGS.secret_groups_substitution_probs:
    try:
      secret_groups_substitution_probs.append(float(substitution_probs_str))
    except ValueError:
      raise ValueError(f'Error parsing secret_groups_substitution_probs value '
                       f'"{substitution_probs_str}" as float.')

  if len(secret_groups_num_clients) != len(secret_groups_substitution_probs):
    raise ValueError(
        f'`secret_groups_num_clients` and `secret_groups_substitution_probs` '
        f'must be the same length. Found "{FLAGS.secret_groups_num_clients}" '
        f'and "{FLAGS.secret_groups_substitution_probs}".')

  secret_group_info = [(n, p) for n, p in zip(secret_groups_num_clients,
                                              secret_groups_substitution_probs)]

  if len(set(secret_group_info)) != len(secret_group_info):
    dups = list(
        set([x for x in secret_group_info if secret_group_info.count(x) > 1]))
    raise ValueError(
        f'Secret configs must be distinct. Found duplicates: {dups}.')

  return secret_group_info


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
    else:

      def client_weighting(local_outputs):
        return tf.cast(tf.squeeze(local_outputs['num_tokens']), tf.float32)

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

  secret_group_info = parse_secret_group_info()
  logging.info('Parsed secret config: %s', secret_group_info)

  runner_spec = stackoverflow_nwp_with_secrets.configure_training(
      task_spec,
      secret_len=FLAGS.secret_len,
      secret_group_info=secret_group_info,
      num_secrets_per_group=FLAGS.num_secrets_per_group,
      num_reference_secrets=FLAGS.num_reference_secrets)

  def round_end_evaluation_fn(state, round_num):
    if round_num % FLAGS.rounds_per_eval == 0:
      validation_metrics = runner_spec.validation_fn(state, round_num)
    else:
      validation_metrics = {}
    return validation_metrics

  checkpoint_manager, metrics_managers = training_utils.configure_managers(
      FLAGS.root_output_dir, FLAGS.experiment_name, FLAGS.rounds_per_checkpoint)

  state = tff.simulation.run_simulation(
      process=runner_spec.iterative_process,
      client_selection_fn=runner_spec.client_datasets_fn,
      total_rounds=FLAGS.total_rounds,
      validation_fn=round_end_evaluation_fn,
      file_checkpoint_manager=checkpoint_manager,
      metrics_managers=metrics_managers)

  test_metrics = runner_spec.test_fn(state)

  logging.info('Test metrics:\n %s', pprint.pformat(test_metrics))

  for metrics_manager in metrics_managers:
    metrics_manager.save_metrics(test_metrics, FLAGS.total_rounds + 1)


if __name__ == '__main__':
  app.run(main)
