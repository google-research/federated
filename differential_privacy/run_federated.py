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

import functools
import pprint

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_federated as tff

from utils import task_utils
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
  flags.DEFINE_integer('total_rounds', 200, 'Number of total training rounds.')
  flags.DEFINE_string(
      'experiment_name', None, 'The name of this experiment. Will be append to '
      '--root_output_dir to separate experiment results.')
  flags.DEFINE_string('root_output_dir', '/tmp/fed_opt/',
                      'Root directory for writing experiment output.')
  flags.DEFINE_integer(
      'rounds_per_eval', 1,
      'How often to evaluate the global model on the validation dataset.')
  flags.DEFINE_integer(
      'num_validation_examples', -1, 'The number of validation'
      'examples to use. If set to -1, all available examples '
      'are used.')
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
with utils_impl.record_hparam_flags() as task_flags:
  task_utils.define_task_flags()

FLAGS = flags.FLAGS


def _write_hparam_flags():
  """Returns an ordered dictionary of pertinent hyperparameter flags."""
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

  client_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('client')
  server_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('server')

  train_client_spec = tff.simulation.baselines.ClientSpec(
      num_epochs=FLAGS.client_epochs_per_round,
      batch_size=FLAGS.client_batch_size,
      max_elements=FLAGS.max_elements_per_client)
  task = task_utils.create_task_from_flags(train_client_spec)

  logging.info('Trainable weights:')
  for weight in task.model_fn().weights.trainable:
    logging.info('name: %s  shape: %s', weight.name, weight.shape)

  if FLAGS.uniform_weighting:
    client_weighting = tff.learning.ClientWeighting.UNIFORM
  elif FLAGS.task == 'shakespeare_character' or FLAGS.task == 'stackoverflow_word':

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

  iterative_process = tff.learning.build_federated_averaging_process(
      model_fn=task.model_fn,
      server_optimizer_fn=server_optimizer_fn,
      client_weighting=client_weighting,
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
      num_validation_examples=FLAGS.num_validation_examples)

  def validation_fn_from_state(state, round_num):
    return validation_fn(state.model, round_num)

  checkpoint_manager, metrics_managers = training_utils.configure_managers(
      FLAGS.root_output_dir,
      FLAGS.experiment_name,
      rounds_per_checkpoint=FLAGS.rounds_per_checkpoint)
  _write_hparam_flags()
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
