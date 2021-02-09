# Copyright 2019, Google LLC.
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
"""Trains and evaluates an EMNIST classification model with DP-FedAvg."""

import functools
import os.path

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf
import tensorflow_federated as tff

from optimization.shared import optimizer_utils
from utils import training_loop
from utils import training_utils
from utils import utils_impl
from utils.datasets import emnist_dataset
from utils.models import emnist_models

with utils_impl.record_hparam_flags():
  # Experiment hyperparameters
  flags.DEFINE_enum(
      'model', 'cnn', ['cnn', '2nn'], 'Which model to use. This '
      'can be a convolutional model (cnn) or a two hidden-layer '
      'densely connected network (2nn).')
  flags.DEFINE_integer('client_batch_size', 20,
                       'Batch size used on the client.')
  flags.DEFINE_integer('clients_per_round', 10,
                       'How many clients to sample per round.')
  flags.DEFINE_integer(
      'client_epochs_per_round', 1,
      'Number of client (inner optimizer) epochs per federated round.')
  flags.DEFINE_boolean(
      'uniform_weighting', False,
      'Whether to weigh clients uniformly. If false, clients '
      'are weighted by the number of samples.')

  # Optimizer configuration (this defines one or more flags per optimizer).
  utils_impl.define_optimizer_flags('server')
  utils_impl.define_optimizer_flags('client')

  # Differential privacy flags
  flags.DEFINE_float('clip', 0.05, 'Initial clip.')
  flags.DEFINE_float('noise_multiplier', None,
                     'Noise multiplier. If None, no DP is used.')
  flags.DEFINE_float('adaptive_clip_learning_rate', 0,
                     'Adaptive clip learning rate.')
  flags.DEFINE_float('target_unclipped_quantile', 0.5,
                     'Target unclipped quantile.')

with utils_impl.record_new_flags() as training_loop_flags:
  flags.DEFINE_integer('total_rounds', 200, 'Number of total training rounds.')
  flags.DEFINE_string(
      'experiment_name', None, 'The name of this experiment. Will be append to '
      '--root_output_dir to separate experiment results.')
  flags.DEFINE_string('root_output_dir', '/tmp/differential_privacy/',
                      'Root directory for writing experiment output.')
  flags.DEFINE_integer(
      'rounds_per_eval', 1,
      'How often to evaluate the global model on the validation dataset.')
  flags.DEFINE_integer('rounds_per_checkpoint', 50,
                       'How often to checkpoint the global model.')
  flags.DEFINE_integer(
      'rounds_per_profile', 0,
      '(Experimental) How often to run the experimental TF profiler, if >0.')

FLAGS = flags.FLAGS

# End of hyperparameter flags.


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  emnist_train, _ = emnist_dataset.get_federated_datasets(
      train_client_batch_size=FLAGS.client_batch_size,
      train_client_epochs_per_round=FLAGS.client_epochs_per_round,
      only_digits=False)

  _, emnist_test = emnist_dataset.get_centralized_datasets()

  if FLAGS.model == 'cnn':
    model_builder = functools.partial(
        emnist_models.create_conv_dropout_model, only_digits=False)
  elif FLAGS.model == '2nn':
    model_builder = functools.partial(
        emnist_models.create_two_hidden_layer_model, only_digits=False)
  else:
    raise ValueError('Cannot handle model flag [{!s}].'.format(FLAGS.model))

  loss_builder = tf.keras.losses.SparseCategoricalCrossentropy
  metrics_builder = lambda: [tf.keras.metrics.SparseCategoricalAccuracy()]

  if FLAGS.uniform_weighting:
    client_weighting = tff.learning.ClientWeighting.UNIFORM
  else:
    client_weighting = tff.learning.ClientWeighting.NUM_EXAMPLES

  def model_fn():
    return tff.learning.from_keras_model(
        model_builder(),
        loss_builder(),
        input_spec=emnist_test.element_spec,
        metrics=metrics_builder())

  if FLAGS.noise_multiplier is not None:
    if not FLAGS.uniform_weighting:
      raise ValueError(
          'Differential privacy is only implemented for uniform weighting.')
    if FLAGS.noise_multiplier <= 0:
      raise ValueError('noise_multiplier must be positive if DP is enabled.')
    if FLAGS.clip is None or FLAGS.clip <= 0:
      raise ValueError('clip must be positive if DP is enabled.')

    if not FLAGS.adaptive_clip_learning_rate:
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
  else:
    if FLAGS.uniform_weighting:
      aggregation_factory = tff.aggregators.UnweightedMeanFactory()
    else:
      aggregation_factory = tff.aggregators.MeanFactory()

  server_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('server')
  client_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('client')
  iterative_process = tff.learning.build_federated_averaging_process(
      model_fn=model_fn,
      server_optimizer_fn=server_optimizer_fn,
      client_weighting=client_weighting,
      client_optimizer_fn=client_optimizer_fn,
      model_update_aggregation_factory=aggregation_factory)

  client_datasets_fn = training_utils.build_client_datasets_fn(
      emnist_train, FLAGS.clients_per_round)

  evaluate_fn = training_utils.build_centralized_evaluate_fn(
      eval_dataset=emnist_test,
      model_builder=model_builder,
      loss_builder=loss_builder,
      metrics_builder=metrics_builder)
  validation_fn = lambda model_weights, round_num: evaluate_fn(model_weights)

  logging.info('Training model:')
  logging.info(model_builder().summary())

  # Log hyperparameters to CSV
  hparam_dict = utils_impl.lookup_flag_values(utils_impl.get_hparam_flags())
  results_dir = os.path.join(FLAGS.root_output_dir, 'results',
                             FLAGS.experiment_name)
  utils_impl.create_directory_if_not_exists(results_dir)
  hparam_file = os.path.join(results_dir, 'hparams.csv')
  utils_impl.atomic_write_series_to_csv(hparam_dict, hparam_file)

  training_loop.run(
      iterative_process=iterative_process,
      client_datasets_fn=client_datasets_fn,
      validation_fn=validation_fn,
      total_rounds=FLAGS.total_rounds,
      experiment_name=FLAGS.experiment_name,
      root_output_dir=FLAGS.root_output_dir,
      rounds_per_eval=FLAGS.rounds_per_eval,
      rounds_per_checkpoint=FLAGS.rounds_per_checkpoint,
      rounds_per_profile=FLAGS.rounds_per_profile)


if __name__ == '__main__':
  app.run(main)
