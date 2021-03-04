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
"""Trains and evaluates Stackoverflow NWP model using TFF."""

import functools
import os.path

from absl import app
from absl import flags
from absl import logging

import tensorflow as tf
import tensorflow_federated as tff

from optimization.shared import keras_metrics
from optimization.shared import optimizer_utils
from utils import training_loop
from utils import training_utils
from utils import utils_impl
from utils.datasets import stackoverflow_word_prediction
from utils.models import stackoverflow_models

with utils_impl.record_new_flags():
  # Training hyperparameters
  flags.DEFINE_integer('clients_per_round', 10,
                       'How many clients to sample per round.')
  flags.DEFINE_integer('client_epochs_per_round', 1,
                       'Number of epochs in the client to take per round.')
  flags.DEFINE_integer('client_batch_size', 8, 'Batch size used on the client.')
  flags.DEFINE_integer('sequence_length', 20, 'Max sequence length to use.')
  flags.DEFINE_integer('max_elements_per_user', 1000, 'Max number of training '
                       'sentences to use per user.')
  flags.DEFINE_integer(
      'num_validation_examples', 10000, 'Number of examples '
      'to use from test set for per-round validation.')
  flags.DEFINE_boolean(
      'uniform_weighting', False,
      'Whether to weigh clients uniformly. If false, clients '
      'are weighted by the number of tokens.')

  # Optimizer configuration (this defines one or more flags per optimizer).
  utils_impl.define_optimizer_flags('server')
  utils_impl.define_optimizer_flags('client')

  # Modeling flags
  flags.DEFINE_integer('vocab_size', 10000, 'Size of vocab to use.')
  flags.DEFINE_integer('embedding_size', 96,
                       'Dimension of word embedding to use.')
  flags.DEFINE_integer('latent_size', 670,
                       'Dimension of latent size to use in recurrent cell')
  flags.DEFINE_integer('num_layers', 1,
                       'Number of stacked recurrent layers to use.')
  flags.DEFINE_boolean(
      'shared_embedding', False,
      'Boolean indicating whether to tie input and output embeddings.')

  # Differential privacy flags
  flags.DEFINE_float('clip', 0.05, 'Initial clip.')
  flags.DEFINE_float('noise_multiplier', None,
                     'Noise multiplier. If None, no DP is used.')
  flags.DEFINE_float('adaptive_clip_learning_rate', 0,
                     'Adaptive clip learning rate.')
  flags.DEFINE_float('target_unclipped_quantile', 0.5,
                     'Target unclipped quantile.')
  flags.DEFINE_float(
      'clipped_count_budget_allocation', 0.1,
      'Fraction of privacy budget to allocate for clipped counts.')

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

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))
  tff.backends.native.set_local_execution_context(max_fanout=10)

  model_builder = functools.partial(
      stackoverflow_models.create_recurrent_model,
      vocab_size=FLAGS.vocab_size,
      embedding_size=FLAGS.embedding_size,
      latent_size=FLAGS.latent_size,
      num_layers=FLAGS.num_layers,
      shared_embedding=FLAGS.shared_embedding)

  loss_builder = functools.partial(
      tf.keras.losses.SparseCategoricalCrossentropy, from_logits=True)

  special_tokens = stackoverflow_word_prediction.get_special_tokens(
      FLAGS.vocab_size)
  pad_token = special_tokens.pad
  oov_tokens = special_tokens.oov
  eos_token = special_tokens.eos

  def metrics_builder():
    return [
        keras_metrics.MaskedCategoricalAccuracy(
            name='accuracy_with_oov', masked_tokens=[pad_token]),
        keras_metrics.MaskedCategoricalAccuracy(
            name='accuracy_no_oov', masked_tokens=[pad_token] + oov_tokens),
        # Notice BOS never appears in ground truth.
        keras_metrics.MaskedCategoricalAccuracy(
            name='accuracy_no_oov_or_eos',
            masked_tokens=[pad_token, eos_token] + oov_tokens),
        keras_metrics.NumBatchesCounter(),
        keras_metrics.NumTokensCounter(masked_tokens=[pad_token]),
    ]

  train_dataset, _ = stackoverflow_word_prediction.get_federated_datasets(
      vocab_size=FLAGS.vocab_size,
      train_client_batch_size=FLAGS.client_batch_size,
      train_client_epochs_per_round=FLAGS.client_epochs_per_round,
      max_sequence_length=FLAGS.sequence_length,
      max_elements_per_train_client=FLAGS.max_elements_per_user)
  _, validation_dataset, test_dataset = stackoverflow_word_prediction.get_centralized_datasets(
      vocab_size=FLAGS.vocab_size,
      max_sequence_length=FLAGS.sequence_length,
      num_validation_examples=FLAGS.num_validation_examples)

  if FLAGS.uniform_weighting:
    client_weighting = tff.learning.ClientWeighting.UNIFORM
  else:
    client_weighting = tff.learning.ClientWeighting.NUM_EXAMPLES

  def model_fn():
    return tff.learning.from_keras_model(
        model_builder(),
        loss_builder(),
        input_spec=validation_dataset.element_spec,
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
      train_dataset, FLAGS.clients_per_round)

  evaluate_fn = training_utils.build_centralized_evaluate_fn(
      model_builder=model_builder,
      eval_dataset=validation_dataset,
      loss_builder=loss_builder,
      metrics_builder=metrics_builder)
  validation_fn = lambda state, round_num: evaluate_fn(state.model)

  evaluate_test_fn = training_utils.build_centralized_evaluate_fn(
      model_builder=model_builder,
      # Use both val and test for symmetry with other experiments, which
      # evaluate on the entire test set.
      eval_dataset=validation_dataset.concatenate(test_dataset),
      loss_builder=loss_builder,
      metrics_builder=metrics_builder)
  test_fn = lambda state: evaluate_test_fn(state.model)

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
      test_fn=test_fn,
      total_rounds=FLAGS.total_rounds,
      experiment_name=FLAGS.experiment_name,
      root_output_dir=FLAGS.root_output_dir,
      rounds_per_eval=FLAGS.rounds_per_eval,
      rounds_per_checkpoint=FLAGS.rounds_per_checkpoint)


if __name__ == '__main__':
  app.run(main)
