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

import collections
import functools
import itertools

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from differential_privacy import secret_sharer
from utils import task_utils
from utils import training_utils
from utils import utils_impl
from utils.datasets import stackoverflow_word_prediction
from utils.models import stackoverflow_models
from utils.optimizers import optimizer_utils

INITIAL_CLIP = 0.1
ADAPTIVE_CLIP_LEARNING_RATE = 0.2

with utils_impl.record_hparam_flags() as optimizer_flags:
  # Defining optimizer flags
  optimizer_utils.define_optimizer_flags('client')
  optimizer_utils.define_optimizer_flags('server')

with utils_impl.record_hparam_flags() as shared_flags:
  # Federated training hyperparameters
  flags.DEFINE_integer('clients_per_thread', 1, 'TFF executor configuration.')
  flags.DEFINE_integer('client_epochs_per_round', 1,
                       'Number of epochs in the client to take per round.')
  flags.DEFINE_integer('client_batch_size', 16, 'Batch size on the clients.')
  flags.DEFINE_integer('clients_per_round', 100,
                       'How many clients to sample per round.')
  flags.DEFINE_integer('client_datasets_random_seed', 1,
                       'Random seed for client sampling.')
  flags.DEFINE_integer(
      'max_elements_per_client', 256, 'Maximum number of '
      'elements for each training client. If set to None, all '
      'available examples are used.')

  # Training loop configuration
  flags.DEFINE_integer('total_rounds', 1600, 'Number of total training rounds.')
  flags.DEFINE_string(
      'experiment_name', None, 'The name of this experiment. Will be append to '
      '--root_output_dir to separate experiment results.')
  flags.DEFINE_string('root_output_dir', '/tmp/fed_opt/',
                      'Root directory for writing experiment output.')
  flags.DEFINE_integer(
      'rounds_per_eval', 20,
      'How often to evaluate the global model on the validation dataset.')
  flags.DEFINE_integer('rounds_per_checkpoint', 50,
                       'How often to checkpoint the global model.')
  flags.DEFINE_integer(
      'num_validation_examples', -1, 'The number of validation'
      'examples to use. If set to -1, all available examples '
      'are used.')
  flags.DEFINE_boolean(
      'uniform_sampling', False, 'Whether to use uniform '
      'client sampling or permuted batches.')

with utils_impl.record_hparam_flags() as dp_flags:
  # Differential privacy flags
  flags.DEFINE_float('noise_multiplier', 0.0,
                     'Noise multiplier. If 0, non-DP aggregator is used.')
  flags.DEFINE_float(
      'target_unclipped_quantile', 0.5,
      'Target unclipped quantile. If 1.0, no clipping will be performed.')
  flags.DEFINE_boolean('uniform_weighting', False,
                       'Whether to weigh clients uniformly.')

with utils_impl.record_hparam_flags() as secrets_flags:
  # Flags for secret configuration.
  flags.DEFINE_integer('secret_len', 5, 'Number of tokens per secret.')
  flags.DEFINE_list(
      'secret_groups_num_clients', None, 'Comma separated list of integers '
      'representing the number of clients who receive that secret. Secret '
      'groups will be formed as cross-product of this with '
      'secret_groups_substitution_probs.')
  flags.DEFINE_list(
      'secret_groups_substitution_probs', None, 'Comma separated list of floats'
      ' representing the probability that a sentence will be replaced by the '
      'secret if the client has a secret. Secret groups will be formed as '
      'cross-product of this with secret_groups_num_clients.')
  flags.DEFINE_integer(
      'num_secrets_per_group', 5, 'Number of secrets for each secret group. '
      'So if there is a secret group "(1, 0.2)" and num_secrets_per_group is '
      '5, then there will be five different secrets that are inserted into one '
      'client each, with sentence substitution probability 0.2.')
  flags.DEFINE_integer(
      'num_reference_secrets', 65536, 'Number of reference secrets for '
      'computing exposure by extrapolation.')
  flags.DEFINE_integer('secret_seed', 0,
                       'Random seed for generating and inserting secrets.')

# Task specification
with utils_impl.record_hparam_flags() as task_flags:
  task_utils.define_task_flags()

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

  def find_dups(l):
    return list(set([x for x in l if l.count(x) > 1]))

  num_client_dups = find_dups(secret_groups_num_clients)
  if num_client_dups:
    raise ValueError(
        f'Num clients must be distinct. Found duplicates: {num_client_dups}.')

  secret_groups_substitution_probs = []
  for substitution_probs_str in FLAGS.secret_groups_substitution_probs:
    try:
      secret_groups_substitution_probs.append(float(substitution_probs_str))
    except ValueError:
      raise ValueError(f'Error parsing secret_groups_substitution_probs value '
                       f'"{substitution_probs_str}" as float.')

  sub_prob_dups = find_dups(secret_groups_substitution_probs)
  if sub_prob_dups:
    raise ValueError(f'Substitution probabilities must be distinct. '
                     f'Found duplicates: {sub_prob_dups}.')

  secret_group_info = list(
      itertools.product(secret_groups_num_clients,
                        secret_groups_substitution_probs))

  return secret_group_info


def _write_hparam_flags():
  """Returns an ordered dictionary of pertinent hyperparameter flags."""
  hparam_dict = utils_impl.lookup_flag_values(shared_flags)

  # Update with optimizer flags corresponding to the chosen optimizers.
  opt_flag_dict = utils_impl.lookup_flag_values(optimizer_flags)
  opt_flag_dict = optimizer_utils.remove_unused_flags('client', opt_flag_dict)
  opt_flag_dict = optimizer_utils.remove_unused_flags('server', opt_flag_dict)
  hparam_dict.update(opt_flag_dict)


def make_aggregation_factory():
  """Constructs aggregation factory from flags."""
  if FLAGS.noise_multiplier == 0:
    if FLAGS.uniform_weighting:
      aggregation_factory = tff.aggregators.UnweightedMeanFactory()
    else:
      aggregation_factory = tff.aggregators.MeanFactory()
    if FLAGS.target_unclipped_quantile < 1.0:
      clip = tff.aggregators.PrivateQuantileEstimationProcess.no_noise(
          initial_estimate=INITIAL_CLIP,
          target_quantile=FLAGS.target_unclipped_quantile,
          learning_rate=ADAPTIVE_CLIP_LEARNING_RATE)
      aggregation_factory = tff.aggregators.clipping_factory(
          clip, aggregation_factory)
    return aggregation_factory
  else:
    if not FLAGS.uniform_weighting:
      raise ValueError(
          'Differential privacy is only implemented for uniform weighting.')
    if FLAGS.noise_multiplier <= 0:
      raise ValueError('noise_multiplier must be positive if DP is enabled.')
    return tff.aggregators.DifferentiallyPrivateFactory.gaussian_adaptive(
        noise_multiplier=FLAGS.noise_multiplier,
        clients_per_round=FLAGS.clients_per_round,
        initial_l2_norm_clip=INITIAL_CLIP,
        target_unclipped_quantile=FLAGS.target_unclipped_quantile,
        learning_rate=ADAPTIVE_CLIP_LEARNING_RATE)


class PermuteAndBatch():
  """Permute users and return batches.

  This class creates a permutation of the supplied values and at each round
  returns the next batch of values, stepping through the permutation.
  """

  def __init__(self, values, seed, clients_per_round):
    rng = np.random.default_rng(seed)
    self._perm = rng.permutation(values)
    self._clients_per_round = clients_per_round

  def __call__(self, round_num):
    """Gets the batch for the `round_num`th round."""
    i = (round_num * self._clients_per_round) % len(self._perm)
    next_i = i + self._clients_per_round
    if next_i > len(self._perm):
      return list(self._perm[i:]) + list(self._perm[:next_i - len(self._perm)])
    else:
      return list(self._perm[i:next_i])


def train_and_eval():
  """Train and evaluate StackOver NWP task with secrets."""
  client_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('client')
  server_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('server')

  model_builder = functools.partial(
      stackoverflow_models.create_recurrent_model,
      vocab_size=FLAGS.stackoverflow_word_vocab_size,
      num_oov_buckets=FLAGS.stackoverflow_word_num_out_of_vocab_buckets)

  train_client_spec = tff.simulation.baselines.ClientSpec(
      num_epochs=FLAGS.client_epochs_per_round,
      batch_size=FLAGS.client_batch_size,
      max_elements=FLAGS.max_elements_per_client)
  task = tff.simulation.baselines.stackoverflow.create_word_prediction_task(
      train_client_spec, vocab_size=FLAGS.stackoverflow_word_vocab_size)

  logging.info('Trainable weights:')
  for weight in task.model_fn().trainable_variables:
    logging.info('name: %s  shape: %s', weight.name, weight.shape)

  if FLAGS.uniform_weighting:
    client_weighting = tff.learning.ClientWeighting.UNIFORM
  else:

    def client_weighting(local_outputs):
      return tf.cast(tf.squeeze(local_outputs['num_tokens']), tf.float32)

  aggregation_factory = make_aggregation_factory()

  training_process = tff.learning.build_federated_averaging_process(
      model_fn=task.model_fn,
      server_optimizer_fn=server_optimizer_fn,
      client_weighting=client_weighting,
      client_optimizer_fn=client_optimizer_fn,
      model_update_aggregation_factory=aggregation_factory)

  secret_group_info = parse_secret_group_info()

  num_train_secrets = len(secret_group_info) * FLAGS.num_secrets_per_group
  word_counts = tff.simulation.datasets.stackoverflow.load_word_counts(
      vocab_size=FLAGS.stackoverflow_word_vocab_size)
  secrets = secret_sharer.generate_secrets(
      word_counts, FLAGS.secret_len,
      num_train_secrets + FLAGS.num_reference_secrets)
  expanded_secret_group_info = []
  for config in secret_group_info:
    expanded_secret_group_info.extend([config] * FLAGS.num_secrets_per_group)
  train_secrets = collections.OrderedDict(
      zip(secrets[:num_train_secrets], expanded_secret_group_info))

  train_data = secret_sharer.stackoverflow_with_secrets(
      task.datasets.train_data, train_secrets, FLAGS.secret_seed)
  vocab = stackoverflow_word_prediction.create_vocab(
      FLAGS.stackoverflow_word_vocab_size)
  to_ids_fn = stackoverflow_word_prediction.build_to_ids_fn(
      vocab=vocab,
      max_sequence_length=FLAGS.secret_len,
      num_oov_buckets=FLAGS.stackoverflow_word_num_out_of_vocab_buckets)
  train_preprocess_fn = stackoverflow_word_prediction.create_preprocess_fn(
      vocab=vocab,
      num_oov_buckets=FLAGS.stackoverflow_word_num_out_of_vocab_buckets,
      client_batch_size=FLAGS.client_batch_size,
      client_epochs_per_round=FLAGS.client_epochs_per_round,
      max_sequence_length=FLAGS.stackoverflow_word_sequence_length,
      max_elements_per_client=FLAGS.max_elements_per_client)
  train_data = train_data.preprocess(train_preprocess_fn)
  training_process = (
      tff.simulation.compose_dataset_computation_with_iterative_process(
          train_data.dataset_computation, training_process))

  if FLAGS.uniform_sampling:
    training_selection_fn = functools.partial(
        tff.simulation.build_uniform_sampling_fn(
            train_data.client_ids,
            random_seed=FLAGS.client_datasets_random_seed),
        size=FLAGS.clients_per_round)
  else:
    training_selection_fn = PermuteAndBatch(train_data.client_ids,
                                            FLAGS.client_datasets_random_seed,
                                            FLAGS.clients_per_round)
  test_data = task.datasets.get_centralized_test_data()
  validation_data = test_data.take(FLAGS.num_validation_examples)
  federated_eval = tff.learning.build_federated_evaluation(task.model_fn)
  evaluation_selection_fn = lambda round_num: [validation_data]

  # TODO(b/210890827): Use a polymorphic computation if possible
  @tff.federated_computation(training_process.initialize.type_signature.result,
                             federated_eval.type_signature.parameter[1])
  def evaluation_fn(state, evaluation_data):
    return federated_eval(state.model, evaluation_data)

  program_state_manager, metrics_managers = training_utils.create_managers(
      FLAGS.root_output_dir, FLAGS.experiment_name)
  _write_hparam_flags()
  state = tff.simulation.run_training_process(
      training_process=training_process,
      training_selection_fn=training_selection_fn,
      total_rounds=FLAGS.total_rounds,
      evaluation_fn=evaluation_fn,
      evaluation_selection_fn=evaluation_selection_fn,
      rounds_per_evaluation=FLAGS.rounds_per_eval,
      program_state_manager=program_state_manager,
      rounds_per_saving_program_state=FLAGS.rounds_per_checkpoint,
      metrics_managers=metrics_managers)

  orig_test_fn = training_utils.create_test_fn(task)

  def test_fn(model_weights):
    """Computes standard eval metrics and adds exposure of all secrets."""
    test_metric_dict = orig_test_fn(model_weights)
    prediction_model = model_builder()
    model_weights.assign_weights_to(prediction_model)
    prediction_model.compile(loss=tf.keras.losses.CategoricalCrossentropy())
    cce = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.SUM)

    log_2_e = np.log2(np.e)

    def get_perplexity(secret):
      ids = to_ids_fn({'tokens': tf.convert_to_tensor(secret)})
      prediction = prediction_model.predict(tf.expand_dims(ids[:-1], axis=0))
      return log_2_e * cce(tf.expand_dims(ids[1:], axis=0), prediction)

    exposures = secret_sharer.compute_exposure(
        secrets=secrets[:num_train_secrets],
        reference_secrets=secrets[num_train_secrets:],
        get_perplexity=get_perplexity)
    for i, exposure in enumerate(exposures):
      group_info = secret_group_info[i // FLAGS.num_secrets_per_group]
      j = i % FLAGS.num_secrets_per_group
      metric_name = f'exposure_{group_info[0]}_{group_info[1]}_{j}'
      test_metric_dict[metric_name] = exposure

    return test_metric_dict

  test_metrics = test_fn(state.model)
  for metrics_manager in metrics_managers:
    metrics_manager.release(test_metrics, FLAGS.total_rounds + 1)


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
