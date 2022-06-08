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
"""Trains and evaluates Stackoverflow NWP model."""

import collections
import functools
import random
from typing import List, Tuple
from absl import app
from absl import flags
from absl import logging

import tensorflow as tf
import tensorflow_federated as tff

from dp_matrix_factorization.dp_ftrl import aggregator_builder
from dp_matrix_factorization.dp_ftrl import dp_fedavg
from dp_matrix_factorization.dp_ftrl import training_loop
from dp_matrix_factorization.dp_ftrl.datasets import stackoverflow_word_prediction
from dp_matrix_factorization.dp_ftrl.models import stackoverflow_models
from utils import keras_metrics

IRRELEVANT_FLAGS = frozenset(iter(flags.FLAGS))

flags.DEFINE_string(
    'experiment_name', 'stackoverflow', 'The name of this experiment. Will be'
    'append to  --root_output_dir to separate experiment results.')
flags.DEFINE_string('root_output_dir', '/tmp/dpftrl/stackoverflow',
                    'Root directory for writing experiment output.')
flags.DEFINE_integer('rounds_per_checkpoint', 100,
                     'How often to checkpoint the global model.')
flags.DEFINE_integer(
    'rounds_per_eval', 20,
    'How often to evaluate the global model on the validation dataset.')
flags.DEFINE_integer('clients_per_thread', 1, 'TFF executor configuration.')

# Training
flags.DEFINE_integer('clients_per_round', 100,
                     'How many clients to sample per round.')
flags.DEFINE_integer('client_epochs_per_round', 1,
                     'Number of epochs in the client to take per round.')
flags.DEFINE_integer('client_batch_size', 16, 'Batch size used on the client.')
flags.DEFINE_integer('total_rounds', 10, 'Number of total training rounds.')
flags.DEFINE_integer(
    'total_epochs', 1,
    'If not None, use shuffling of clients instead of random sampling.')

# Optimizer
flags.DEFINE_enum('client_optimizer', 'sgd', ['sgd'], 'Client optimzier')
flags.DEFINE_float('client_lr', 0.02, 'Client learning rate.')
flags.DEFINE_float('server_lr', 1.0, 'Server learning rate.')
flags.DEFINE_float('server_momentum', 0.9, 'Server momentum for SGDM.')

# Differential privacy
flags.DEFINE_float('clip_norm', 1.0, 'Clip L2 norm.')
flags.DEFINE_float('noise_multiplier', 0.01,
                   'Noise multiplier for DP algorithm.')

_AGGREGATOR_METHOD = flags.DEFINE_enum(
    'aggregator_method', 'tree_aggregation',
    list(aggregator_builder.AGGREGATION_METHODS),
    'Enum indicating the aggregator method to use.')

flags.DEFINE_string(
    'lr_momentum_matrix_name', None,
    'Name of the mechanism (and partial path to stored matrix) '
    'for --aggregator_method=lr_momentum_matrix')

# Data
flags.DEFINE_integer('sequence_length', 20, 'Max sequence length to use.')
flags.DEFINE_integer('max_elements_per_user', 256, 'Max number of training '
                     'sentences to use per user.')
flags.DEFINE_integer(
    'num_validation_examples', 10000, 'Number of examples '
    'to use from test set for per-round validation.')

# Model
flags.DEFINE_integer('vocab_size', 10000, 'Size of vocab to use.')
flags.DEFINE_integer('num_oov_buckets', 1,
                     'Number of out of vocabulary buckets.')
flags.DEFINE_integer('embedding_size', 96,
                     'Dimension of word embedding to use.')
flags.DEFINE_integer('latent_size', 670,
                     'Dimension of latent size to use in recurrent cell')
flags.DEFINE_integer('num_layers', 1,
                     'Number of stacked recurrent layers to use.')
flags.DEFINE_boolean(
    'shared_embedding', False,
    'Boolean indicating whether to tie input and output embeddings.')

flags.DEFINE_boolean('use_synthetic_data', False,
                     'Use synthetic data (for testing)')

HPARAM_FLAGS = [f for f in flags.FLAGS if f not in IRRELEVANT_FLAGS]
FLAGS = flags.FLAGS


def _get_stackoverflow_metrics(vocab_size, num_oov_buckets):
  """Metrics for stackoverflow dataset."""
  special_tokens = stackoverflow_word_prediction.get_special_tokens(
      vocab_size, num_oov_buckets)
  pad_token = special_tokens.pad
  oov_tokens = special_tokens.oov
  eos_token = special_tokens.eos
  return [
      keras_metrics.MaskedCategoricalAccuracy(
          name='accuracy_with_oov', masked_tokens=[pad_token]),
      keras_metrics.MaskedCategoricalAccuracy(
          name='accuracy_no_oov', masked_tokens=[pad_token] + oov_tokens),
      keras_metrics.MaskedCategoricalAccuracy(
          name='accuracy_no_oov_or_eos',
          masked_tokens=[pad_token, eos_token] + oov_tokens),
  ]


def _preprocess_stackoverflow(vocab_size, num_oov_buckets, sequence_length,
                              num_validation_examples, client_batch_size,
                              client_epochs_per_round, max_elements_per_user):
  """Prepare stackoverflow dataset."""
  if FLAGS.use_synthetic_data:
    d = tff.simulation.datasets.stackoverflow.get_synthetic()
    train_clientdata, test_clientdata = d, d
    vocab_dict = tff.simulation.datasets.stackoverflow.get_synthetic_word_counts(
    )
    dataset_vocab = list(vocab_dict.keys())[:vocab_size]
  else:
    train_clientdata, _, test_clientdata = (
        tff.simulation.datasets.stackoverflow.load_data())
    dataset_vocab = stackoverflow_word_prediction.create_vocab(vocab_size)

  base_test_dataset = test_clientdata.create_tf_dataset_from_all_clients(seed=0)
  preprocess_val_and_test = stackoverflow_word_prediction.create_preprocess_fn(
      vocab=dataset_vocab,
      num_oov_buckets=num_oov_buckets,
      client_batch_size=128,
      client_epochs_per_round=client_epochs_per_round,
      max_sequence_length=sequence_length,
      max_elements_per_client=-1,
      max_shuffle_buffer_size=1)
  test_set = preprocess_val_and_test(
      base_test_dataset.skip(num_validation_examples))
  validation_set = preprocess_val_and_test(
      base_test_dataset.take(num_validation_examples))

  train_dataset_preprocess_comp = stackoverflow_word_prediction.create_preprocess_fn(
      vocab=dataset_vocab,
      num_oov_buckets=num_oov_buckets,
      client_batch_size=client_batch_size,
      client_epochs_per_round=client_epochs_per_round,
      max_sequence_length=sequence_length,
      max_elements_per_client=max_elements_per_user,
      max_shuffle_buffer_size=max_elements_per_user)

  @tff.tf_computation(tf.string)
  def train_dataset_computation(client_id):
    client_train_data = train_clientdata.dataset_computation(client_id)
    return train_dataset_preprocess_comp(client_train_data)

  return train_dataset_computation, train_clientdata, validation_set, test_set


def _client_optimizer_fn(name, learning_rate):
  if name == 'sgd':
    return tf.keras.optimizers.SGD(learning_rate)
  else:
    raise ValueError('Unknown client optimizer name {}'.format(name))


def _sample_client_ids(
    num_clients: int,
    client_data: tff.simulation.datasets.ClientData,
    round_num: int,
    epoch: int,
) -> Tuple[List, int]:  # pylint: disable=g-bare-generic
  """Returns a random subset of client ids."""
  del round_num  # Unused.
  return random.sample(client_data.client_ids, num_clients), epoch


def _build_tff_learning_model_and_process(input_spec, test_metrics):
  """Build `tff.learning` iterative process."""

  def tff_model_fn():
    keras_model = stackoverflow_models.create_recurrent_model(
        vocab_size=FLAGS.vocab_size,
        embedding_size=FLAGS.embedding_size,
        latent_size=FLAGS.latent_size,
        num_layers=FLAGS.num_layers,
        shared_embedding=FLAGS.shared_embedding)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    return tff.learning.from_keras_model(
        keras_model=keras_model, input_spec=input_spec, loss=loss)

  def evaluate_fn(model_weights, dataset):
    keras_model = stackoverflow_models.create_recurrent_model(
        vocab_size=FLAGS.vocab_size,
        embedding_size=FLAGS.embedding_size,
        latent_size=FLAGS.latent_size,
        num_layers=FLAGS.num_layers,
        shared_embedding=FLAGS.shared_embedding)
    model_weights.assign_weights_to(keras_model)
    metrics = dp_fedavg.keras_evaluate(keras_model, dataset, test_metrics)
    return collections.OrderedDict(
        (test_metric.name, metric.numpy())
        for test_metric, metric in zip(test_metrics, metrics))

  client_optimizer_fn = functools.partial(
      _client_optimizer_fn,
      name=FLAGS.client_optimizer,
      learning_rate=FLAGS.client_lr)

  aggregator_factory = aggregator_builder.build_aggregator(
      aggregator_method=_AGGREGATOR_METHOD.value,
      model_fn=tff_model_fn,
      clip_norm=FLAGS.clip_norm,
      noise_multiplier=FLAGS.noise_multiplier,
      clients_per_round=FLAGS.clients_per_round,
      num_rounds=FLAGS.total_rounds,
      noise_seed=None,
      momentum=FLAGS.server_momentum,
      lr_momentum_matrix_name=FLAGS.lr_momentum_matrix_name)

  if _AGGREGATOR_METHOD.value in ['opt_momentum_matrix', 'lr_momentum_matrix']:
    # If we directly factorize the momentum matrix, the momentum portion of the
    # update is already handled directly in the aggregator--so we disable in the
    # server optimizer.
    server_optimizer_momentum_value = 0
  else:
    server_optimizer_momentum_value = FLAGS.server_momentum

  iterative_process = dp_fedavg.build_dpftrl_fedavg_process(
      tff_model_fn,
      client_optimizer_fn=client_optimizer_fn,
      server_learning_rate=FLAGS.server_lr,
      server_momentum=server_optimizer_momentum_value,
      server_nesterov=False,
      use_experimental_simulation_loop=True,
      dp_aggregator_factory=aggregator_factory,
  )

  return iterative_process, evaluate_fn


def train_and_eval():
  """Train and evaluate StackOver NWP task."""
  logging.info('Show FLAGS for debugging:')
  for f in HPARAM_FLAGS:
    logging.info('%s=%s', f, FLAGS[f].value)

  hparam_dict = collections.OrderedDict([
      (name, FLAGS[name].value) for name in HPARAM_FLAGS
  ])

  train_dataset_computation, train_set, validation_set, test_set = _preprocess_stackoverflow(
      FLAGS.vocab_size, FLAGS.num_oov_buckets, FLAGS.sequence_length,
      FLAGS.num_validation_examples, FLAGS.client_batch_size,
      FLAGS.client_epochs_per_round, FLAGS.max_elements_per_user)

  input_spec = train_dataset_computation.type_signature.result.element
  stackoverflow_metrics = _get_stackoverflow_metrics(FLAGS.vocab_size,
                                                     FLAGS.num_oov_buckets)
  iterative_process, evaluate_fn = _build_tff_learning_model_and_process(
      input_spec, stackoverflow_metrics)
  iterative_process = tff.simulation.compose_dataset_computation_with_learning_process(
      dataset_computation=train_dataset_computation, process=iterative_process)

  if FLAGS.total_epochs is None:

    def client_dataset_ids_fn(round_num: int, epoch: int):
      return _sample_client_ids(FLAGS.clients_per_round, train_set, round_num,
                                epoch)

    logging.info('Sample clients for max %d rounds', FLAGS.total_rounds)
    total_epochs = 0
  else:
    client_shuffer = training_loop.ClientIDShuffler(FLAGS.clients_per_round,
                                                    train_set)
    client_dataset_ids_fn = client_shuffer.sample_client_ids
    logging.info('Shuffle clients for max %d epochs and %d rounds',
                 FLAGS.total_epochs, FLAGS.total_rounds)
    total_epochs = FLAGS.total_epochs

  training_loop.run(
      iterative_process,
      client_dataset_ids_fn,
      validation_fn=functools.partial(evaluate_fn, dataset=validation_set),
      total_epochs=total_epochs,
      total_rounds=FLAGS.total_rounds,
      experiment_name=FLAGS.experiment_name,
      train_eval_fn=None,
      test_fn=functools.partial(evaluate_fn, dataset=test_set),
      root_output_dir=FLAGS.root_output_dir,
      hparam_dict=hparam_dict,
      rounds_per_eval=FLAGS.rounds_per_eval,
      rounds_per_checkpoint=FLAGS.rounds_per_checkpoint,
      rounds_per_train_eval=2000)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  train_and_eval()


if __name__ == '__main__':
  app.run(main)
