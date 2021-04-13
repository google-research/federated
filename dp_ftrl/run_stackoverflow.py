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

from dp_ftrl import dp_fedavg
from dp_ftrl import optimizer_utils
from dp_ftrl import training_loop
from optimization.shared import keras_metrics
from utils.datasets import stackoverflow_word_prediction as stackoverflow_dataset
from utils.models import stackoverflow_models as models


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
    'total_epochs', None,
    'If not None, use shuffling of clients instead of random sampling.')

# Optimizer
flags.DEFINE_enum('client_optimizer', 'sgd', ['sgd'], 'Client optimzier')
flags.DEFINE_enum('server_optimizer', 'sgd',
                  ['sgd', 'dpftrl', 'dpsgd', 'dpsgdm', 'dpftrlm'],
                  'Server optimizer in federated optimizaiotn.')
flags.DEFINE_float('client_lr', 0.02, 'Client learning rate.')
flags.DEFINE_float('server_lr', 1.0, 'Server learning rate.')
flags.DEFINE_float('server_momentum', 0.9, 'Server momentum for SGDM.')

# Differential privacy
flags.DEFINE_float('clip_norm', 1.0, 'Clip L2 norm.')
flags.DEFINE_float('noise_multiplier', 0.01,
                   'Noise multiplier for DP algorithm.')

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

HPARAM_FLAGS = [f for f in flags.FLAGS if f not in IRRELEVANT_FLAGS]
FLAGS = flags.FLAGS


def _get_stackoverflow_metrics(vocab_size, num_oov_buckets):
  """Metrics for stackoverflow dataset."""
  special_tokens = stackoverflow_dataset.get_special_tokens(
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
  train_clientdata, _, test_clientdata = (
      tff.simulation.datasets.stackoverflow.load_data())
  dataset_vocab = stackoverflow_dataset.create_vocab(vocab_size)

  base_test_dataset = test_clientdata.create_tf_dataset_from_all_clients()
  preprocess_val_and_test = stackoverflow_dataset.create_preprocess_fn(
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

  train_dataset_preprocess_comp = stackoverflow_dataset.create_preprocess_fn(
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


def _server_optimizer_fn(model_weights, name, learning_rate, noise_std):
  """Returns server optimizer."""
  model_weight_shape = tf.nest.map_structure(tf.shape, model_weights)
  if name == 'sgd':
    return optimizer_utils.SGDServerOptimizer(learning_rate)
  elif name == 'sgdm':
    return optimizer_utils.DPSGDMServerOptimizer(
        learning_rate,
        momentum=FLAGS.server_momentum,
        noise_std=0,
        model_weight_shape=model_weight_shape)
  elif name == 'dpftrl':
    return optimizer_utils.DPFTRLMServerOptimizer(
        learning_rate,
        momentum=0,
        noise_std=noise_std,
        model_weight_shape=model_weight_shape)
  elif name == 'dpsgd':
    return optimizer_utils.DPSGDMServerOptimizer(
        learning_rate,
        momentum=0,
        noise_std=noise_std,
        model_weight_shape=model_weight_shape)
  elif name == 'dpsgdm':
    return optimizer_utils.DPSGDMServerOptimizer(
        learning_rate,
        momentum=FLAGS.server_momentum,
        noise_std=noise_std,
        model_weight_shape=model_weight_shape)
  elif name == 'dpftrlm':
    return optimizer_utils.DPFTRLMServerOptimizer(
        learning_rate,
        momentum=FLAGS.server_momentum,
        noise_std=noise_std,
        model_weight_shape=model_weight_shape)
  else:
    raise ValueError('Unknown server optimizer name {}'.format(name))


def _build_server_state_epoch_update_fn(server_optimizer_name, model_fn,
                                        server_optimizer_fn):
  """Build server update function: tree restart for FTRL."""
  if server_optimizer_name == 'dpftrl' or server_optimizer_name == 'dpftrlm':
    # A server optimzier is built to get a new state to restart the optimizer.
    # A model is built to initialize the optimizer because the optimizer state
    # depends on the shape of the model weights. The model and optimizer are
    # only constructed once.
    model = model_fn()
    optimizer = server_optimizer_fn(model.weights.trainable)

    def server_state_update(state):
      return tff.structure.update_struct(
          state,
          model=state.model,
          optimizer_state=optimizer.restart_dp_tree(state.model.trainable),
          round_num=state.round_num)

    return server_state_update
  else:
    return None


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


def train_and_eval():
  """Train and evaluate StackOver NWP task."""
  logging.info('Show FLAGS for debugging:')
  for f in HPARAM_FLAGS:
    logging.info('%s=%s', f, FLAGS[f].value)

  train_dataset_computation, train_set, validation_set, test_set = _preprocess_stackoverflow(
      FLAGS.vocab_size, FLAGS.num_oov_buckets, FLAGS.sequence_length,
      FLAGS.num_validation_examples, FLAGS.client_batch_size,
      FLAGS.client_epochs_per_round, FLAGS.max_elements_per_user)

  input_spec = train_dataset_computation.type_signature.result.element

  def tff_model_fn():
    keras_model = models.create_recurrent_model(
        vocab_size=FLAGS.vocab_size,
        embedding_size=FLAGS.embedding_size,
        latent_size=FLAGS.latent_size,
        num_layers=FLAGS.num_layers,
        shared_embedding=FLAGS.shared_embedding)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    return dp_fedavg.KerasModelWrapper(keras_model, input_spec, loss)

  noise_std = FLAGS.clip_norm * FLAGS.noise_multiplier / float(
      FLAGS.clients_per_round)
  server_optimizer_fn = functools.partial(
      _server_optimizer_fn,
      name=FLAGS.server_optimizer,
      learning_rate=FLAGS.server_lr,
      noise_std=noise_std)
  client_optimizer_fn = functools.partial(
      _client_optimizer_fn,
      name=FLAGS.client_optimizer,
      learning_rate=FLAGS.client_lr)
  iterative_process = dp_fedavg.build_federated_averaging_process(
      tff_model_fn,
      dp_clip_norm=FLAGS.clip_norm,
      server_optimizer_fn=server_optimizer_fn,
      client_optimizer_fn=client_optimizer_fn)
  iterative_process = tff.simulation.compose_dataset_computation_with_iterative_process(
      dataset_computation=train_dataset_computation, process=iterative_process)

  keras_metics = _get_stackoverflow_metrics(FLAGS.vocab_size,
                                            FLAGS.num_oov_buckets)
  model = tff_model_fn()
  def evaluate_fn(model_weights, dataset):
    model.from_weights(model_weights)
    metrics = dp_fedavg.keras_evaluate(model.keras_model, dataset, keras_metics)
    return collections.OrderedDict(
        (metric.name, metric.result().numpy()) for metric in metrics)

  hparam_dict = collections.OrderedDict([
      (name, FLAGS[name].value) for name in HPARAM_FLAGS
  ])

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

  server_state_update_fn = _build_server_state_epoch_update_fn(
      FLAGS.server_optimizer, tff_model_fn, server_optimizer_fn)
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
      rounds_per_train_eval=2000,
      server_state_epoch_update_fn=server_state_update_fn)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  # Multi-GPU configuration
  client_devices = tf.config.list_logical_devices('GPU')
  server_device = tf.config.list_logical_devices('CPU')[0]
  tff.backends.native.set_local_execution_context(
      max_fanout=2 * FLAGS.clients_per_round,
      server_tf_device=server_device,
      client_tf_devices=client_devices,
      clients_per_thread=FLAGS.clients_per_thread)

  train_and_eval()


if __name__ == '__main__':
  app.run(main)
