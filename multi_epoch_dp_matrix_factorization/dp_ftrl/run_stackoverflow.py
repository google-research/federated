# Copyright 2023, Google LLC.
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

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from multi_epoch_dp_matrix_factorization.dp_ftrl import aggregator_builder
from multi_epoch_dp_matrix_factorization.dp_ftrl import dp_fedavg
from multi_epoch_dp_matrix_factorization.dp_ftrl import training_loop
from multi_epoch_dp_matrix_factorization.dp_ftrl.datasets import stackoverflow_word_prediction
from multi_epoch_dp_matrix_factorization.dp_ftrl.models import stackoverflow_models
from multi_epoch_dp_matrix_factorization.multiple_participations import contrib_matrix_builders
from utils import keras_metrics

IRRELEVANT_FLAGS = frozenset(iter(flags.FLAGS))

_RUN_NAME = flags.DEFINE_string(
    'run_name',
    'stackoverflow',
    (
        'The name of this run (work unit). Will be'
        'append to  --root_output_dir to separate experiment results.'
    ),
)

flags.DEFINE_string(
    'run_tags',
    '',
    (
        'Unused "hyperparameter" that can be used to tag runs with additional '
        'strings that are useful in filtering and organizing results.'
    ),
)
_ROOT_OUTPUT_DIR = flags.DEFINE_string(
    'root_output_dir',
    '/tmp/dpftrl/stackoverflow',
    'Root directory for writing experiment output.',
)
_ROUNDS_PER_CHECKPOINT = flags.DEFINE_integer(
    'rounds_per_checkpoint', 100, 'How often to checkpoint the global model.'
)
_ROUNDS_PER_EVAL = flags.DEFINE_integer(
    'rounds_per_eval',
    20,
    'How often to evaluate the global model on the validation dataset.',
)

# Training
_CLIENTS_PER_ROUND = flags.DEFINE_integer(
    'clients_per_round', 100, 'How many clients to sample per round.'
)
_CLIENT_EPOCHS_PER_ROUND = flags.DEFINE_integer(
    'client_epochs_per_round',
    1,
    'Number of epochs in the client to take per round.',
)
_CLIENT_BATCH_SIZE = flags.DEFINE_integer(
    'client_batch_size', 16, 'Batch size used on the client.'
)
_TOTAL_ROUNDS = flags.DEFINE_integer(
    'total_rounds', 10, 'Number of total training rounds.'
)
_TOTAL_EPOCHS = flags.DEFINE_integer(
    'total_epochs',
    1,
    'If not None, use shuffling of clients instead of random sampling.',
)
_RESHUFFLE_EACH_EPOCH = flags.DEFINE_boolean(
    'reshuffle_each_epoch',
    True,
    (
        'Requires --total_epochs >= 1. If set, reshuffle mapping of clients '
        'to rounds on each epoch.'
    ),
)
_CLIENT_SELECTION_SEED = flags.DEFINE_integer(
    'client_selection_seed',
    random.getrandbits(32),
    'Random seed for client selection.',
)

# Optimizer
_CLIENT_OPTIMIZER = flags.DEFINE_enum(
    'client_optimizer', 'sgd', ['sgd'], 'Client optimzier'
)
_CLIENT_LR = flags.DEFINE_float('client_lr', 0.02, 'Client learning rate.')
_SERVER_LR = flags.DEFINE_float('server_lr', 1.0, 'Server learning rate.')
_SERVER_OPTIMIZER_LR_COOLDOWN = flags.DEFINE_boolean(
    'server_optimizer_lr_cooldown',
    False,
    (
        'If True, apply a hard-coded learning-rate cooldown schedule '
        'equal to the ones used in (some) matrix factorizations.'
    ),
)
_SERVER_MOMENTUM = flags.DEFINE_float(
    'server_momentum', 0.9, 'Server momentum for SGDM.'
)

# Differential privacy
_CLIP_NORM = flags.DEFINE_float('clip_norm', 1.0, 'Clip L2 norm.')
_NOISE_MULTIPLIER = flags.DEFINE_float(
    'noise_multiplier', 0.01, 'Noise multiplier for DP algorithm.'
)

_AGGREGATOR_METHOD = flags.DEFINE_enum(
    'aggregator_method',
    'tree_aggregation',
    list(aggregator_builder.AGGREGATION_METHODS),
    'Enum indicating the aggregator method to use.',
)

_LR_MOMENTUM_MATRIX_NAME = flags.DEFINE_string(
    'lr_momentum_matrix_name',
    None,
    (
        'Name of the mechanism (and partial path to stored matrix) '
        'for --aggregator_method=lr_momentum_matrix'
    ),
)

_ZERO_LARGE_UPDATES = flags.DEFINE_boolean(
    'zero_large_updates',
    True,
    (
        'Whether or not to zero updates with L_infinity norm larger than the '
        '10*FLAGS.clip_norm.'
    ),
)

# Data
_SEQUENCE_LENGTH = flags.DEFINE_integer(
    'sequence_length', 20, 'Max sequence length to use.'
)
_MAX_ELEMENTS_PER_USER = flags.DEFINE_integer(
    'max_elements_per_user',
    256,
    'Max number of training sentences to use per user.',
)
_NUM_VALIDATION_EXAMPLES = flags.DEFINE_integer(
    'num_validation_examples',
    10000,
    'Number of examples to use from test set for per-round validation.',
)

# Model
_VOCAB_SIZE = flags.DEFINE_integer('vocab_size', 10000, 'Size of vocab to use.')
_NUM_OOV_BUCKETS = flags.DEFINE_integer(
    'num_oov_buckets', 1, 'Number of out of vocabulary buckets.'
)
_EMBEDDING_SIZE = flags.DEFINE_integer(
    'embedding_size', 96, 'Dimension of word embedding to use.'
)
_LATENT_SIZE = flags.DEFINE_integer(
    'latent_size', 670, 'Dimension of latent size to use in recurrent cell'
)
_NUM_LAYERS = flags.DEFINE_integer(
    'num_layers', 1, 'Number of stacked recurrent layers to use.'
)
_SHARED_EMBEDDING = flags.DEFINE_boolean(
    'shared_embedding',
    False,
    'Boolean indicating whether to tie input and output embeddings.',
)

_USE_SYNTHETIC_DATA = flags.DEFINE_boolean(
    'use_synthetic_data', False, 'Use synthetic data (for testing)'
)

HPARAM_FLAGS = [f for f in flags.FLAGS if f not in IRRELEVANT_FLAGS]
FLAGS = flags.FLAGS


def _get_stackoverflow_metrics(vocab_size, num_oov_buckets):
  """Metrics for stackoverflow dataset."""
  special_tokens = stackoverflow_word_prediction.get_special_tokens(
      vocab_size, num_oov_buckets
  )
  pad_token = special_tokens.pad
  oov_tokens = special_tokens.oov
  eos_token = special_tokens.eos
  return [
      keras_metrics.MaskedCategoricalAccuracy(
          name='accuracy_with_oov', masked_tokens=[pad_token]
      ),
      keras_metrics.MaskedCategoricalAccuracy(
          name='accuracy_no_oov', masked_tokens=[pad_token] + oov_tokens
      ),
      keras_metrics.MaskedCategoricalAccuracy(
          name='accuracy_no_oov_or_eos',
          masked_tokens=[pad_token, eos_token] + oov_tokens,
      ),
  ]


def _preprocess_stackoverflow(
    vocab_size,
    num_oov_buckets,
    sequence_length,
    num_validation_examples,
    client_batch_size,
    client_epochs_per_round,
    max_elements_per_user,
):
  """Prepare stackoverflow dataset."""
  if _USE_SYNTHETIC_DATA.value:
    d = tff.simulation.datasets.stackoverflow.get_synthetic()
    train_clientdata, test_clientdata = d, d
    vocab_dict = (
        tff.simulation.datasets.stackoverflow.get_synthetic_word_counts()
    )
    dataset_vocab = list(vocab_dict.keys())[:vocab_size]
  else:
    train_clientdata, _, test_clientdata = (
        tff.simulation.datasets.stackoverflow.load_data()
    )
    dataset_vocab = stackoverflow_word_prediction.create_vocab(vocab_size)

  base_test_dataset = test_clientdata.create_tf_dataset_from_all_clients(seed=0)
  preprocess_val_and_test = stackoverflow_word_prediction.create_preprocess_fn(
      vocab=dataset_vocab,
      num_oov_buckets=num_oov_buckets,
      client_batch_size=128,
      client_epochs_per_round=1,
      max_sequence_length=sequence_length,
      max_elements_per_client=-1,
      max_shuffle_buffer_size=1,
  )
  test_set = preprocess_val_and_test(
      base_test_dataset.skip(num_validation_examples)
  )
  validation_set = preprocess_val_and_test(
      base_test_dataset.take(num_validation_examples)
  )

  train_dataset_preprocess_comp = (
      stackoverflow_word_prediction.create_preprocess_fn(
          vocab=dataset_vocab,
          num_oov_buckets=num_oov_buckets,
          client_batch_size=client_batch_size,
          client_epochs_per_round=client_epochs_per_round,
          max_sequence_length=sequence_length,
          max_elements_per_client=max_elements_per_user,
          max_shuffle_buffer_size=max_elements_per_user,
      )
  )

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


def _get_lr_schedule(
    n: int, base_learning_rate: float
) -> tf.keras.optimizers.schedules.LearningRateSchedule:
  """Returns a LearningRateSchedule with a fixed cooldown."""

  # Hard-coded for now based on previous experiments,
  # should be in sync with factorize_multi_epoch_prefix_sum
  cooldown_period = n // 4
  cooldown_target = 0.05
  lr = np.ones(n)
  lr[-cooldown_period:] = np.linspace(1.0, cooldown_target, num=cooldown_period)
  lr *= base_learning_rate

  class FixedSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, lr_schedule):
      self._lr = tf.constant(lr_schedule)

    @tf.function
    def __call__(self, step):
      return self._lr[tf.cast(step, tf.int32)]

  return FixedSchedule(lr)


def _build_tff_learning_model_and_process(input_spec, test_metrics):
  """Build `tff.learning` iterative process."""

  def tff_model_fn():
    keras_model = stackoverflow_models.create_recurrent_model(
        vocab_size=_VOCAB_SIZE.value,
        embedding_size=_EMBEDDING_SIZE.value,
        latent_size=_LATENT_SIZE.value,
        num_layers=_NUM_LAYERS.value,
        shared_embedding=_SHARED_EMBEDDING.value,
    )
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    return tff.learning.from_keras_model(
        keras_model=keras_model, input_spec=input_spec, loss=loss
    )

  def evaluate_fn(model_weights, dataset):
    keras_model = stackoverflow_models.create_recurrent_model(
        vocab_size=_VOCAB_SIZE.value,
        embedding_size=_EMBEDDING_SIZE.value,
        latent_size=_LATENT_SIZE.value,
        num_layers=_NUM_LAYERS.value,
        shared_embedding=_SHARED_EMBEDDING.value,
    )
    model_weights.assign_weights_to(keras_model)
    metrics = dp_fedavg.keras_evaluate(keras_model, dataset, test_metrics)
    return collections.OrderedDict(
        (test_metric.name, metric.numpy())
        for test_metric, metric in zip(test_metrics, metrics)
    )

  client_optimizer_fn = functools.partial(
      _client_optimizer_fn,
      name=_CLIENT_OPTIMIZER.value,
      learning_rate=_CLIENT_LR.value,
  )

  if (
      _TOTAL_EPOCHS.value == 6
      and not _RESHUFFLE_EACH_EPOCH.value
      and _TOTAL_ROUNDS.value in (2048, 2052)
  ):
    # This special-cases the experiments for the planned
    # experiments for the ICLR submission. It could be generalized
    # if needed.
    logging.info(
        'Found %s epochs and %s rounds with fixed shuffling, '
        'checking that matrix mechanisms have sensitivity 1 under '
        'the fixed participation pattern.'
    )
    contrib_matrix = contrib_matrix_builders.epoch_participation_matrix(
        n=2052, num_epochs=6
    )

    def verify_sensitivity_fn(h_matrix):
      n = h_matrix.shape[1]
      assert n == _TOTAL_ROUNDS.value
      assert n <= contrib_matrix.shape[0]
      contrib_matrix_n = contrib_matrix[:n, :]
      contrib_norms = np.linalg.norm(h_matrix @ contrib_matrix_n, axis=0)
      sensitivity = np.max(contrib_norms)
      if sensitivity > 1 + 1e-6:
        raise ValueError(
            f'Expected sensitivity <= 1.0, but calculated {sensitivity}'
        )

  else:
    verify_sensitivity_fn = None

  aggregator_factory = aggregator_builder.build_aggregator(
      aggregator_method=_AGGREGATOR_METHOD.value,
      model_fn=tff_model_fn,
      clip_norm=_CLIP_NORM.value,
      noise_multiplier=_NOISE_MULTIPLIER.value,
      clients_per_round=_CLIENTS_PER_ROUND.value,
      num_rounds=_TOTAL_ROUNDS.value,
      noise_seed=None,
      momentum=_SERVER_MOMENTUM.value,
      lr_momentum_matrix_name=_LR_MOMENTUM_MATRIX_NAME.value,
      verify_sensitivity_fn=verify_sensitivity_fn,
  )

  if _ZERO_LARGE_UPDATES.value:
    # Note that on-device processing is applied in the reverse order
    # of the code here, so zeroing will happen first on device.
    # Thus, the measurements from the measured_query will be after zeroing
    # but before clipping.
    # N.B. Not this is actually doing L_infinity
    # clipping, which we might not want. But keeping this
    # behavior for now for consistency with ICLR submission.
    logging.info('Configuring aggregator to zero large updates.')
    aggregator_factory = tff.aggregators.zeroing_factory(
        100.0 * _CLIP_NORM.value, aggregator_factory
    )

  def server_optimizer_fn():
    if _AGGREGATOR_METHOD.value in [
        'opt_momentum_matrix',
        'lr_momentum_matrix',
    ]:
      # If we directly factorize the momentum matrix, the momentum portion
      # of the update is already handled directly in the aggregator,
      # so we disable it in the server optimizer.
      server_optimizer_momentum_value = 0
    else:
      server_optimizer_momentum_value = _SERVER_MOMENTUM.value

    if _SERVER_OPTIMIZER_LR_COOLDOWN.value:
      if _AGGREGATOR_METHOD.value == 'lr_momentum_matrix':
        raise ValueError(
            'This codepath would combine the learning rate schedule from the '
            'matrix with post-processing learning rate cooldown in the server '
            'optimizer. If this is really what you want, this check can be '
            'removed.'
        )
      server_learning_rate = _get_lr_schedule(
          n=_TOTAL_ROUNDS.value, base_learning_rate=_SERVER_LR.value
      )
    else:
      server_learning_rate = _SERVER_LR.value

    return tf.keras.optimizers.legacy.SGD(
        learning_rate=server_learning_rate,
        momentum=server_optimizer_momentum_value,
        nesterov=False,
    )

  iterative_process = tff.learning.algorithms.build_unweighted_fed_avg(
      tff_model_fn,
      client_optimizer_fn,
      server_optimizer_fn,
      model_aggregator=aggregator_factory,
      use_experimental_simulation_loop=True,
  )

  return iterative_process, evaluate_fn


def train_and_eval():
  """Train and evaluate StackOver NWP task."""
  logging.info('Show FLAGS for debugging:')
  for f in HPARAM_FLAGS:
    logging.info('%s=%s', f, FLAGS[f].value)

  hparam_dict = collections.OrderedDict(
      [(name, FLAGS[name].value) for name in HPARAM_FLAGS]
  )

  train_dataset_computation, train_set, validation_set, test_set = (
      _preprocess_stackoverflow(
          _VOCAB_SIZE.value,
          _NUM_OOV_BUCKETS.value,
          _SEQUENCE_LENGTH.value,
          _NUM_VALIDATION_EXAMPLES.value,
          _CLIENT_BATCH_SIZE.value,
          _CLIENT_EPOCHS_PER_ROUND.value,
          _MAX_ELEMENTS_PER_USER.value,
      )
  )

  input_spec = train_dataset_computation.type_signature.result.element
  stackoverflow_metrics = _get_stackoverflow_metrics(
      _VOCAB_SIZE.value, _NUM_OOV_BUCKETS.value
  )
  iterative_process, evaluate_fn = _build_tff_learning_model_and_process(
      input_spec, stackoverflow_metrics
  )
  iterative_process = (
      tff.simulation.compose_dataset_computation_with_learning_process(
          dataset_computation=train_dataset_computation,
          process=iterative_process,
      )
  )

  if not _TOTAL_EPOCHS.value:  # None or 0
    rng = random.Random(_CLIENT_SELECTION_SEED.value)

    def client_dataset_ids_fn(round_num: int):
      del round_num
      return rng.sample(train_set.client_ids, _CLIENTS_PER_ROUND.value), 0

    logging.info(
        'Sampling %s clients independently each round for max %d rounds',
        _CLIENTS_PER_ROUND.value,
        _TOTAL_ROUNDS.value,
    )
    total_epochs = 0
  else:
    client_dataset_ids_fn = training_loop.ClientIDShuffler(
        _CLIENTS_PER_ROUND.value,
        train_set.client_ids,
        reshuffle_each_epoch=_RESHUFFLE_EACH_EPOCH.value,
        seed=_CLIENT_SELECTION_SEED.value,
    )
    logging.info(
        'Shuffle clients within epoch for max %d epochs and %d rounds',
        _TOTAL_EPOCHS.value,
        _TOTAL_ROUNDS.value,
    )
    total_epochs = _TOTAL_EPOCHS.value

  training_loop.run(
      iterative_process,
      client_dataset_ids_fn,
      validation_fn=functools.partial(evaluate_fn, dataset=validation_set),
      total_epochs=total_epochs,
      total_rounds=_TOTAL_ROUNDS.value,
      run_name=_RUN_NAME.value,
      train_eval_fn=None,
      test_fn=functools.partial(evaluate_fn, dataset=test_set),
      root_output_dir=_ROOT_OUTPUT_DIR.value,
      hparam_dict=hparam_dict,
      rounds_per_eval=_ROUNDS_PER_EVAL.value,
      rounds_per_checkpoint=_ROUNDS_PER_CHECKPOINT.value,
      rounds_per_train_eval=2000,
  )


def main(argv):
  if len(argv) > 1:
    raise app.UsageError(
        'Expected no command-line arguments, got: {}'.format(argv)
    )

  train_and_eval()


if __name__ == '__main__':
  app.run(main)
