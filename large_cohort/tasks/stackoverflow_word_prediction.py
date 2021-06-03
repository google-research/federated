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
"""Configures federated Stack Overflow next word prediction tasks."""

import functools

import tensorflow as tf
import tensorflow_federated as tff

from large_cohort import simulation_specs
from utils import keras_metrics
from utils.datasets import stackoverflow_word_prediction
from utils.models import stackoverflow_models

# Dataset constants
VOCAB_SIZE = 10000
NUM_OOV_BUCKETS = 1
SEQUENCE_LENGTH = 20

# Model constants
EMBEDING_SIZE = 96
LATENT_SIZE = 670
NUM_LAYERS = 1
SHARED_EMBEDING = False


def get_model_spec(seed: int = 0) -> simulation_specs.ModelSpec:
  """Configures a model for Stack Overflow word prediction."""
  keras_model_builder = functools.partial(
      stackoverflow_models.create_recurrent_model,
      vocab_size=VOCAB_SIZE,
      num_oov_buckets=NUM_OOV_BUCKETS,
      embedding_size=EMBEDING_SIZE,
      latent_size=LATENT_SIZE,
      num_layers=NUM_LAYERS,
      shared_embedding=SHARED_EMBEDING,
      seed=seed)

  loss_builder = functools.partial(
      tf.keras.losses.SparseCategoricalCrossentropy, from_logits=True)

  special_tokens = stackoverflow_word_prediction.get_special_tokens(
      VOCAB_SIZE, NUM_OOV_BUCKETS)
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
        keras_metrics.NumTokensCounter(masked_tokens=[pad_token])
    ]

  return simulation_specs.ModelSpec(
      keras_model_builder=keras_model_builder,
      loss_builder=loss_builder,
      metrics_builder=metrics_builder)


def get_data_spec(
    train_client_spec: simulation_specs.ClientSpec,
    eval_client_spec: simulation_specs.ClientSpec,
    use_synthetic_data: bool = False) -> simulation_specs.DataSpec:
  """Configures data for Stack Overflow next-word prediction.

  Args:
    train_client_spec: A `simulation_specs.ClientSpec` used to configure
      training clients.
    eval_client_spec: A `simulation_specs.ClientSpec` used to configure
      evaluation clients.
    use_synthetic_data: A boolean indicating whether to use synthetic data.
      Suitable for testing purposes.

  Returns:
    A `simulation_specs.DataSpec`.
  """
  if use_synthetic_data:
    synthetic_data = tff.simulation.datasets.stackoverflow.get_synthetic()
    train_data = synthetic_data
    validation_data = synthetic_data
    test_data = synthetic_data
  else:
    train_data, validation_data, test_data = (
        tff.simulation.datasets.stackoverflow.load_data())

  vocab = stackoverflow_word_prediction.create_vocab(VOCAB_SIZE)

  train_preprocess_fn = stackoverflow_word_prediction.create_preprocess_fn(
      client_epochs_per_round=train_client_spec.num_epochs,
      client_batch_size=train_client_spec.batch_size,
      max_elements_per_client=train_client_spec.max_elements,
      vocab=vocab,
      num_oov_buckets=NUM_OOV_BUCKETS,
      max_sequence_length=SEQUENCE_LENGTH)

  eval_preprocess_fn = stackoverflow_word_prediction.create_preprocess_fn(
      client_epochs_per_round=eval_client_spec.num_epochs,
      client_batch_size=eval_client_spec.batch_size,
      max_elements_per_client=eval_client_spec.max_elements,
      vocab=vocab,
      num_oov_buckets=NUM_OOV_BUCKETS,
      max_sequence_length=SEQUENCE_LENGTH,
      max_shuffle_buffer_size=1)

  return simulation_specs.DataSpec(
      train_data=train_data,
      validation_data=validation_data,
      test_data=test_data,
      train_preprocess_fn=train_preprocess_fn,
      eval_preprocess_fn=eval_preprocess_fn)
