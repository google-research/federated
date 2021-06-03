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
"""Configures federated Stack Overflow tag prediction tasks."""

import functools

import tensorflow as tf
import tensorflow_federated as tff

from large_cohort import simulation_specs
from utils.datasets import stackoverflow_tag_prediction
from utils.models import stackoverflow_lr_models


TOKEN_VOCAB_SIZE = 10000
TAG_VOCAB_SIZE = 500


def get_model_spec(seed: int = 0) -> simulation_specs.ModelSpec:
  """Configures a `simulation_specs.ModelSpec` for Stack Overflow tag prediction."""
  keras_model_builder = functools.partial(
      stackoverflow_lr_models.create_logistic_model,
      vocab_tokens_size=TOKEN_VOCAB_SIZE,
      vocab_tags_size=TAG_VOCAB_SIZE,
      seed=seed)
  loss_builder = functools.partial(
      tf.keras.losses.BinaryCrossentropy,
      from_logits=False,
      reduction=tf.keras.losses.Reduction.SUM)

  def metrics_builder():
    return [
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(top_k=5, name='recall_at_5'),
    ]

  return simulation_specs.ModelSpec(
      keras_model_builder=keras_model_builder,
      loss_builder=loss_builder,
      metrics_builder=metrics_builder)


def get_data_spec(
    train_client_spec: simulation_specs.ClientSpec,
    eval_client_spec: simulation_specs.ClientSpec,
    use_synthetic_data: bool = False) -> simulation_specs.DataSpec:
  """Configures data for Stack Overflow tag prediction.

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

  word_vocab = stackoverflow_tag_prediction.create_word_vocab(TOKEN_VOCAB_SIZE)
  tag_vocab = stackoverflow_tag_prediction.create_tag_vocab(TAG_VOCAB_SIZE)

  train_preprocess_fn = stackoverflow_tag_prediction.create_preprocess_fn(
      client_epochs_per_round=train_client_spec.num_epochs,
      client_batch_size=train_client_spec.batch_size,
      max_elements_per_client=train_client_spec.max_elements,
      word_vocab=word_vocab,
      tag_vocab=tag_vocab)

  eval_preprocess_fn = stackoverflow_tag_prediction.create_preprocess_fn(
      client_epochs_per_round=eval_client_spec.num_epochs,
      client_batch_size=eval_client_spec.batch_size,
      max_elements_per_client=eval_client_spec.max_elements,
      word_vocab=word_vocab,
      tag_vocab=tag_vocab,
      max_shuffle_buffer_size=1)

  return simulation_specs.DataSpec(
      train_data=train_data,
      validation_data=validation_data,
      test_data=test_data,
      train_preprocess_fn=train_preprocess_fn,
      eval_preprocess_fn=eval_preprocess_fn)
