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
"""StackOverflow bandits task."""
from collections.abc import Callable
import enum
from typing import Any, Optional
from absl import logging

import tensorflow as tf
import tensorflow_federated as tff

from bandits.tasks import tag_prediction_preprocessing
from bandits.tasks import task_utils


DEFAULT_SHUFFLE_BUFFER_SIZE = 1000
DEFAULT_TAG_VOCAB_SIZE = 50
DEFAULT_WORD_VOCAB_SIZE = 10000


class ModelType(enum.Enum):
  LINEAR = enum.auto()


def get_model_types():
  return list(ModelType)


class VecNormLayer(tf.keras.layers.Layer):
  """A keras layer to normalize the vector."""

  def call(self, vectors: tf.Tensor) -> tf.Tensor:
    norms = tf.norm(vectors, ord='euclidean', axis=1, keepdims=True)
    return tf.math.divide_no_nan(vectors, norms)


def _get_keras_linear_model(
    *,
    input_size: int,
    output_size: int,
    normalize_input: bool = True,
    zero_init: bool = False,
) -> tf.keras.Model:
  """Returns a linear model for StackOverflow.

  Args:
    input_size: The input size of the linear model
    output_size: The output size of the linear model.
    normalize_input: Whether to normalize the vectorized input to have unit L2
      norm.
    zero_init: Whether to initialize the linear model to all zero.
  """
  layers = [tf.keras.Input(shape=(input_size,))]
  if normalize_input:
    layers += [VecNormLayer()]
  if zero_init:
    layers += [tf.keras.layers.Dense(output_size, kernel_initializer='zeros')]
  else:
    layers += [tf.keras.layers.Dense(output_size)]
  return tf.keras.models.Sequential(layers)


def create_stackoverflow_bandits_model_fn(
    bandits_data_spec: Any,
    *,
    input_size: int,
    output_size: int,
    model: ModelType = ModelType.LINEAR,
    loss: tf.keras.losses.Loss = task_utils.BanditsMSELoss(),
) -> Callable[[], tff.learning.models.VariableModel]:
  """Returns `model_fn` for bandits process in TFF simulation.

  Args:
    bandits_data_spec: The format of bandits data.
    input_size: The input size of the model.
    output_size: The output size of the model.
    model: The type of model, can be LeNet or a linear model.
    loss: The loss for the returned `model_fn`.
  """
  if isinstance(loss, task_utils.SupervisedCELoss):
    raise ValueError(
        'task_utils.SupervisedCELoss does not support multi-label '
        'stackoverflow tag prediction; use task_utils.MultiLabelCELoss'
    )
  if isinstance(loss, task_utils.SupervisedMSELoss):
    raise ValueError(
        'task_utils.SupervisedMSELoss does not support multi-label '
        'stackoverflow tag prediction; use task_utils.MultiLabelMSELoss'
    )

  def keras_model_fn():
    if model == ModelType.LINEAR:
      return _get_keras_linear_model(
          input_size=input_size, output_size=output_size
      )
    else:
      raise ValueError(f'Unknown StackOverflow model type: {model}')

  def model_fn():
    return tff.learning.from_keras_model(
        keras_model=keras_model_fn(),
        loss=loss,
        input_spec=bandits_data_spec,
        metrics=[task_utils.WrapRecall(top_k=5, name='recall_at_5')],
    )

  return model_fn


def create_stackoverflow_preprocessed_datasets(
    *,
    train_client_batch_size: int = 16,
    test_client_batch_size: int = 128,
    train_client_epochs_per_round: int = 1,
    train_shuffle_buffer_size: Optional[int] = DEFAULT_SHUFFLE_BUFFER_SIZE,
    train_client_max_elements: Optional[int] = None,
    word_vocab_size: int = DEFAULT_WORD_VOCAB_SIZE,
    tag_vocab_size: int = DEFAULT_TAG_VOCAB_SIZE,
    use_synthetic_data: bool = False,
    label_distribution_shift: bool = False,
    population_client_selection: Optional[str] = None,
) -> tff.simulation.baselines.BaselineTaskDatasets:
  """Returns preprocessed *supervised* client datasets in TFF simulation.

  The datasets will be used to generate bandits data for training and
  evaluation.

  Args:
    train_client_batch_size: The batch size for train clients.
    test_client_batch_size: The batch size for test clients.
    train_client_epochs_per_round: The number of epochs each train client should
      iterate over their local dataset, via `tf.data.Dataset.repeat`. Must be
      set to a positive integer.
    train_shuffle_buffer_size: An integer representing the shuffle buffer size
      (as in `tf.data.Dataset.shuffle`) for each train client's dataset. By
      default, this is set to the largest dataset size among all clients. If set
      to some integer less than or equal to 1, no shuffling occurs. If set to
      None, use `DEFAULT_SHUFFLE_BUFFER_SIZE`.
    train_client_max_elements: The maximum number of samples per client via
      `tf.data.Dataset.take`. If `None`, all the client data will be used.
    word_vocab_size: Integer dictating the number of most frequent words in the
      entire corpus to use for the task's vocabulary. If set to None, use
      `DEFAULT_WORD_VOCAB_SIZE`.
    tag_vocab_size: Integer dictating the number of most frequent tags in the
      entire corpus to use for the task's labels. If set to None, use
      `DEFAULT_TAG_VOCAB_SIZE`.
    use_synthetic_data: Use synthetic data for fast and robust unit tests.
    label_distribution_shift: Label distribution shift by masking 40% of the
      selected tags. For example, if we select top 50 tags, the 0/1 multi-label
      value for tag 30-49 will always be zero if
      `label_distribution_shift=True`.
    population_client_selection: Use a subset of clients to form the training
      population; can be useful for distribution shift settings. Should be in
      the format of "start_index-end_index" for [start_index, end_index) of all
      the sorted clients in a simulation dataset. For example, "0-1000" will
      select the first 1000 clients.
  """
  if word_vocab_size < 1:
    raise ValueError('word_vocab_size must be a positive integer')
  if tag_vocab_size < 1:
    raise ValueError('tag_vocab_size must be a positive integer')

  if use_synthetic_data:
    synthetic_data = tff.simulation.datasets.stackoverflow.get_synthetic()
    stackoverflow_train = synthetic_data
    stackoverflow_validation = synthetic_data
    stackoverflow_test = synthetic_data
    word_vocab_dict = (
        tff.simulation.datasets.stackoverflow.get_synthetic_word_counts()
    )
    tag_vocab_dict = (
        tff.simulation.datasets.stackoverflow.get_synthetic_tag_counts()
    )
  else:
    stackoverflow_train, stackoverflow_validation, stackoverflow_test = (
        tff.simulation.datasets.stackoverflow.load_data()
    )
    word_vocab_dict = tff.simulation.datasets.stackoverflow.load_word_counts(
        vocab_size=word_vocab_size
    )
    tag_vocab_dict = tff.simulation.datasets.stackoverflow.load_tag_counts()

  if population_client_selection is not None:
    total_population = len(stackoverflow_train.client_ids)
    stackoverflow_train = task_utils.clientdata_for_select_clients(
        stackoverflow_train, population_client_selection
    )
    logging.info(
        'Use %d of %d clients selected by %s',
        len(stackoverflow_train.client_ids),
        total_population,
        population_client_selection,
    )

  word_vocab = list(word_vocab_dict.keys())[:word_vocab_size]
  tag_vocab = list(tag_vocab_dict.keys())[:tag_vocab_size]

  train_client_spec = tff.simulation.baselines.ClientSpec(
      num_epochs=train_client_epochs_per_round,
      batch_size=train_client_batch_size,
      max_elements=train_client_max_elements,
      shuffle_buffer_size=train_shuffle_buffer_size,
  )
  eval_client_spec = tff.simulation.baselines.ClientSpec(
      num_epochs=1,
      batch_size=test_client_batch_size,
      max_elements=None,
      shuffle_buffer_size=1,
  )

  word_vocab_size = len(word_vocab)
  tag_vocab_size = len(tag_vocab)
  train_preprocess_fn = tag_prediction_preprocessing.create_preprocess_fn(
      train_client_spec,
      word_vocab,
      tag_vocab,
      label_distribution_shift=label_distribution_shift,
  )
  eval_preprocess_fn = tag_prediction_preprocessing.create_preprocess_fn(
      eval_client_spec, word_vocab, tag_vocab, label_distribution_shift=False
  )
  task_datasets = tff.simulation.baselines.BaselineTaskDatasets(
      train_data=stackoverflow_train,
      test_data=stackoverflow_test,
      validation_data=stackoverflow_validation,
      train_preprocess_fn=train_preprocess_fn,
      eval_preprocess_fn=eval_preprocess_fn,
  )
  return task_datasets
