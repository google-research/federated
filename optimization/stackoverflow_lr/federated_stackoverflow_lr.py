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
"""Federated Stack Overflow tag prediction (via logistic regression) using TFF."""

import functools

import tensorflow as tf
import tensorflow_federated as tff

from optimization.shared import training_specs
from utils.datasets import stackoverflow_tag_prediction
from utils.models import stackoverflow_lr_models


def metrics_builder():
  """Returns a `list` of `tf.keras.metric.Metric` objects."""
  return [
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(top_k=5, name='recall_at_5'),
  ]


def configure_training(
    task_spec: training_specs.TaskSpec,
    vocab_tokens_size: int = 10000,
    vocab_tags_size: int = 500,
    max_elements_per_user: int = 1000,
    num_validation_examples: int = 10000) -> training_specs.RunnerSpec:
  """Configures training for the Stack Overflow tag prediction task.

  This tag prediction is performed via multi-class one-versus-rest logistic
  regression. This method will load and pre-process datasets and construct a
  model used for the task. It then uses `iterative_process_builder` to create an
  iterative process compatible with `federated_research.utils.training_loop`.

  Args:
    task_spec: A `TaskSpec` class for creating federated training tasks.
    vocab_tokens_size: Integer dictating the number of most frequent words to
      use in the vocabulary.
    vocab_tags_size: Integer dictating the number of most frequent tags to use
      in the label creation.
    max_elements_per_user: The maximum number of elements processed for each
      client's dataset.
    num_validation_examples: The number of test examples to use for validation.

  Returns:
    A `RunnerSpec` containing attributes used for running the newly created
    federated task.
  """

  stackoverflow_train, _, _ = tff.simulation.datasets.stackoverflow.load_data()

  _, stackoverflow_validation, stackoverflow_test = stackoverflow_tag_prediction.get_centralized_datasets(
      train_batch_size=task_spec.client_batch_size,
      word_vocab_size=vocab_tokens_size,
      tag_vocab_size=vocab_tags_size,
      num_validation_examples=num_validation_examples)

  word_vocab = stackoverflow_tag_prediction.create_word_vocab(vocab_tokens_size)
  tag_vocab = stackoverflow_tag_prediction.create_tag_vocab(vocab_tags_size)

  train_preprocess_fn = stackoverflow_tag_prediction.create_preprocess_fn(
      word_vocab=word_vocab,
      tag_vocab=tag_vocab,
      client_batch_size=task_spec.client_batch_size,
      client_epochs_per_round=task_spec.client_epochs_per_round,
      max_elements_per_client=max_elements_per_user)
  input_spec = train_preprocess_fn.type_signature.result.element

  model_builder = functools.partial(
      stackoverflow_lr_models.create_logistic_model,
      vocab_tokens_size=vocab_tokens_size,
      vocab_tags_size=vocab_tags_size)

  loss_builder = functools.partial(
      tf.keras.losses.BinaryCrossentropy,
      from_logits=False,
      reduction=tf.keras.losses.Reduction.SUM)

  def tff_model_fn() -> tff.learning.Model:
    return tff.learning.from_keras_model(
        keras_model=model_builder(),
        input_spec=input_spec,
        loss=loss_builder(),
        metrics=metrics_builder())

  iterative_process = task_spec.iterative_process_builder(tff_model_fn)

  @tff.tf_computation(tf.string)
  def build_train_dataset_from_client_id(client_id):
    client_dataset = stackoverflow_train.dataset_computation(client_id)
    return train_preprocess_fn(client_dataset)

  training_process = tff.simulation.compose_dataset_computation_with_iterative_process(
      build_train_dataset_from_client_id, iterative_process)
  client_ids_fn = tff.simulation.build_uniform_sampling_fn(
      stackoverflow_train.client_ids,
      size=task_spec.clients_per_round,
      replace=False,
      random_seed=task_spec.client_datasets_random_seed)
  # We convert the output to a list (instead of an np.ndarray) so that it can
  # be used as input to the iterative process.
  client_sampling_fn = lambda x: list(client_ids_fn(x))

  training_process.get_model_weights = iterative_process.get_model_weights

  evaluate_fn = tff.learning.build_federated_evaluation(tff_model_fn)

  def validation_fn(state, round_num):
    del round_num
    return evaluate_fn(
        iterative_process.get_model_weights(state), [stackoverflow_validation])

  def test_fn(state):
    return evaluate_fn(
        iterative_process.get_model_weights(state),
        [stackoverflow_validation.concatenate(stackoverflow_test)])

  return training_specs.RunnerSpec(
      iterative_process=training_process,
      client_datasets_fn=client_sampling_fn,
      validation_fn=validation_fn,
      test_fn=test_fn)
