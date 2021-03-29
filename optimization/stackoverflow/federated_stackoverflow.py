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
"""Federated Stack Overflow next word prediction library using TFF."""

import functools

import tensorflow as tf
import tensorflow_federated as tff

from optimization.shared import keras_metrics
from optimization.shared import training_specs
from utils.datasets import stackoverflow_word_prediction
from utils.models import stackoverflow_models


def configure_training(
    task_spec: training_specs.TaskSpec,
    vocab_size: int = 10000,
    num_oov_buckets: int = 1,
    sequence_length: int = 20,
    max_elements_per_user: int = 1000,
    num_validation_examples: int = 10000,
    embedding_size: int = 96,
    latent_size: int = 670,
    num_layers: int = 1,
    shared_embedding: bool = False) -> training_specs.RunnerSpec:
  """Configures training for Stack Overflow next-word prediction.

  This method will load and pre-process datasets and construct a model used for
  the task. It then uses `iterative_process_builder` to create an iterative
  process compatible with `federated_research.utils.training_loop`.

  Args:
    task_spec: A `TaskSpec` class for creating federated training tasks.
    vocab_size: Integer dictating the number of most frequent words to use in
      the vocabulary.
    num_oov_buckets: The number of out-of-vocabulary buckets to use.
    sequence_length: The maximum number of words to take for each sequence.
    max_elements_per_user: The maximum number of elements processed for each
      client's dataset.
    num_validation_examples: The number of test examples to use for validation.
    embedding_size: The dimension of the word embedding layer.
    latent_size: The dimension of the latent units in the recurrent layers.
    num_layers: The number of stacked recurrent layers to use.
    shared_embedding: Boolean indicating whether to tie input and output
      embeddings.

  Returns:
    A `RunnerSpec` containing attributes used for running the newly created
    federated task.
  """

  model_builder = functools.partial(
      stackoverflow_models.create_recurrent_model,
      vocab_size=vocab_size,
      num_oov_buckets=num_oov_buckets,
      embedding_size=embedding_size,
      latent_size=latent_size,
      num_layers=num_layers,
      shared_embedding=shared_embedding)

  loss_builder = functools.partial(
      tf.keras.losses.SparseCategoricalCrossentropy, from_logits=True)

  special_tokens = stackoverflow_word_prediction.get_special_tokens(
      vocab_size, num_oov_buckets)
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

  train_clientdata, _, _ = tff.simulation.datasets.stackoverflow.load_data()

  # TODO(b/161914546): consider moving evaluation to use
  # `tff.learning.build_federated_evaluation` to get metrics over client
  # distributions, as well as the example weight means from this centralized
  # evaluation.
  _, validation_dataset, test_dataset = stackoverflow_word_prediction.get_centralized_datasets(
      vocab_size=vocab_size,
      max_sequence_length=sequence_length,
      num_validation_examples=num_validation_examples,
      num_oov_buckets=num_oov_buckets)

  train_dataset_preprocess_comp = stackoverflow_word_prediction.create_preprocess_fn(
      vocab=stackoverflow_word_prediction.create_vocab(vocab_size),
      num_oov_buckets=num_oov_buckets,
      client_batch_size=task_spec.client_batch_size,
      client_epochs_per_round=task_spec.client_epochs_per_round,
      max_sequence_length=sequence_length,
      max_elements_per_client=max_elements_per_user)

  input_spec = train_dataset_preprocess_comp.type_signature.result.element

  def tff_model_fn() -> tff.learning.Model:
    return tff.learning.from_keras_model(
        keras_model=model_builder(),
        input_spec=input_spec,
        loss=loss_builder(),
        metrics=metrics_builder())

  iterative_process = task_spec.iterative_process_builder(tff_model_fn)

  @tff.tf_computation(tf.string)
  def train_dataset_computation(client_id):
    client_train_data = train_clientdata.dataset_computation(client_id)
    return train_dataset_preprocess_comp(client_train_data)

  training_process = tff.simulation.compose_dataset_computation_with_iterative_process(
      train_dataset_computation, iterative_process)
  client_ids_fn = tff.simulation.build_uniform_sampling_fn(
      train_clientdata.client_ids,
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
        iterative_process.get_model_weights(state), [validation_dataset])

  def test_fn(state):
    return evaluate_fn(
        iterative_process.get_model_weights(state),
        [validation_dataset.concatenate(test_dataset)])

  return training_specs.RunnerSpec(
      iterative_process=training_process,
      client_datasets_fn=client_sampling_fn,
      validation_fn=validation_fn,
      test_fn=test_fn)
