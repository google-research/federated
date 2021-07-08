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
"""TFF Federated Stack Overflow next word prediction library with secrets.

Adds secret phrases to training data, and reports exposure using extrapolation
technique as described in Carlini et al., 2019
https://arxiv.org/pdf/1802.08232.pdf.
"""

import collections
import functools
from typing import List, Tuple

import numpy as np

import tensorflow as tf
import tensorflow_federated as tff

from differential_privacy import secret_sharer
from optimization.tasks import training_specs
from utils import keras_metrics
from utils.datasets import stackoverflow_word_prediction
from utils.models import stackoverflow_models


def configure_training(
    task_spec: training_specs.TaskSpec,
    secret_len: int,
    secret_group_info: List[Tuple[int, float]],
    num_secrets_per_group: int,
    num_reference_secrets: int,
    secret_seed: int = 0,
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

  Secret phrases are inserted into the training data, and exposure of secrets
  according to extrapolation technique are reported as a test metric.

  Args:
    task_spec: A `TaskSpec` class for creating federated training tasks.
    secret_len: The number of tokens in each secret.
    secret_group_info: Each secret has a number of clients (the number of
      clients that share the secret) and an insertion probability (the
      probability that a sentence from one of those clients will be replaced by
      the secret). Typically we want to have several secrets with the same
      values for those parameters so we can make general conclusions about how
      an algorithm memorizes secrets with those characteristics. Thus we have
      "secret groups", or sets of `num_secrets_per_group` secrets that share
      those two parameters. The secret_group_info arg is a list of (int, float)
      tuples specifying, for each secret group, the number of clients and the
      insertion probability for secrets in that group. For example, if
      secret_group_info is [(1, 0.1), (1, 0.5), (10, 0.1)] and
      num_secrets_per_group is 2, there will be six (different) secrets total.
      Two will have a single associated client each, and each sentence of those
      clients will be selected for insertion with probability 0.1. Two will have
      a single associated client each and each sentence of those clients will be
      selected for insertion with probability 0.5. And two will have ten
      (different) associated clients each and each sentence of those clients
      will be selected for insertion with probability 0.1. Secrets are inserted
      permanently into the data, so the same sentences (if any) will be replaced
      by some client's secret on every round a client participates.
    num_secrets_per_group: The number of distinct secrets for each group from
      secret_group_info.
    num_reference_secrets: The number of reference secrets to use for estimating
      exposure using the 'extrapolation' technique of Carlini et al., 2019
      https://arxiv.org/pdf/1802.08232.pdf.
    secret_seed: The random seed to use for generating secret insertions.
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

  num_train_secrets = len(secret_group_info) * num_secrets_per_group
  word_counts = tff.simulation.datasets.stackoverflow.load_word_counts(
      vocab_size)
  secrets = secret_sharer.generate_secrets(
      word_counts, secret_len, num_train_secrets + num_reference_secrets)
  expanded_secret_group_info = []
  for config in secret_group_info:
    expanded_secret_group_info.extend([config] * num_secrets_per_group)
  train_secrets = collections.OrderedDict(
      zip(secrets[:num_train_secrets], expanded_secret_group_info))

  train_clientdata, _, _ = tff.simulation.datasets.stackoverflow.load_data()
  train_clientdata = secret_sharer.stackoverflow_with_secrets(
      train_clientdata, train_secrets, secret_seed)

  # TODO(b/161914546): consider moving evaluation to use
  # `tff.learning.build_federated_evaluation` to get metrics over client
  # distributions, as well as the example weight means from this centralized
  # evaluation.
  _, validation_dataset, test_dataset = stackoverflow_word_prediction.get_centralized_datasets(
      vocab_size=vocab_size,
      max_sequence_length=sequence_length,
      num_validation_examples=num_validation_examples,
      num_oov_buckets=num_oov_buckets)

  vocab = stackoverflow_word_prediction.create_vocab(vocab_size)
  train_preprocess_fn = stackoverflow_word_prediction.create_preprocess_fn(
      vocab=vocab,
      num_oov_buckets=num_oov_buckets,
      client_batch_size=task_spec.client_batch_size,
      client_epochs_per_round=task_spec.client_epochs_per_round,
      max_sequence_length=sequence_length,
      max_elements_per_client=max_elements_per_user)
  train_clientdata = train_clientdata.preprocess(train_preprocess_fn)
  input_spec = train_clientdata.element_type_structure

  def tff_model_fn() -> tff.learning.Model:
    return tff.learning.from_keras_model(
        keras_model=model_builder(),
        input_spec=input_spec,
        loss=loss_builder(),
        metrics=metrics_builder())

  iterative_process = task_spec.iterative_process_builder(tff_model_fn)
  training_process = tff.simulation.compose_dataset_computation_with_iterative_process(
      train_clientdata.dataset_computation, iterative_process)
  client_ids_fn = functools.partial(
      tff.simulation.build_uniform_sampling_fn(
          train_clientdata.client_ids,
          replace=False,
          random_seed=task_spec.client_datasets_random_seed),
      size=task_spec.clients_per_round)
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
    """Computes standard eval metrics and adds exposure of all secrets."""
    model_weights = iterative_process.get_model_weights(state)
    test_metric_dict = evaluate_fn(model_weights, [test_dataset])
    prediction_model = model_builder()
    model_weights.assign_weights_to(prediction_model)
    prediction_model.compile(loss=tf.keras.losses.CategoricalCrossentropy())
    to_ids_fn = stackoverflow_word_prediction.build_to_ids_fn(
        vocab=vocab,
        max_sequence_length=secret_len,
        num_oov_buckets=num_oov_buckets)
    cce = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE)

    def get_perplexity(secret):
      ids = to_ids_fn({'tokens': tf.convert_to_tensor(secret)})
      prediction = prediction_model.predict(ids[:-1])
      return np.mean(cce(ids[1:], prediction))

    exposures = secret_sharer.compute_exposure(
        secrets=secrets[:num_train_secrets],
        reference_secrets=secrets[num_train_secrets:],
        get_perplexity=get_perplexity)
    for i, exposure in enumerate(exposures):
      group_info = secret_group_info[i // num_secrets_per_group]
      j = i % num_secrets_per_group
      metric_name = f'exposure_{group_info[0]}_{group_info[1]}_{j}'
      test_metric_dict[metric_name] = exposure

    return test_metric_dict

  return training_specs.RunnerSpec(
      iterative_process=training_process,
      client_datasets_fn=client_sampling_fn,
      validation_fn=validation_fn,
      test_fn=test_fn)
