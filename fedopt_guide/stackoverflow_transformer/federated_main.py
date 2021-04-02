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
"""Federated Stack Overflow next word prediction library using TFF."""

import functools
from typing import Callable, Optional
from absl import logging

import tensorflow as tf
import tensorflow_federated as tff

from fedopt_guide import training_loop
from fedopt_guide.stackoverflow_transformer import transformer_models
from optimization.shared import keras_metrics
from utils.datasets import stackoverflow_word_prediction


def run_federated(iterative_process_builder: Callable[
    ..., tff.templates.IterativeProcess],
                  client_epochs_per_round: int,
                  client_batch_size: int,
                  clients_per_round: int,
                  max_elements_per_user: int,
                  total_rounds: int = 3000,
                  vocab_size: int = 10000,
                  num_oov_buckets: int = 1,
                  sequence_length: int = 20,
                  num_validation_examples: int = 10000,
                  dim_embed: int = 96,
                  dim_model: int = 512,
                  dim_hidden: int = 2048,
                  num_heads: int = 8,
                  num_layers: int = 1,
                  max_position_encoding: int = 1000,
                  dropout: float = 0.1,
                  client_datasets_random_seed: Optional[int] = None,
                  experiment_name: str = 'federated_stackoverflow',
                  root_output_dir: str = '/tmp/fedopt_guide',
                  max_val_test_batches: Optional[int] = None,
                  **kwargs) -> None:
  """Configures training for Stack Overflow next-word prediction.

  This method will load and pre-process dataset and construct a model used for
  the task. It then uses `iterative_process_builder` to create an iterative
  process that it applies to the task, using
  `federated_research/fedopt_guide/training_loop`.

  We assume that the iterative process has the following functional type
  signatures:

    *   `initialize`: `( -> S@SERVER)` where `S` represents the server state.
    *   `next`: `<S@SERVER, {B*}@CLIENTS> -> <S@SERVER, T@SERVER>` where `S`
        represents the server state, `{B*}` represents the client datasets,
        and `T` represents a python `Mapping` object.

  The iterative process must also have a callable attribute `get_model_weights`
  that takes as input the state of the iterative process, and returns a
  `tff.learning.ModelWeights` object.

  Args:
    iterative_process_builder: A function that accepts a no-arg `model_fn`, a
      `client_weight_fn` and returns a `tff.templates.IterativeProcess`. The
      `model_fn` must return a `tff.learning.Model`.
    client_epochs_per_round: An integer representing the number of epochs of
      training performed per client in each training round.
    client_batch_size: An integer representing the batch size used on clients.
    clients_per_round: An integer representing the number of clients
      participating in each round.
    max_elements_per_user: The maximum number of elements processed for each
      client's dataset. This has be to a positive value or -1 (which means that
      all elements are taken for training).
    total_rounds: The number of federated training rounds.
    vocab_size: Integer dictating the number of most frequent words to use in
      the vocabulary.
    num_oov_buckets: The number of out-of-vocabulary buckets to use.
    sequence_length: The maximum number of words to take for each sequence.
    num_validation_examples: The number of test examples to use for validation.
    dim_embed: An integer for the dimension of the token embeddings.
    dim_model: An integer for the dimension of features of MultiHeadAttention
      layers.
    dim_hidden: An integer for the dimension of hidden layers of the FFN.
    num_heads:  An integer for the number of attention heads.
    num_layers: An integer for the number of Transformer blocks.
    max_position_encoding: Maximum number of positions for position embeddings.
    dropout: Dropout rate.
    client_datasets_random_seed: An optional int used to seed which clients are
      sampled at each round. If `None`, no seed is used.
    experiment_name: The name of the experiment being run. This will be appended
      to the `root_output_dir` for purposes of writing outputs.
    root_output_dir: The name of the root output directory for writing
      experiment outputs.
    max_val_test_batches: If set to a positive integer, val and test datasets
      are capped to at most that many batches. If set to None or a nonpositive
      integer, the full datasets are used.
    **kwargs: Additional arguments configuring the training loop. For details on
      supported arguments, see
      `federated_research/fedopt_guide/training_utils.py`.

  Returns:
    A `RunnerSpec` containing attributes used for running the newly created
    federated task.
  """

  train_clientdata, _, _ = tff.simulation.datasets.stackoverflow.load_data()

  _, validation_dataset, test_dataset = stackoverflow_word_prediction.get_centralized_datasets(
      vocab_size=vocab_size,
      max_sequence_length=sequence_length,
      num_validation_examples=num_validation_examples,
      num_oov_buckets=num_oov_buckets)

  if max_val_test_batches and max_val_test_batches >= 1:
    validation_dataset = validation_dataset.take(max_val_test_batches)
    test_dataset = test_dataset.take(max_val_test_batches)

  model_builder = functools.partial(
      transformer_models.create_transformer_lm,
      vocab_size=vocab_size,
      num_oov_buckets=num_oov_buckets,
      dim_embed=dim_embed,
      dim_model=dim_model,
      dim_hidden=dim_hidden,
      num_heads=num_heads,
      num_layers=num_layers,
      max_position_encoding=max_position_encoding,
      dropout=dropout,
      name='stackoverflow-transformer')

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

  train_dataset_preprocess_comp = stackoverflow_word_prediction.create_preprocess_fn(
      vocab=stackoverflow_word_prediction.create_vocab(vocab_size),
      num_oov_buckets=num_oov_buckets,
      client_batch_size=client_batch_size,
      client_epochs_per_round=client_epochs_per_round,
      max_sequence_length=sequence_length,
      max_elements_per_client=max_elements_per_user)

  input_spec = train_dataset_preprocess_comp.type_signature.result.element

  def tff_model_fn() -> tff.learning.Model:
    return tff.learning.from_keras_model(
        keras_model=model_builder(),
        input_spec=input_spec,
        loss=loss_builder(),
        metrics=metrics_builder())

  def client_weight_fn(local_outputs):
    # Num_tokens is a tensor with type int64[1], to use as a weight need
    # a float32 scalar.
    return tf.cast(tf.squeeze(local_outputs['num_tokens']), tf.float32)

  iterative_process = iterative_process_builder(
      tff_model_fn, client_weight_fn=client_weight_fn)

  if hasattr(train_clientdata, 'dataset_computation'):

    @tff.tf_computation(tf.string)
    def train_dataset_computation(client_id):
      client_train_data = train_clientdata.dataset_computation(client_id)
      return train_dataset_preprocess_comp(client_train_data)

    training_process = tff.simulation.compose_dataset_computation_with_iterative_process(
        train_dataset_computation, iterative_process)
    client_ids_fn = tff.simulation.build_uniform_sampling_fn(
        train_clientdata.client_ids,
        size=clients_per_round,
        replace=False,
        random_seed=client_datasets_random_seed)
    # We convert the output to a list (instead of an np.ndarray) so that it can
    # be used as input to the iterative process.
    client_sampling_fn = lambda x: list(client_ids_fn(x))
  else:
    training_process = tff.simulation.compose_dataset_computation_with_iterative_process(
        train_dataset_preprocess_comp, iterative_process)
    client_sampling_fn = tff.simulation.build_uniform_client_sampling_fn(
        dataset=train_clientdata,
        clients_per_round=clients_per_round,
        random_seed=client_datasets_random_seed)

  training_process.get_model_weights = iterative_process.get_model_weights

  evaluate_fn = tff.learning.build_federated_evaluation(tff_model_fn)

  def validation_fn(model_weights, round_num):
    del round_num
    return evaluate_fn(model_weights, [validation_dataset])

  def test_fn(model_weights):
    return evaluate_fn(model_weights,
                       [validation_dataset.concatenate(test_dataset)])

  logging.info('Training model:')
  logging.info(model_builder().summary())

  training_loop.run(
      iterative_process=training_process,
      train_client_datasets_fn=client_sampling_fn,
      evaluation_fn=validation_fn,
      test_fn=test_fn,
      total_rounds=total_rounds,
      experiment_name=experiment_name,
      root_output_dir=root_output_dir,
      **kwargs)
