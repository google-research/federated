# Copyright 2020, Google LLC.
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
from typing import Callable

import tensorflow as tf
import tensorflow_federated as tff

from optimization.shared import keras_metrics
from reconstruction.shared import federated_trainer_utils
from reconstruction.stackoverflow import models
from reconstruction.stackoverflow import stackoverflow_dataset
from utils import training_loop
from utils.datasets import stackoverflow_word_prediction


def run_federated(
    iterative_process_builder: Callable[..., tff.templates.IterativeProcess],
    evaluation_computation_builder: Callable[..., tff.Computation],
    client_batch_size: int,
    clients_per_round: int,
    global_variables_only: bool,
    vocab_size: int = 10000,
    num_oov_buckets: int = 1,
    sequence_length: int = 20,
    max_elements_per_user: int = 1000,
    embedding_size: int = 96,
    latent_size: int = 670,
    num_layers: int = 1,
    total_rounds: int = 1500,
    experiment_name: str = 'federated_so_nwp',
    root_output_dir: str = '/tmp/fed_recon',
    split_dataset_strategy: str = federated_trainer_utils
    .SPLIT_STRATEGY_AGGREGATED,
    split_dataset_proportion: int = 2,
    compose_dataset_computation: bool = False,
    **kwargs):
  """Runs an iterative process on the Stack Overflow next word prediction task.

  This method will load and pre-process dataset and construct a model used for
  the task. It then uses `iterative_process_builder` to create an iterative
  process that it applies to the task, using
  `federated_research.utils.training_loop`.

  This model only sends updates for its embeddings corresponding to the most
  common words. Embeddings for out of vocabulary buckets are reconstructed on
  device at the beginning of each round, and destroyed at the end of these
  rounds.

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
      `loss_fn`, a `metrics_fn`, and a `client_weight_fn`, and returns a
      `tff.templates.IterativeProcess`. The `model_fn` must return a
      `reconstruction_model.ReconstructionModel`. See `federated_trainer.py` for
      an example.
    evaluation_computation_builder: A function that accepts a no-arg `model_fn`,
      a loss_fn`, and a `metrics_fn`, and returns a `tff.Computation` for
      federated reconstruction evaluation. The `model_fn` must return a
      `reconstruction_model.ReconstructionModel`. See `federated_trainer.py` for
      an example.
    client_batch_size: An integer representing the batch size used on clients.
    clients_per_round: An integer representing the number of clients
      participating in each round.
    global_variables_only: If True, the `ReconstructionModel` contains all model
      variables as global variables. This can be useful for baselines involving
      aggregating all variables.
    vocab_size: Integer dictating the number of most frequent words to use in
      the vocabulary.
    num_oov_buckets: The number of out-of-vocabulary buckets to use.
    sequence_length: The maximum number of words to take for each sequence.
    max_elements_per_user: The maximum number of elements processed for each
      client's dataset.
    embedding_size: The dimension of the word embedding layer.
    latent_size: The dimension of the latent units in the recurrent layers.
    num_layers: The number of stacked recurrent layers to use.
    total_rounds: The number of federated training rounds.
    experiment_name: The name of the experiment being run. This will be appended
      to the `root_output_dir` for purposes of writing outputs.
    root_output_dir: The name of the root output directory for writing
      experiment outputs.
    split_dataset_strategy: The method to use to split the data. Must be one of
      `skip`, in which case every `split_dataset_proportion` example is used for
      reconstruction, or `aggregated`, when the first
      1/`split_dataset_proportion` proportion of the examples is used for
      reconstruction.
    split_dataset_proportion: Parameter controlling how much of the data is used
      for reconstruction. If `split_dataset_proportion` is n, then 1 / n of the
      data is used for reconstruction.
    compose_dataset_computation: Whether to compose dataset computation with
      training and evaluation computations. If True, may speed up experiments by
      parallelizing dataset computations in multimachine setups. Not currently
      supported in OSS.
    **kwargs: Additional arguments configuring the training loop. For details on
      supported arguments, see `training_loop.py`.
  """

  loss_fn = functools.partial(
      tf.keras.losses.SparseCategoricalCrossentropy, from_logits=True)

  special_tokens = stackoverflow_word_prediction.get_special_tokens(
      vocab_size, num_oov_buckets)
  pad_token = special_tokens.pad
  oov_tokens = special_tokens.oov
  eos_token = special_tokens.eos

  def metrics_fn():
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

  train_clientdata, validation_clientdata, test_clientdata = (
      tff.simulation.datasets.stackoverflow.load_data())

  vocab = stackoverflow_word_prediction.create_vocab(vocab_size)
  dataset_preprocess_comp = stackoverflow_dataset.create_preprocess_fn(
      vocab=vocab,
      num_oov_buckets=num_oov_buckets,
      client_batch_size=client_batch_size,
      max_sequence_length=sequence_length,
      max_elements_per_client=max_elements_per_user,
      feature_dtypes=train_clientdata.element_type_structure,
      sort_by_date=True)

  input_spec = dataset_preprocess_comp.type_signature.result.element

  model_fn = functools.partial(
      models.create_recurrent_reconstruction_model,
      vocab_size=vocab_size,
      num_oov_buckets=num_oov_buckets,
      embedding_size=embedding_size,
      latent_size=latent_size,
      num_layers=num_layers,
      input_spec=input_spec,
      global_variables_only=global_variables_only)

  def client_weight_fn(local_outputs):
    # Num_tokens is a tensor with type int64[1], to use as a weight need
    # a float32 scalar.
    return tf.cast(tf.squeeze(local_outputs['num_tokens']), tf.float32)

  iterative_process = iterative_process_builder(
      model_fn,
      loss_fn=loss_fn,
      metrics_fn=metrics_fn,
      client_weight_fn=client_weight_fn,
      dataset_split_fn_builder=functools.partial(
          federated_trainer_utils.build_dataset_split_fn,
          split_dataset_strategy=split_dataset_strategy,
          split_dataset_proportion=split_dataset_proportion))

  base_eval_computation = evaluation_computation_builder(
      model_fn,
      loss_fn=loss_fn,
      metrics_fn=metrics_fn,
      dataset_split_fn_builder=functools.partial(
          federated_trainer_utils.build_dataset_split_fn,
          split_dataset_strategy=split_dataset_strategy,
          split_dataset_proportion=split_dataset_proportion))

  if compose_dataset_computation:
    # Compose dataset computations with client training and evaluation to avoid
    # linear cost of computing centrally. This changes the expected input of
    # the `IterativeProcess` and `tff.Computation` to be a list of client IDs
    # instead of datasets.
    training_process = (
        tff.simulation.compose_dataset_computation_with_iterative_process(
            dataset_preprocess_comp, iterative_process))
    training_process = (
        tff.simulation.compose_dataset_computation_with_iterative_process(
            train_clientdata.dataset_computation, training_process))
    training_process.get_model_weights = iterative_process.get_model_weights

    base_eval_computation = (
        tff.simulation.compose_dataset_computation_with_computation(
            dataset_preprocess_comp, base_eval_computation))
    val_computation = (
        tff.simulation.compose_dataset_computation_with_computation(
            validation_clientdata.dataset_computation, base_eval_computation))
    test_computation = (
        tff.simulation.compose_dataset_computation_with_computation(
            test_clientdata.dataset_computation, base_eval_computation))

    # Create client sampling functions for each of train/val/test.
    # We need to sample client IDs, not datasets, and we do not need to apply
    # `dataset_preprocess_comp` since this is applied as part of the training
    # process and evaluation computation.
    train_client_datasets_fn = federated_trainer_utils.build_list_sample_fn(
        train_clientdata.client_ids, size=clients_per_round, replace=False)
    val_client_datasets_fn = federated_trainer_utils.build_list_sample_fn(
        validation_clientdata.client_ids, size=clients_per_round, replace=False)
    test_client_datasets_fn = federated_trainer_utils.build_list_sample_fn(
        test_clientdata.client_ids, size=clients_per_round, replace=False)
  else:
    training_process = iterative_process
    val_computation = base_eval_computation
    test_computation = base_eval_computation
    # Apply dataset computations.
    train_clientdata = train_clientdata.preprocess(dataset_preprocess_comp)
    validation_clientdata = validation_clientdata.preprocess(
        dataset_preprocess_comp)
    test_clientdata = test_clientdata.preprocess(dataset_preprocess_comp)

    # Create client sampling functions for each of train/val/test.
    train_client_datasets_fn = tff.simulation.build_uniform_client_sampling_fn(
        train_clientdata, clients_per_round=clients_per_round)
    val_client_datasets_fn = tff.simulation.build_uniform_client_sampling_fn(
        validation_clientdata, clients_per_round=clients_per_round)
    test_client_datasets_fn = tff.simulation.build_uniform_client_sampling_fn(
        test_clientdata, clients_per_round=clients_per_round)

  # Create final evaluation functions to pass to `training_loop`.
  val_fn = federated_trainer_utils.build_eval_fn(
      evaluation_computation=val_computation,
      client_datasets_fn=val_client_datasets_fn,
      get_model=training_process.get_model_weights)
  test_fn = federated_trainer_utils.build_eval_fn(
      evaluation_computation=test_computation,
      client_datasets_fn=test_client_datasets_fn,
      get_model=training_process.get_model_weights)
  test_fn = functools.partial(test_fn, round_num=0)

  training_loop.run(
      iterative_process=training_process,
      client_datasets_fn=train_client_datasets_fn,
      validation_fn=val_fn,
      test_fn=test_fn,
      total_rounds=total_rounds,
      experiment_name=experiment_name,
      root_output_dir=root_output_dir,
      **kwargs)
