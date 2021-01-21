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
"""Federated Shakespeare next character prediction library using TFF."""

import functools

import tensorflow as tf
import tensorflow_federated as tff

from optimization.shared import keras_metrics
from optimization.shared import training_specs
from utils import training_utils
from utils.datasets import shakespeare_dataset
from utils.models import shakespeare_models


# Vocabulary with OOV ID, zero for the padding, and BOS, EOS IDs.
VOCAB_SIZE = len(shakespeare_dataset.CHAR_VOCAB) + 4


def create_shakespeare_model(sequence_length):
  """Constructs a `tf.keras.Model` to train."""
  return shakespeare_models.create_recurrent_model(
      vocab_size=VOCAB_SIZE, sequence_length=sequence_length)


def metrics_builder():
  """Returns a `list` of `tf.keras.metric.Metric` objects."""
  pad_token, _, _, _ = shakespeare_dataset.get_special_tokens()

  return [
      keras_metrics.NumBatchesCounter(),
      keras_metrics.NumExamplesCounter(),
      keras_metrics.NumTokensCounter(masked_tokens=[pad_token]),
      keras_metrics.MaskedCategoricalAccuracy(masked_tokens=[pad_token]),
  ]


def configure_training(task_spec: training_specs.TaskSpec,
                       sequence_length: int = 80) -> training_specs.RunnerSpec:
  """Configures training for the Shakespeare next-character prediction task.

  This method will load and pre-process datasets and construct a model used for
  the task. It then uses `iterative_process_builder` to create an iterative
  process compatible with `federated_research.utils.training_loop`.

  Args:
    task_spec: A `TaskSpec` class for creating federated training tasks.
    sequence_length: An int specifying the length of the character sequences
      used for prediction.

  Returns:
    A `RunnerSpec` containing attributes used for running the newly created
    federated task.
  """

  shakespeare_train, _ = tff.simulation.datasets.shakespeare.load_data()
  _, shakespeare_test = shakespeare_dataset.get_centralized_datasets(
      sequence_length=sequence_length)

  train_preprocess_fn = shakespeare_dataset.create_preprocess_fn(
      num_epochs=task_spec.client_epochs_per_round,
      batch_size=task_spec.client_batch_size,
      sequence_length=sequence_length)
  input_spec = train_preprocess_fn.type_signature.result.element

  model_builder = functools.partial(
      create_shakespeare_model, sequence_length=sequence_length)
  loss_builder = functools.partial(
      tf.keras.losses.SparseCategoricalCrossentropy, from_logits=True)

  def client_weight_fn(local_outputs):
    # Num_tokens is a tensor with type int64[1], to use as a weight need
    # a float32 scalar.
    return tf.cast(tf.squeeze(local_outputs['num_tokens']), tf.float32)

  def tff_model_fn() -> tff.learning.Model:
    return tff.learning.from_keras_model(
        keras_model=model_builder(),
        input_spec=input_spec,
        loss=loss_builder(),
        metrics=metrics_builder())

  iterative_process = task_spec.iterative_process_builder(
      tff_model_fn, client_weight_fn=client_weight_fn)

  if hasattr(shakespeare_train, 'dataset_computation'):

    @tff.tf_computation(tf.string)
    def build_train_dataset_from_client_id(client_id):
      client_dataset = shakespeare_train.dataset_computation(client_id)
      return train_preprocess_fn(client_dataset)

    training_process = tff.simulation.compose_dataset_computation_with_iterative_process(
        build_train_dataset_from_client_id, iterative_process)
    client_ids_fn = training_utils.build_sample_fn(
        shakespeare_train.client_ids,
        size=task_spec.clients_per_round,
        replace=False,
        random_seed=task_spec.client_datasets_random_seed)
    # We convert the output to a list (instead of an np.ndarray) so that it can
    # be used as input to the iterative process.
    client_sampling_fn = lambda x: list(client_ids_fn(x))

  else:
    training_process = tff.simulation.compose_dataset_computation_with_iterative_process(
        train_preprocess_fn, iterative_process)
    client_sampling_fn = training_utils.build_client_datasets_fn(
        dataset=shakespeare_train,
        clients_per_round=task_spec.clients_per_round,
        random_seed=task_spec.client_datasets_random_seed)

  training_process.get_model_weights = iterative_process.get_model_weights

  test_fn = training_utils.build_centralized_evaluate_fn(
      eval_dataset=shakespeare_test,
      model_builder=model_builder,
      loss_builder=loss_builder,
      metrics_builder=metrics_builder)

  validation_fn = lambda model_weights, round_num: test_fn(model_weights)

  return training_specs.RunnerSpec(
      iterative_process=training_process,
      client_datasets_fn=client_sampling_fn,
      validation_fn=validation_fn,
      test_fn=test_fn)
