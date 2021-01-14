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
"""Federated EMNIST character recognition library using TFF."""

import functools
from typing import Optional

import tensorflow as tf
import tensorflow_federated as tff

from optimization.shared import training_specs
from utils import training_utils
from utils.datasets import emnist_dataset
from utils.models import emnist_models

EMNIST_MODELS = ['cnn', '2nn']
TOTAL_NUM_TRAIN_CLIENTS = 3400
TOTAL_NUM_TEST_CLIENTS = 3400


def configure_training(task_spec: training_specs.TaskSpec,
                       eval_spec: Optional[training_specs.EvalSpec] = None,
                       model: str = 'cnn') -> training_specs.RunnerSpec:
  """Configures training for the EMNIST character recognition task.

  This method will load and pre-process datasets and construct a model used for
  the task. It then uses `iterative_process_builder` to create an iterative
  process compatible with `federated_research.utils.training_loop`.

  Args:
    task_spec: A `TaskSpec` class for creating federated training tasks.
    eval_spec: An `EvalSpec` class for configuring federated evaluation. If set
      to None, centralized evaluation is used for validation and testing
      instead.
    model: A string specifying the model used for character recognition. Can be
      one of `cnn` and `2nn`, corresponding to a CNN model and a densely
      connected 2-layer model (respectively).

  Returns:
    A `RunnerSpec` containing attributes used for running the newly created
    federated task.
  """
  emnist_task = 'digit_recognition'

  emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(
      only_digits=False)

  train_preprocess_fn = emnist_dataset.create_preprocess_fn(
      num_epochs=task_spec.client_epochs_per_round,
      batch_size=task_spec.client_batch_size,
      emnist_task=emnist_task)

  input_spec = train_preprocess_fn.type_signature.result.element

  if model == 'cnn':
    model_builder = functools.partial(
        emnist_models.create_conv_dropout_model, only_digits=False)
  elif model == '2nn':
    model_builder = functools.partial(
        emnist_models.create_two_hidden_layer_model, only_digits=False)
  else:
    raise ValueError(
        'Cannot handle model flag [{!s}], must be one of {!s}.'.format(
            model, EMNIST_MODELS))

  loss_builder = tf.keras.losses.SparseCategoricalCrossentropy
  metrics_builder = lambda: [tf.keras.metrics.SparseCategoricalAccuracy()]

  def tff_model_fn() -> tff.learning.Model:
    return tff.learning.from_keras_model(
        keras_model=model_builder(),
        input_spec=input_spec,
        loss=loss_builder(),
        metrics=metrics_builder())

  iterative_process = task_spec.iterative_process_builder(tff_model_fn)

  clients_per_train_round = min(task_spec.clients_per_round,
                                TOTAL_NUM_TRAIN_CLIENTS)

  if hasattr(emnist_train, 'dataset_computation'):

    @tff.tf_computation(tf.string)
    def build_train_dataset_from_client_id(client_id):
      client_dataset = emnist_train.dataset_computation(client_id)
      return train_preprocess_fn(client_dataset)

    training_process = tff.simulation.compose_dataset_computation_with_iterative_process(
        build_train_dataset_from_client_id, iterative_process)
    client_ids_fn = training_utils.build_sample_fn(
        emnist_train.client_ids,
        size=clients_per_train_round,
        replace=False,
        random_seed=task_spec.sampling_random_seed)
    # We convert the output to a list (instead of an np.ndarray) so that it can
    # be used as input to the iterative process.
    client_sampling_fn = lambda x: list(client_ids_fn(x))

  else:
    training_process = tff.simulation.compose_dataset_computation_with_iterative_process(
        train_preprocess_fn, iterative_process)
    client_sampling_fn = training_utils.build_client_datasets_fn(
        dataset=emnist_train,
        clients_per_round=clients_per_train_round,
        random_seed=task_spec.sampling_random_seed)

  training_process.get_model_weights = iterative_process.get_model_weights

  if eval_spec:

    if eval_spec.clients_per_validation_round is None:
      clients_per_validation_round = TOTAL_NUM_TEST_CLIENTS
    else:
      clients_per_validation_round = min(eval_spec.clients_per_validation_round,
                                         TOTAL_NUM_TEST_CLIENTS)

    if eval_spec.clients_per_test_round is None:
      clients_per_test_round = TOTAL_NUM_TEST_CLIENTS
    else:
      clients_per_test_round = min(eval_spec.clients_per_test_round,
                                   TOTAL_NUM_TEST_CLIENTS)

    test_preprocess_fn = emnist_dataset.create_preprocess_fn(
        num_epochs=1,
        batch_size=eval_spec.client_batch_size,
        shuffle_buffer_size=1,
        emnist_task=emnist_task)
    emnist_test = emnist_test.preprocess(test_preprocess_fn)

    def eval_metrics_builder():
      return [
          tf.keras.metrics.SparseCategoricalCrossentropy(),
          tf.keras.metrics.SparseCategoricalAccuracy()
      ]

    federated_eval_fn = training_utils.build_federated_evaluate_fn(
        model_builder=model_builder, metrics_builder=eval_metrics_builder)

    validation_client_sampling_fn = training_utils.build_client_datasets_fn(
        emnist_test,
        clients_per_validation_round,
        random_seed=eval_spec.sampling_random_seed)
    test_client_sampling_fn = training_utils.build_client_datasets_fn(
        emnist_test,
        clients_per_test_round,
        random_seed=eval_spec.sampling_random_seed)

    def validation_fn(model_weights, round_num):
      validation_clients = validation_client_sampling_fn(round_num)
      return federated_eval_fn(model_weights, validation_clients)

    def test_fn(model_weights):
      # We fix the round number to get deterministic behavior
      test_round_num = 0
      test_clients = test_client_sampling_fn(test_round_num)
      return federated_eval_fn(model_weights, test_clients)

  else:
    _, central_emnist_test = emnist_dataset.get_centralized_datasets(
        only_digits=False, emnist_task=emnist_task)

    test_fn = training_utils.build_centralized_evaluate_fn(
        eval_dataset=central_emnist_test,
        model_builder=model_builder,
        loss_builder=loss_builder,
        metrics_builder=metrics_builder)

    validation_fn = lambda model_weights, round_num: test_fn(model_weights)

  return training_specs.RunnerSpec(
      iterative_process=training_process,
      client_datasets_fn=client_sampling_fn,
      validation_fn=validation_fn,
      test_fn=test_fn)
