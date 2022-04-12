# Copyright 2022, Google LLC.
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
"""Implementation of different HypCluster evaluation algorithms."""

import collections
from typing import Callable

import tensorflow as tf
import tensorflow_federated as tff

from data_poor_fl import hypcluster_utils


def build_hypcluster_eval(
    model_fn: Callable[[], tff.learning.Model],
    num_clusters: int) -> tff.Computation:
  """Builds a computation for performing the standard HypCluster evaluation.

  The returned federated TFF computation has the following type signature:
  `list_of_model_weights@SERVER, dataset@CLIENTS -> eval_metrics@SERVER`. The
  client-side input is a single dataset, which is used to select the best model
  *and* evaluate the selected model.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`. This method
      must *not* capture TensorFlow tensors or variables and use them. The model
      must be constructed entirely from scratch on each invocation, returning
      the same pre-constructed model each call will result in an error.
    num_clusters: An integer specifying the number of clusters to use.

  Returns:
    A federated TFF computation that performs HypCluster evaluation.
  """

  with tf.Graph().as_default():
    # Wrap model construction in a graph to avoid polluting the global context
    # with variables created for this model.
    model = model_fn()
    model_weights_type = tff.learning.framework.weights_type_from_model(model)
    unfinalized_metrics_type = tff.types.type_from_tensors(
        model.report_local_unfinalized_metrics())
    metrics_aggregation_fn = tff.learning.metrics.sum_then_finalize(
        model.metric_finalizers(), unfinalized_metrics_type)

  data_type = tff.types.SequenceType(model.input_spec)
  metrics_gather_fn = hypcluster_utils.build_gather_fn(unfinalized_metrics_type,
                                                       num_clusters)
  list_weights_type = tff.StructWithPythonType(
      [model_weights_type for _ in range(num_clusters)], container_type=list)

  @tff.tf_computation(list_weights_type, data_type)
  def local_hypcluster_eval(model_weights, dataset):
    eval_models = [model_fn() for _ in range(num_clusters)]
    eval_models_outputs = hypcluster_utils.multi_model_eval(
        eval_models, model_weights, dataset)
    best_model_index = hypcluster_utils.select_best_model(eval_models_outputs)
    return metrics_gather_fn(eval_models_outputs, best_model_index)

  @tff.federated_computation(
      tff.type_at_server(list_weights_type), tff.type_at_clients(data_type))
  def hypcluster_eval(server_model_weights, client_datasets):
    client_model_weights = tff.federated_broadcast(server_model_weights)
    client_metrics = tff.federated_map(local_hypcluster_eval,
                                       (client_model_weights, client_datasets))
    eval_metrics = metrics_aggregation_fn(client_metrics)
    return tff.federated_zip(collections.OrderedDict(eval=eval_metrics))

  return hypcluster_eval


SELECTION_DATA_KEY = 'selection_data'
TEST_DATA_KEY = 'test_data'


@tf.function
def _get_metrics_for_select_and_test(model_list, weights_list, data):
  """Computes metrics on selection data and test data."""
  outputs_for_select = hypcluster_utils.multi_model_eval(
      model_list, weights_list, data[SELECTION_DATA_KEY])
  # Resets the metrics variables before evaluation. This is necessary, because
  # without resetting, the model's metrics will be from *all* previous
  # `forward_pass` calls, including the metrics on the selection data.
  for model in model_list:
    model.reset_metrics()
  outputs_for_metrics = hypcluster_utils.multi_model_eval(
      model_list, weights_list, data[TEST_DATA_KEY])
  return outputs_for_select, outputs_for_metrics


def build_hypcluster_eval_with_dataset_split(
    model_fn: Callable[[], tff.learning.Model],
    num_clusters: int) -> tff.Computation:
  """Builds a computation for performing a special HypCluster evaluation.

  This is different from the standard `build_hypcluster_evel`:
  1. The client-side input is an `OrderedDict` of two keys `SELECTION_DATA_KEY`
     and `TEST_DATA_KEY`, each mapped to a dataset. The selection data is used
     to select the best model; the test data is used to evaluate the selected
     model. Evaluation metrics on the test data is reported.
  2. This function adds additional metrics, including the performance of
     individual models, and the percentage that each model is selected.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`. This method
      must *not* capture TensorFlow tensors or variables and use them. The model
      must be constructed entirely from scratch on each invocation, returning
      the same pre-constructed model each call will result in an error.
    num_clusters: An integer specifying the number of clusters to use.

  Returns:
    A federated TFF computation that performs HypCluster evaluation.
  """
  with tf.Graph().as_default():
    # Wrap model construction in a graph to avoid polluting the global context
    # with variables created for this model.
    model = model_fn()
    model_weights_type = tff.learning.framework.weights_type_from_model(model)
    unfinalized_metrics_type = tff.types.type_from_tensors(
        model.report_local_unfinalized_metrics())
    metrics_aggregation_fn = tff.learning.metrics.sum_then_finalize(
        model.metric_finalizers(), unfinalized_metrics_type)
    client_data_type = tff.to_type(
        collections.OrderedDict([
            (SELECTION_DATA_KEY, tff.types.SequenceType(model.input_spec)),
            (TEST_DATA_KEY, tff.types.SequenceType(model.input_spec))
        ]))

  metrics_gather_fn = hypcluster_utils.build_gather_fn(unfinalized_metrics_type,
                                                       num_clusters)
  list_weights_type = tff.StructWithPythonType(
      [model_weights_type for _ in range(num_clusters)], container_type=list)

  @tff.tf_computation(list_weights_type, client_data_type)
  def local_hypcluster_eval(model_weights, dataset):
    eval_models = [model_fn() for _ in range(num_clusters)]
    eval_models_outputs_for_select, eval_models_outputs_for_metrics = (
        _get_metrics_for_select_and_test(eval_models, model_weights, dataset))
    best_model_index = hypcluster_utils.select_best_model(
        eval_models_outputs_for_select)
    local_metrics = collections.OrderedDict(
        best=metrics_gather_fn(eval_models_outputs_for_metrics,
                               best_model_index))
    for i in range(num_clusters):
      local_metrics[f'model_{i}'] = metrics_gather_fn(
          eval_models_outputs_for_metrics, i)
    for i in range(num_clusters):
      local_metrics[f'choose_{i}'] = tf.cast(
          tf.equal(best_model_index, i), tf.float32)
    return local_metrics

  @tff.federated_computation(
      tff.type_at_server(list_weights_type),
      tff.type_at_clients(client_data_type))
  def hypcluster_eval(server_model_weights, client_datasets):
    client_model_weights = tff.federated_broadcast(server_model_weights)
    client_metrics = tff.federated_map(local_hypcluster_eval,
                                       (client_model_weights, client_datasets))
    eval_metrics = collections.OrderedDict()
    metric_names = tff.structure.name_list(
        local_hypcluster_eval.type_signature.result)
    for name in metric_names:
      if 'choose' in name:
        eval_metrics[name] = tff.federated_mean(client_metrics[name])
      else:
        eval_metrics[name] = metrics_aggregation_fn(client_metrics[name])
    return tff.federated_zip(eval_metrics)

  return hypcluster_eval
