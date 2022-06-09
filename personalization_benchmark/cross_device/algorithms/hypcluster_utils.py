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
"""Utilities for building the HypCluster algorithm."""

from typing import Callable, List

import tensorflow as tf
import tensorflow_federated as tff


def build_coordinate_finalizer(
    base_finalizer: tff.learning.templates.FinalizerProcess,
    num_coordinates: int) -> tff.learning.templates.FinalizerProcess:
  """Applies a finalizer across multiple coordinates.

  Args:
    base_finalizer : A `tff.learning.templates.FinalizerProcess` to apply to
      each coordinate.
    num_coordinates: An integer representing the number of coordinates.

  Returns:
    A `tff.learning.templates.FinalizerProcess` that applies `base_finalizer`
    to each of its `num_coordinates` coordinates.
  """
  weights_type = base_finalizer.next.type_signature.parameter[1].member
  update_type = base_finalizer.next.type_signature.parameter[2].member

  weights_list_type = tff.StructWithPythonType(
      [weights_type for _ in range(num_coordinates)], container_type=list)
  update_list_type = tff.StructWithPythonType(
      [update_type for _ in range(num_coordinates)], container_type=list)

  @tff.federated_computation()
  def init_fn():
    state = [base_finalizer.initialize() for _ in range(num_coordinates)]
    return tff.federated_zip(state)

  state_list_type = init_fn.type_signature.result

  @tff.federated_computation(state_list_type,
                             tff.type_at_server(weights_list_type),
                             tff.type_at_server(update_list_type))
  def next_fn(state_list, weights_list, update_list):
    output_list = [
        base_finalizer.next(state, weights, update)
        for state, weights, update in zip(state_list, weights_list, update_list)
    ]
    updated_state = tff.federated_zip([output.state for output in output_list])
    updated_weights = tff.federated_zip(
        [output.result for output in output_list])
    measurements = tff.federated_zip(
        [output.measurements for output in output_list])
    return tff.templates.MeasuredProcessOutput(
        state=updated_state, result=updated_weights, measurements=measurements)

  return tff.learning.templates.FinalizerProcess(init_fn, next_fn)


def build_coordinate_aggregator(
    base_process: tff.templates.AggregationProcess,
    num_coordinates: int) -> tff.templates.AggregationProcess:
  """Applies an aggregator across multiple coordinates.

  Args:
    base_process: A `tff.templates.AggregationProcess` to apply to each
      coordinate.
    num_coordinates: An integer representing the number of coordinates.

  Returns:
    A `tff.templates.AggregationProcess` that applies `base_process` to each of
    its `num_coordinates` coordinates.
  """
  if len(base_process.next.type_signature.parameter) != 3:
    raise ValueError('The base_process must be a weighted aggregation process '
                     'such that base_process.next expects 3 input arguments.')

  value_type = base_process.next.type_signature.parameter[1].member
  weight_type = base_process.next.type_signature.parameter[2].member

  value_list_type = tff.StructWithPythonType(
      [value_type for _ in range(num_coordinates)], container_type=list)
  weight_list_type = tff.StructWithPythonType(
      [weight_type for _ in range(num_coordinates)], container_type=list)

  @tff.federated_computation()
  def init_fn():
    state = [base_process.initialize() for _ in range(num_coordinates)]
    return tff.federated_zip(state)

  state_list_type = init_fn.type_signature.result

  @tff.federated_computation(state_list_type,
                             tff.type_at_clients(value_list_type),
                             tff.type_at_clients(weight_list_type))
  def next_fn(state_list, value_list, weight_list):
    output_list = [
        base_process.next(state, value, weight)
        for state, value, weight in zip(state_list, value_list, weight_list)
    ]
    updated_state = tff.federated_zip([output.state for output in output_list])
    result = tff.federated_zip([output.result for output in output_list])
    measurements = tff.federated_zip(
        [output.measurements for output in output_list])
    return tff.templates.MeasuredProcessOutput(
        state=updated_state, result=result, measurements=measurements)

  return tff.templates.AggregationProcess(init_fn, next_fn)


def build_gather_fn(list_element_type: tff.Type, num_indices: int):
  """Builds a gather-type computation for a list of TFF structures."""
  list_type = tff.StructWithPythonType(
      [list_element_type for _ in range(num_indices)], container_type=list)

  @tff.tf_computation(list_type, tf.int32)
  def gather_fn(list_of_structures, index):
    nested_structs = [tf.nest.flatten(x) for x in list_of_structures]
    grouped_tensors = [list(a) for a in zip(*nested_structs)]
    selected_tensors = [tf.gather(a, index) for a in grouped_tensors]
    return tf.nest.pack_sequence_as(list_of_structures[0], selected_tensors)

  return gather_fn


def build_scatter_fn(value_type: tff.Type, num_indices: int) -> tff.Computation:
  """Builds a computation that scatters a value and weight to a specific index.

  The returned TFF computation takes in three arguments `value`, `index`,
  `weight`, and returns two lists of size `num_indices`: a one-hot value list
  and a one-hot weight list. The value list has `value` at `index` and a zero
  structure at the other indices. Similarly, the weight list has `weight` at
  `index`, and a zero float at the other indices.

  Args:
    value_type: A `tff.Type` of the value to be scattered.
    num_indices: The length of the scatterred value list and weight list.

  Returns:
    A TFF computation.
  """

  def scale_nested_structure(structure, scale):
    return tf.nest.map_structure(lambda x: x * tf.cast(scale, x.dtype),
                                 structure)

  @tff.tf_computation(value_type, tf.int32, tf.float32)
  def scatter_fn(value, index, weight):
    one_hot_index = [tf.math.equal(i, index) for i in range(num_indices)]
    one_hot_value = [scale_nested_structure(value, j) for j in one_hot_index]
    one_hot_weight = [tf.cast(j, tf.float32) * weight for j in one_hot_index]
    return one_hot_value, one_hot_weight

  return scatter_fn


@tf.function
def multi_model_eval(models, eval_weights, data):
  """Evaluate multiple `tff.learning.Model`s on a single dataset."""
  for model, updated_weights in zip(models, eval_weights):
    model_weights = tff.learning.ModelWeights.from_model(model)
    tf.nest.map_structure(lambda a, b: a.assign(b), model_weights,
                          updated_weights)

  def reduce_fn(state, batch):
    for model in models:
      model.forward_pass(batch, training=False)
    return state

  data.reduce(0, reduce_fn)
  return [model.report_local_unfinalized_metrics() for model in models]


@tf.function
def select_best_model(model_outputs):
  # This works only when keras loss metric has two states: total and count; so
  # the finalized loss is computed as total/count.
  losses = [a['loss'][0] / a['loss'][1] for a in model_outputs]
  min_index = tf.math.argmin(losses, axis=0, output_type=tf.int32)
  return min_index


def build_find_best_model_fn(
    model_fn: Callable[[], tff.learning.Model], num_clusters: int,
    model_weights_type: tff.Type, data_type: tff.Type
) -> Callable[[List[tff.learning.ModelWeights], tf.data.Dataset], int]:
  """Creates a tff.Computation for selecting the best model for a dataset."""
  list_weights_type = tff.StructWithPythonType(
      [model_weights_type for _ in range(num_clusters)], container_type=list)

  @tff.tf_computation(list_weights_type, data_type)
  def find_best_model(weights, dataset):
    eval_models = [model_fn() for _ in range(num_clusters)]
    eval_models_outputs = multi_model_eval(eval_models, weights, dataset)
    return select_best_model(eval_models_outputs)

  return find_best_model
