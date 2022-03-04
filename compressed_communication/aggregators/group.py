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
"""A tff.aggregator for grouping input tensor components."""

import collections
import tensorflow_federated as tff


class GroupFactory(tff.aggregators.UnweightedAggregationFactory):
  """Aggregator grouping input tensor components.

  The `create` method of `GroupFactory` requires that `value_type` supports
  indexing. The created `tff.templates.AggregationProcess` groups the indices
  specified in grouped_indices, and applies an inner aggregation factory to each
  respective group.

  The process returns a dictionary mapping each group name to the state returned
  by the inner aggregation process of each group in state, and similarly a
  dictionary mapping each group name to the measurement collected by the inner
  aggregation process of each group in measurements. The result of inner
  aggregation processes are restructured into the original ungrouped structure.
  """

  def __init__(self, grouped_indices, inner_agg_factories):
    """Initializer for GroupFactory.

    Defines the tensor components to group and the inner aggregation factory
    to apply to each group.

    Args:
      grouped_indices: A dictionary that maps each group name to a list of
        indices of structure components that belong to that group.
      inner_agg_factories: A dictionary that maps each group name to an inner
        aggregation factory for processing that group.
    """
    self._groups = grouped_indices.keys()
    self._grouped_indices = grouped_indices
    if self._groups != inner_agg_factories.keys():
      raise ValueError("Group keys must match across `grouped_indices` and "
                       f"`inner_agg_factories`, found {self._groups} and "
                       f"{inner_agg_factories.keys()}.")
    self._inner_agg_factories = inner_agg_factories
    self._inner_agg_processes = collections.OrderedDict()

  def create(self, value_type):
    if not tff.types.is_structure_of_floats(value_type):
      raise ValueError("Expect value_type to be structure of "
                       f"float tensors, found {value_type}.")

    @tff.tf_computation(value_type)
    def group_impl(structure):
      """Groups components of the structure by grouped indices."""
      grouped_vectors = collections.OrderedDict()
      for group in self._groups:
        grouped_vectors[group] = [
            structure[i] for i in self._grouped_indices[group]
        ]
      return grouped_vectors

    for group in self._groups:
      inner_val_type = group_impl.type_signature.result[group]
      self._inner_agg_processes[group] = self._inner_agg_factories[
          group].create(inner_val_type)

    @tff.federated_computation()
    def init_fn():
      """Initializes the inner aggregation process for each group."""
      state = collections.OrderedDict()
      for group in self._groups:
        state[group] = self._inner_agg_processes[group].initialize()
      return tff.federated_zip(state)

    @tff.tf_computation(group_impl.type_signature.result)
    def ungroup_impl(grouped_vectors):
      """Applies the inverse of `group_impl` using the grouped indices."""
      components_vector = []
      ordered_indices = []
      for group in self._groups:
        ordered_indices.extend(self._grouped_indices[group])
        for component in grouped_vectors[group]:
          components_vector.append(component)
      ungrouped_structure = [0] * len(ordered_indices)
      for i in range(len(ordered_indices)):
        ungrouped_structure[ordered_indices[i]] = components_vector[i]
      return ungrouped_structure

    @tff.federated_computation(init_fn.type_signature.result,
                               tff.type_at_clients(
                                   group_impl.type_signature.result))
    def inner_agg_on_groups(state, grouped_vectors):
      """Calls `next` on each inner aggregation process."""
      inner_agg_states = collections.OrderedDict()
      inner_agg_results = collections.OrderedDict()
      inner_agg_measurements = collections.OrderedDict()
      for group in self._groups:
        outputs = self._inner_agg_processes[group].next(
            state[group], grouped_vectors[group])
        inner_agg_states[group] = outputs.state
        inner_agg_results[group] = outputs.result
        inner_agg_measurements[group] = outputs.measurements
      return collections.OrderedDict(states=inner_agg_states,
                                     results=inner_agg_results,
                                     measurements=inner_agg_measurements)

    @tff.federated_computation(init_fn.type_signature.result,
                               tff.type_at_clients(value_type))
    def next_fn(state, value):
      """Groups tensor components, applies inner aggregators, ungroups result."""
      grouped_vectors = tff.federated_map(group_impl, value)
      inner_agg_outputs = inner_agg_on_groups(state, grouped_vectors)

      ungrouped_result = tff.federated_map(ungroup_impl,
                                           inner_agg_outputs.results)

      return tff.templates.MeasuredProcessOutput(
          state=tff.federated_zip(inner_agg_outputs.states),
          result=ungrouped_result,
          measurements=tff.federated_zip(inner_agg_outputs.measurements))

    return tff.templates.AggregationProcess(init_fn, next_fn)
