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
"""Utilities for coordinate-wise aggregation."""

import tensorflow_federated as tff


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
