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
"""Utilities for coordinate-wise finalizers."""

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
