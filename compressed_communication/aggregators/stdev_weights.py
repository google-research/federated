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
"""A tff.aggregator for tracking the standard deviation of client model weights."""

import tensorflow as tf
import tensorflow_federated as tff


class StdevWeightsFactory(tff.aggregators.UnweightedAggregationFactory):
  r"""Aggregator reporting the standard deviation of client weights as a metric.

  The created tff.templates.AggregationProcess sums values placed at CLIENTS,
  and outputs the sum placed at SERVER.

  The process has empty state and returns the standard deviation over client
  values in measurements. For computing the result summation over client values,
  implementation delegates to the tff.federated_sum operator.

  The standard deviation returned in measurements is either a single value if
  value_type is a single tensor of weights, or a list of values - one for each
  layer - if the client value_type is a struct of weight tensors. Standard
  deviation is computed by taking the second moment of weights on each client,
  ie mean_{v \in values}(v**2), then taking a federated mean of these second
  moments on the server and federated square root.
  """

  def create(self, value_type):
    if not (tff.types.is_structure_of_floats(value_type) or
            (value_type.is_tensor() and value_type.dtype == tf.float32)):
      raise ValueError("Expect value_type to be float tensor or structure of "
                       f"float tensors, found {value_type}.")

    @tff.federated_computation()
    def init_fn():
      return tff.federated_value((), tff.SERVER)

    @tff.tf_computation(value_type)
    def compute_client_mean_second_moment(value):
      second_moment = tf.nest.map_structure(tf.math.square, value)
      client_mean_second_moment = tf.nest.map_structure(tf.math.reduce_mean,
                                                        second_moment)
      return client_mean_second_moment

    @tff.tf_computation(compute_client_mean_second_moment.type_signature.result)
    def compute_sqrt(mean_client_second_moments):
      return tf.math.sqrt(mean_client_second_moments)

    @tff.federated_computation(init_fn.type_signature.result,
                               tff.type_at_clients(value_type))
    def next_fn(state, value):
      summed_value = tff.federated_sum(value)

      client_second_moments = tff.federated_map(
          compute_client_mean_second_moment, value)
      mean_client_second_moments = tff.federated_mean(client_second_moments)
      server_stdev = tff.federated_map(compute_sqrt,
                                       mean_client_second_moments)

      return tff.templates.MeasuredProcessOutput(
          state=state, result=summed_value, measurements=server_stdev)

    return tff.templates.AggregationProcess(init_fn, next_fn)

