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
"""A tff.aggregator for collecting statistics on client model weights."""

import collections

import tensorflow as tf
import tensorflow_federated as tff


class MinMaxMeanWeightsFactory(tff.aggregators.UnweightedAggregationFactory):
  """Aggregator reporting the min, max and mean of client weights as a metric.

  The created tff.templates.AggregationProcess sums values placed at CLIENTS,
  and outputs the sum placed at SERVER.

  The process has empty state and returns the minimum, maximum and mean client
  values in measurements. For computing the result summation over client values,
  implementation delegates to the tff.federated_sum operator.

  The value returned in measurements is an OrderedDict with "min", "max" and
  "mean" keys that map to either a single corresponding value if value_type is
  a single tensor of weights, or a list of corresponding values - one for each
  layer - if the client value_type is a struct of weight tensors.
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
    def compute_client_metrics(value):
      client_min = tf.nest.map_structure(tf.math.reduce_min, value)
      client_max = tf.nest.map_structure(tf.math.reduce_max, value)
      client_mean = tf.nest.map_structure(tf.math.reduce_mean, value)
      return collections.OrderedDict([("min", client_min),
                                      ("max", client_max),
                                      ("mean", client_mean)])

    @tff.federated_computation(init_fn.type_signature.result,
                               tff.type_at_clients(value_type))
    def next_fn(state, value):
      summed_value = tff.federated_sum(value)

      client_metrics = tff.federated_map(compute_client_metrics, value)
      mn = tff.aggregators.federated_min(client_metrics["min"])
      mx = tff.aggregators.federated_max(client_metrics["max"])
      mean = tff.federated_mean(client_metrics["mean"])
      server_metrics = tff.federated_zip(
          collections.OrderedDict([("min", mn), ("max", mx), ("mean", mean)]))

      return tff.templates.MeasuredProcessOutput(
          state=state, result=summed_value, measurements=server_metrics)

    return tff.templates.AggregationProcess(init_fn, next_fn)

