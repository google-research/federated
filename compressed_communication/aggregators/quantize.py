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
"""A tff.aggregator for quantizing client model weight updates."""

import collections

import tensorflow as tf
import tensorflow_federated as tff

from compressed_communication.aggregators.utils.quantize_utils import stochastic_quantize
from compressed_communication.aggregators.utils.quantize_utils import uniform_dequantize
from compressed_communication.aggregators.utils.quantize_utils import uniform_quantize


class QuantizeFactory(tff.aggregators.UnweightedAggregationFactory):
  """Aggregator that quantizes input tensor elements.

  The created tff.templates.AggregationProcess quantizes input tensor elements
  to levels defined by the `step_size` as `x = round(x / step_size)`. The
  larger the `step_size`, the fewer quantization levels and lower resolution.
  Higher quantization resolution is achieved with a smaller `step_size`.
  These quantized values are sent to an inner aggregation process.

  The `value_type` is expected to be a float tensor.

  The process returns a state from the inner aggregation process (state), the
  dequantized tensor (result) and measurements from the inner aggregation
  process (measurements).
  """

  def __init__(self, step_size, inner_agg_factory, rounding_type="uniform"):
    """Initializer for QuantizeFactory.

    Defines the step size between quantization levels and what inner aggregation
    should be applied next.

    Args:
      step_size: Float that parametrizes the quantization step size.
      inner_agg_factory: UnweightedAggregationFactory to call after quantizing
        weights.
      rounding_type: String specifying quantization rounding method, one of
        ["uniform", "stochastic"].
    """
    self.step_size = step_size
    self.inner_agg_factory = inner_agg_factory
    self.rounding_type = rounding_type

  def create(self, value_type):
    if not tff.types.is_structure_of_floats(
        value_type) or not value_type.is_tensor():
      raise ValueError("Expect value_type to be a float tensor, "
                       f"found {value_type}.")

    @tff.tf_computation
    def dequantize(value):
      return tf.nest.map_structure(
          lambda x: uniform_dequantize(x, self.step_size, None), value)

    @tff.tf_computation(value_type)
    def quantize(value):
      if self.rounding_type == "stochastic":
        seed = tf.cast(
            tf.stack([tf.timestamp() * 1e6,
                      tf.timestamp() * 1e6]),
            dtype=tf.int64)
        quantized_value = tf.nest.map_structure(
            lambda x: stochastic_quantize(x, self.step_size, seed), value)
      else:
        quantized_value = tf.nest.map_structure(
            lambda x: uniform_quantize(x, self.step_size, None), value)

      dequantized_value = dequantize(quantized_value)
      distortion = tf.reduce_mean(tf.square(value - dequantized_value))

      return quantized_value, distortion

    inner_agg_process = self.inner_agg_factory.create(
        quantize.type_signature.result[0])

    @tff.federated_computation()
    def init_fn():
      return inner_agg_process.initialize()

    @tff.federated_computation(init_fn.type_signature.result,
                               tff.type_at_clients(value_type))
    def next_fn(state, value):
      quantized_value, distortion = tff.federated_map(quantize, value)
      avg_distortion = tff.federated_mean(distortion)
      inner_agg_output = inner_agg_process.next(state, quantized_value)
      result = tff.federated_map(dequantize, inner_agg_output.result)

      return tff.templates.MeasuredProcessOutput(
          state=inner_agg_output.state,
          result=result,
          measurements=tff.federated_zip(
              collections.OrderedDict(
                  avg_distortion=avg_distortion,
                  inner_agg=inner_agg_output.measurements)))

    return tff.templates.AggregationProcess(init_fn, next_fn)
