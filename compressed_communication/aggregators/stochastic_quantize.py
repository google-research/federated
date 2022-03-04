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
"""A tff.aggregator for stochastically quantizing client model weight updates."""

import tensorflow as tf
import tensorflow_federated as tff

from compressed_communication.aggregators.utils import quantize_utils


class StochasticQuantizeFactory(tff.aggregators.UnweightedAggregationFactory):
  """Aggregator that stochastically quantizes input tensor elements.

  The created tff.templates.AggregationProcess quantizes input tensor elements
  to levels defined by the `scale_factor` as `x = {ceil(x / scale_factor) with
  prob p, floor(x / scale_factor) with prob 1 - p} where p = (x / scale_factor)
  - floor(x / scale_factor)`. The larger the `scale_factor`, the fewer
  quantization levels and lower resolution. Higher quantization resolution is
  achieved with a smaller `scale_factor`. These quantized values are sent to an
  inner aggregation process.

  The process returns a state from the inner aggregation process (state), the
  dequantized tensor (result) and measurements from the inner aggregation
  process (measurements).
  """

  def __init__(self, scale_factor, inner_agg_factory):
    """Initializer for StochasicQuantizeFactory.

    Defines the scale of quantization and what inner aggregation should be
    applied next.

    Args:
      scale_factor: Float that parametrizes quantization levels.
      inner_agg_factory: UnweightedAggregationFactory to call after quantizing
        weights.
    """
    self.scale_factor = scale_factor
    self.inner_agg_factory = inner_agg_factory

  def create(self, value_type):
    if not tff.types.is_structure_of_floats(value_type):
      raise ValueError("Expect value_type to be structure of "
                       f"float tensors, found {value_type}.")

    @tff.tf_computation(value_type)
    def quantize(value):
      seed = tf.cast(tf.stack([tf.timestamp() * 1e6, tf.timestamp() * 1e6]),
                     dtype=tf.int64)
      scale = self.scale_factor
      return tf.nest.map_structure(
          lambda x: quantize_utils.stochastic_quantize(x, scale, seed), value)

    inner_agg_process = self.inner_agg_factory.create(
        quantize.type_signature.result)

    @tff.federated_computation()
    def init_fn():
      return inner_agg_process.initialize()

    @tff.tf_computation(quantize.type_signature.result)
    def dequantize(value):
      scale = self.scale_factor
      return tf.nest.map_structure(
          lambda x: quantize_utils.uniform_dequantize(x, scale, None), value)

    @tff.federated_computation(init_fn.type_signature.result,
                               tff.type_at_clients(value_type))
    def next_fn(state, value):
      quantized_value = tff.federated_map(quantize, value)
      inner_agg_output = inner_agg_process.next(state, quantized_value)
      result = tff.federated_map(dequantize, inner_agg_output.result)

      return tff.templates.MeasuredProcessOutput(
          state=inner_agg_output.state,
          result=result,
          measurements=inner_agg_output.measurements)

    return tff.templates.AggregationProcess(init_fn, next_fn)
