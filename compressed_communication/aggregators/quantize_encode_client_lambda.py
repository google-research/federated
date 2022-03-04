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
import tensorflow_compression as tfc
import tensorflow_federated as tff

from compressed_communication.aggregators import elias_gamma_encode
from compressed_communication.aggregators.utils import quantize_utils


_SEED_TYPE = tff.TensorType(tf.int64, [2])


class QuantizeEncodeClientLambdaFactory(
    tff.aggregators.UnweightedAggregationFactory):
  r"""Aggregator that quantizes and encodes input tensor elements over training.

  The created tff.templates.AggregationProcess quantizes input tensor elements
  to levels defined by the `step_size` as `y = round(x / step_size)`. The
  larger the `step_size`, the fewer quantization levels and lower resolution.
  Higher quantization resolution is achieved with a smaller `step_size`.
  Every round, each client additionally quantizes according to step sizes in
  `step_size_options` and reports the step size within those options that
  minimizes `D + \lambda * R` for the given `lambda`. The `step_size` is
  adjusted over training every round to the option that receives the highest
  number of votes across clients. By default uniform rounding is used, but
  stochastic and dithered `rounding_type` can be specified. After
  quantization, the values are sent to the `elias_gamma_encode` inner
  aggregation process.

  The process returns round number and updated `step_size` (state), the
  dequantized tensor (result) and a dictionary of `step_size` mapped to the
  step size used in that round, `step_size_options` mapped to the step size
  options each client voted on and `step_size_vote_counts` mapped to a list of
  vote counts across clients ordered by `step_size_options` (measurements).
  """

  def __init__(self, lagrange_multiplier, step_size, step_size_options,
               rounding_type="uniform"):
    r"""Initializer for QuantizeEncodeClientLambdaFactory.

    Defines the target \lambda and quantization step size options for clients.

    Args:
      lagrange_multiplier: Float that is the target \lambda to trade off rate
        and distortion.
      step_size: Float that parametrizes the initial quantization level.
      step_size_options: List of floats that each client uses to quantize its
        value every round to vote on the optimal `step_size` for given
        \lambda, and set the `step_size` to the winning option.
      rounding_type: Optional string indicating what type of rounding to
        apply, one of ["uniform", "stochastic", "dithered"].
    """
    self._lagrange_multiplier = lagrange_multiplier
    self._step_size = step_size
    self._step_size_options = step_size_options

    if rounding_type == "uniform":
      self._quantize_fn = quantize_utils.uniform_quantize
      self._dequantize_fn = quantize_utils.uniform_dequantize
      self._generate_noise = lambda seed, shape: ()
    elif rounding_type == "stochastic":
      self._quantize_fn = quantize_utils.stochastic_quantize
      self._dequantize_fn = quantize_utils.uniform_dequantize
      self._generate_noise = lambda seed, shape: ()
    elif rounding_type == "dithered":
      self._quantize_fn = quantize_utils.dithered_quantize
      self._dequantize_fn = tf.function(quantize_utils.dithered_dequantize)
      self._generate_noise = quantize_utils.generate_noise
    else:
      raise ValueError("Expected `rounding_type` to be one one of "
                       "[\"uniform\", \"stochastic\", \"dithered\"], found "
                       f"{rounding_type}.")

    self._inner_agg_factory = elias_gamma_encode.EliasGammaEncodedSumFactory()

  def create(self, value_type):
    if not tff.types.is_structure_of_floats(
        value_type) or not value_type.is_tensor():
      raise ValueError("Expect value_type to be a float tensor, "
                       f"found {value_type}.")

    @tff.tf_computation(value_type, tf.float32)
    def quantize(value, step_size):
      seed = tf.cast(tf.stack([tf.timestamp() * 1e6, tf.timestamp() * 1e6]),
                     dtype=tf.int64)
      quantized_value = self._quantize_fn(value, step_size, seed)
      noise = self._generate_noise(seed, value.shape)
      return quantized_value, noise

    @tff.tf_computation(value_type)
    @tf.function
    def vote_step_size(value):
      seed = tf.cast(tf.stack([tf.timestamp() * 1e6, tf.timestamp() * 1e6]),
                     dtype=tf.int64)
      noise = self._generate_noise(seed, value.shape)
      objective = []

      for step_size in self._step_size_options:
        quantized_value = self._quantize_fn(value, step_size, seed)
        dequantized_value = self._dequantize_fn(quantized_value, step_size,
                                                noise)

        value_size = tf.size(quantized_value, out_type=tf.float32)
        distortion = tf.reduce_sum(
            tf.square(value - dequantized_value)) / value_size
        rate = tf.cast(
            elias_gamma_encode.get_bitstring_length(
                tfc.run_length_gamma_encode(quantized_value)),
            tf.float32) / value_size

        loss = distortion + self._lagrange_multiplier * rate
        objective.append(loss)

      return tf.one_hot(
          tf.argmin(objective), depth=len(objective), dtype=tf.int32)

    inner_agg_process = self._inner_agg_factory.create(
        quantize.type_signature.result[0])

    @tff.federated_computation()
    def init_fn():
      state = collections.OrderedDict(
          step_size=tff.federated_value(self._step_size, tff.SERVER),
          inner_state=inner_agg_process.initialize())
      return tff.federated_zip(state)

    @tff.tf_computation
    def dequantize(value, step_size, noise_sum):
      return self._dequantize_fn(value, step_size, noise_sum)

    @tff.federated_computation(init_fn.type_signature.result,
                               tff.type_at_clients(value_type))
    def next_fn(state, value):
      step_size = state["step_size"]
      inner_state = state["inner_state"]

      step_size_at_clients = tff.federated_broadcast(step_size)
      quantized_value, noise = tff.federated_map(
          quantize, (value, step_size_at_clients))
      noise_sum = tff.federated_sum(noise)

      inner_agg_output = inner_agg_process.next(inner_state, quantized_value)
      result = tff.federated_map(
          dequantize, (inner_agg_output.result, step_size, noise_sum))

      step_size_votes = tff.federated_map(vote_step_size, value)
      step_size_vote_counts = tff.federated_sum(step_size_votes)
      next_step_size = tff.federated_map(
          tff.tf_computation(
              lambda x: tf.gather(self._step_size_options, tf.argmax(x))),
          step_size_vote_counts)

      next_inner_state = inner_agg_output.state

      next_state = collections.OrderedDict(
          step_size=next_step_size,
          inner_state=next_inner_state)

      return tff.templates.MeasuredProcessOutput(
          state=tff.federated_zip(next_state),
          result=result,
          measurements=tff.federated_zip(
              collections.OrderedDict(
                  step_size=step_size,
                  step_size_options=tff.federated_value(
                      self._step_size_options, tff.SERVER),
                  step_size_vote_counts=step_size_vote_counts)))

    return tff.templates.AggregationProcess(init_fn, next_fn)
