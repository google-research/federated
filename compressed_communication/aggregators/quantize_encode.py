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

from compressed_communication.aggregators import elias_gamma_encode
from compressed_communication.aggregators.utils import quantize_utils


_SEED_TYPE = tff.TensorType(tf.int64, [2])


class QuantizeEncodeFactory(tff.aggregators.UnweightedAggregationFactory):
  """Aggregator that quantizes and encodes input tensor elements over training.

  The created tff.templates.AggregationProcess quantizes input tensor elements
  to levels defined by the `step_size` as `y = round(x / step_size)`. The
  `step_size` is adjusted over training according to a defined schedule. The
  larger the `step_size`, the fewer quantization levels and lower resolution.
  Higher quantization resolution is achieved with a smaller `step_size`.
  Optionally the `step_size` used on each client is scaled by some
  `normalization`. By default uniform quantization is used, but stochastic and
  dithered `type` can be specified. After quantization, the values are sent to
  the `elias_gamma_encode` inner aggregation process.

  The process returns round number and updated `step_size` (state), the
  dequantized tensor (result) and a dictionary of `avg_bitrate`,
  `avg_distortion`, `avg_sparsity` and `step_size` (measurements).
  """

  def __init__(self,
               initial_step_size,
               rounding_type="uniform",
               normalization_type="constant",
               schedule="fixed",
               schedule_hparam=None,
               min_step_size=0.01):
    """Initializer for QuantizeEncodeFactory.

    Defines the initial quantization step size, as well as what type of
    quantization should be applied and what normalization (if any) should be
    used to scale client updates.

    Args:
      initial_step_size: Float that parametrizes the quantization levels,
        equal to the initial quantization step size.
      rounding_type: Optional string indicating what type of rounding to
        apply, one of ["uniform", "stochastic", "dithered"].
      normalization_type: Optional string indicating what normalization function
        to use to scale the client updates, one of ["constant",
        "mean_magnitude", "max_magnitude", "dimensionless_norm"].
      schedule: Optional string indicating what schedule should be used to
        adjust the `initial_step_size`, one of ["fixed", "linear_decay",
        "exponential_decay", "step_decay"].
      schedule_hparam: Optional hyperparameter for the selected `schedule`
        function (None for "fixed", total_rounds for "linear_decay", exp for
        "exponential_decay", freq for "step_decay").
      min_step_size: Optional float specifying the minimum quantization step
        size to use when decaying `step_size` according to `schedule`.
    """
    self._initial_step_size = initial_step_size
    self._step_size = initial_step_size
    self._min_step_size = min_step_size

    if normalization_type == "constant":
      self._normalize_fn = lambda _: 1
    elif normalization_type == "mean_magnitude":
      self._normalize_fn = quantize_utils.mean_magnitude
    elif normalization_type == "max_magnitude":
      self._normalize_fn = quantize_utils.max_magnitude
    elif normalization_type == "dimensionless_norm":
      self._normalize_fn = quantize_utils.dimensionless_norm
    else:
      raise ValueError("Expected `normalization_type` to be one one of "
                       "[\"constant\", \"mean_magnitude\", \"max_magnitude\", "
                       f"\"dimensionless_norm\"], found {normalization_type}.")

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

    # pylint: disable=g-long-lambda
    if schedule == "fixed":
      self._schedule_fn = lambda _: self._step_size
    elif schedule == "linear_decay":
      self._schedule_fn = lambda round_num: quantize_utils.linear_decay(
          initial_step_size,
          min_step_size,
          round_num,
          total_rounds=schedule_hparam)
    elif schedule == "exponential_decay":
      self._schedule_fn = lambda round_num: quantize_utils.exponential_decay(
          initial_step_size, min_step_size, round_num, exp=schedule_hparam)
    elif schedule == "step_decay":
      self._schedule_fn = lambda round_num: quantize_utils.step_decay(
          initial_step_size, min_step_size, round_num, freq=schedule_hparam)
    else:
      raise ValueError("Expected `schedule` to be one one of [\"fixed\", "
                       "\"linear_decay\", \"exponential_decay\", "
                       f"\"step_decay\"], found {schedule}.")
    # pylint: enable=g-long-lambda
    self._schedule_fn = tf.function(self._schedule_fn)

    self._inner_agg_factory = elias_gamma_encode.EliasGammaEncodedSumFactory()

  def create(self, value_type):
    if not tff.types.is_structure_of_floats(
        value_type) or not value_type.is_tensor():
      raise ValueError("Expect value_type to be a float tensor, "
                       f"found {value_type}.")

    @tff.tf_computation(value_type, tf.float32)
    def quantize(value, step_size):
      seed = tf.cast(
          tf.stack([tf.timestamp() * 1e6,
                    tf.timestamp() * 1e6]),
          dtype=tf.int64)
      step_size = self._normalize_fn(value) * step_size
      quantized_value = self._quantize_fn(value, step_size, seed)
      noise = self._generate_noise(seed, value.shape)
      dequantized_value = self._dequantize_fn(quantized_value, step_size,
                                              noise)
      value_size = tf.size(quantized_value, out_type=tf.float32)
      distortion = tf.reduce_sum(
          tf.square(value - dequantized_value)) / value_size
      value_nonzero_ct = tf.math.count_nonzero(
          quantized_value, dtype=tf.float32)
      sparsity = (value_size - value_nonzero_ct) / value_size
      return quantized_value, noise, distortion, sparsity

    inner_agg_process = self._inner_agg_factory.create(
        quantize.type_signature.result[0])

    @tff.federated_computation()
    def init_fn():
      state = collections.OrderedDict(
          round_num=tff.federated_value(0.0, tff.SERVER),
          step_size=tff.federated_value(self._step_size, tff.SERVER),
          inner_state=inner_agg_process.initialize())
      return tff.federated_zip(state)

    @tff.tf_computation
    def dequantize(value, step_size, noise_sum):
      return self._dequantize_fn(value, step_size, noise_sum)

    @tff.federated_computation(init_fn.type_signature.result,
                               tff.type_at_clients(value_type))
    def next_fn(state, value):
      round_num = state["round_num"]
      step_size = state["step_size"]
      inner_state = state["inner_state"]

      step_size_at_clients = tff.federated_broadcast(step_size)
      quantized_value, noise, distortion, sparsity = tff.federated_map(
          quantize, (value, step_size_at_clients))
      noise_sum = tff.federated_sum(noise)
      avg_distortion = tff.federated_mean(distortion)
      avg_sparsity = tff.federated_mean(sparsity)

      inner_agg_output = inner_agg_process.next(inner_state, quantized_value)
      avg_bitrate = inner_agg_output.measurements.avg_bitrate
      result = tff.federated_map(
          dequantize, (inner_agg_output.result, step_size, noise_sum))

      next_round_num = tff.federated_map(
          tff.tf_computation(lambda x: x + 1.0), round_num)
      next_step_size = tff.federated_map(
          tff.tf_computation(self._schedule_fn), next_round_num)
      next_inner_state = inner_agg_output.state

      next_state = collections.OrderedDict(
          round_num=next_round_num,
          step_size=next_step_size,
          inner_state=next_inner_state)

      return tff.templates.MeasuredProcessOutput(
          state=tff.federated_zip(next_state),
          result=result,
          measurements=tff.federated_zip(
              collections.OrderedDict(
                  avg_bitrate=avg_bitrate,
                  avg_distortion=avg_distortion,
                  avg_sparsity=avg_sparsity,
                  step_size=step_size)))

    return tff.templates.AggregationProcess(init_fn, next_fn)
