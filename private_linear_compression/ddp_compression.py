# Copyright 2022, Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A wrapper tff.aggregator for distributed DP with compression."""
import math

import tensorflow_federated as tff

from private_linear_compression import count_sketching
from private_linear_compression import count_sketching_utils


def _default_zeroing(
    inner_factory: tff.aggregators.AggregationFactory
) -> tff.aggregators.AggregationFactory:
  """The default adaptive zeroing wrapper."""

  # Adapts very quickly to a value somewhat higher than the highest values so
  # far seen.
  zeroing_norm = tff.aggregators.PrivateQuantileEstimationProcess.no_noise(
      initial_estimate=10.0,
      target_quantile=0.98,
      learning_rate=math.log(10.0),
      multiplier=2.0,
      increment=1.0,
      secure_estimation=True)
  secure_count_factory = tff.aggregators.SecureSumFactory(
      upper_bound_threshold=1, lower_bound_threshold=0)
  return tff.aggregators.zeroing_factory(
      zeroing_norm,
      inner_factory,
      zeroed_count_sum_factory=secure_count_factory)


def compressed_ddp_factory(
    noise_multiplier: float,
    expected_clients_per_round: int,
    compression_rate: float,
    bits: int = 20,
    decode_method: count_sketching_utils.DecodeMethod = count_sketching_utils
    .DecodeMethod.MEAN,
    num_repeats: int = 15,
    rotation_type: str = 'hd',
    zeroing: bool = False,
) -> tff.aggregators.UnweightedAggregationFactory:
  """Wrapper to combine DDP factory with linear compression via sketching.

  Note that the `create` method of this factory needs to be executed in TF
  eager mode because `distributed_dp.DistributedDpSumFactory` requires it.

  Args:
    noise_multiplier: A float specifying the noise multiplier (central noise
      stddev / L2 clip norm) for model updates. Note that this is with respect
      to the initial L2 clip norm, and the quantization procedure as part of the
      DDP algorithm may inflated the L2 sensitivity. The specified noise will be
      split into `expected_clients_per_round` noise shares to be added locally
      on the clients. A value of 1.0 or higher may be needed for strong privacy.
      Must be nonnegative. A value of 0.0 means no noise will be added.
    expected_clients_per_round: An integer specifying the expected number of
      clients to participate in this round. Must be a positive integer.
    compression_rate: Ratio of ambient dimension to latent dimension. Because
      the sketch_width must use integer increments, the actual compression rate
      may be slightly larger. The `min_compression_rate` and `num_repeats`
      defines the `sketch_width`.
    bits: A positive integer specifying the communication bit-width B (where 2^B
      will be the field size for SecAgg operations). Note that this is for the
      noisy quantized aggregate at the server and thus should account for the
      number of clients. Must be in the inclusive range [1, 22], and should be
      at least as large as log_2(expected_clients_per_round).
    decode_method: a count_sketching_utils.DecodeMethod. `MEAN` is preferred for
      runtime and memory.
    num_repeats: The number of independent hashes to apply, i.e., the sketch
      length.
    rotation_type: (Optional) The rotation operation used to spread out input
      values across vector dimensions. Possible options are `ht` (randomized
      Hadamard transform) or `dft` (discrete Fourier transform). Defaults to
      `ht`.
    zeroing: A bool indicating whether to enable adaptive zeroing for data
      corruption mitigation. Defaults to `False`.

  Returns:
    tf.aggaregators.UnweightedAggregationFactory performing the following
    operations:
      1. (Optionally) zeroing gradient weights for data corruption mitigation.
      2. Concatenate weights to a single vector.
      2. Compress weights using the
        `count_sketching.GradientCountSketchFactory`
      3. Perform aggregation as in `tff.learning.ddp_secure_aggregator`
  """
  # TODO(b/228072455): Switch to public API when its available.
  ddp_aggregator = tff.aggregators.distributed_dp.DistributedDpSumFactory(
      noise_multiplier=noise_multiplier,
      expected_clients_per_round=expected_clients_per_round,
      bits=bits,
      l2_clip=0.1,
      mechanism='distributed_skellam',
      rotation_type=rotation_type,
      auto_l2_clip=True)
  ddp_aggregator = tff.aggregators.UnweightedMeanFactory(
      value_sum_factory=ddp_aggregator,
      count_sum_factory=tff.aggregators.SecureSumFactory(
          upper_bound_threshold=1, lower_bound_threshold=0))
  if compression_rate == 0:
    sketching = ddp_aggregator
  else:
    sketching = count_sketching.GradientCountSketchFactory(
        min_compression_rate=compression_rate,
        decode_method=decode_method,
        num_repeats=num_repeats,
        inner_agg_factory=ddp_aggregator)
  # TODO(b/225446368): Consider removing application of flattening in DDP.
  agg_factory = tff.aggregators.concat_factory(sketching)
  if zeroing:
    return _default_zeroing(agg_factory)
  else:
    return agg_factory


def compressed_central_dp_factory(
    noise_multiplier: float,
    expected_clients_per_round: int,
    compression_rate: float,
    decode_method: count_sketching_utils.DecodeMethod = count_sketching_utils
    .DecodeMethod.MEAN,
    num_repeats: int = 15,
    zeroing: bool = False,
) -> tff.aggregators.UnweightedAggregationFactory:
  """Wrapper to combine baseline central DP factory with sketching.

  Args:
    noise_multiplier: A float specifying the noise multiplier (central noise
      stddev / L2 clip norm) for model updates. Note that this is with respect
      to the initial L2 clip norm, and the quantization procedure as part of the
      DDP algorithm may inflated the L2 sensitivity. The specified noise will be
      split into `expected_clients_per_round` noise shares to be added locally
      on the clients. A value of 1.0 or higher may be needed for strong privacy.
      Must be nonnegative. A value of 0.0 means no noise will be added.
    expected_clients_per_round: An integer specifying the expected number of
      clients to participate in this round. Must be a positive integer.
    compression_rate: Ratio of ambient dimension to latent dimension. Because
      the sketch_width must use integer increments, the actual compression rate
      may be slightly larger. The `min_compression_rate` and `num_repeats`
      defines the `sketch_width`.
    decode_method: a count_sketching_utils.DecodeMethod. `MEAN` is preferred for
      runtime and memory.
    num_repeats: The number of independent hashes to apply, i.e., the sketch
      length.
    zeroing: A bool indicating whether to enable adaptive zeroing for data
      corruption mitigation. Defaults to `False`.

  Returns:
    tf.aggaregators.UnweightedAggregationFactory performing the following
    operations:
      1. (Optionally) zeroing gradient weights for data corruption mitigation.
      2. Concatenate weights to a single vector.
      2. Compress weights using the
        `count_sketching.GradientCountSketchFactory`
      3. Perform aggregation as in `tff.learning.ddp_secure_aggregator`
  """
  cdp_aggregator = (
      tff.aggregators.DifferentiallyPrivateFactory.gaussian_adaptive(
          noise_multiplier,
          expected_clients_per_round,
          initial_l2_norm_clip=0.1))
  cdp_aggregator = tff.aggregators.UnweightedMeanFactory(
      value_sum_factory=cdp_aggregator)
  if compression_rate == 0:
    sketching = cdp_aggregator
  else:
    sketching = count_sketching.GradientCountSketchFactory(
        min_compression_rate=compression_rate,
        decode_method=decode_method,
        num_repeats=num_repeats,
        inner_agg_factory=cdp_aggregator)
  agg_factory = tff.aggregators.concat_factory(sketching)
  if zeroing:
    return _default_zeroing(agg_factory)
  else:
    return agg_factory
