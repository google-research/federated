# Copyright 2021, Google LLC. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utils for distributed mean estimation."""

import numpy as np
import tensorflow as tf
import tensorflow_privacy as tfp

from distributed_discrete_gaussian import compression_query
from distributed_discrete_gaussian import distributed_discrete_gaussian_query
from distributed_discrete_gaussian.accounting_utils import compute_l2_sensitivy_squared
from distributed_discrete_gaussian.modular_clipping_factory import modular_clip_by_value


def get_dp_query(mechanism, l2_norm_bound, noise_scale):
  """Factory for DPQuery instances.

  Args:
    mechanism: The mechanism name string.
    l2_norm_bound: The L2 norm bound to be checked by the DPQueries. Note that
      for discrete queries, these are the bounds after scaling.
    noise_scale: The noise scale (stddev) for the mechanism. Note that for
      central queries, this is the central stddev; for distributed queries, this
      is the local stddev for each client.

  Returns:
    A DPQuery object.
  """
  mechanism = mechanism.lower()
  if mechanism == 'gauss':
    return tfp.GaussianSumQuery(l2_norm_clip=l2_norm_bound, stddev=noise_scale)
  elif mechanism == 'distributed_dgauss':
    return distributed_discrete_gaussian_query.DistributedDiscreteGaussianSumQuery(
        l2_norm_bound=l2_norm_bound, local_scale=noise_scale)
  else:
    raise ValueError(f'Not yet implemented: {mechanism}')


def generate_client_data(d, n, l2_norm=1):
  """Sample `n` of `d`-dim vectors on the l2 ball with radius `l2_norm`.

  Args:
    d: The dimension of the client vector.
    n: The number of clients.
    l2_norm: The L2 norm of the sampled vector.

  Returns:
    A list of `n` np.array each with shape (d,).
  """
  vectors = np.random.normal(size=(n, d))
  unit_vectors = vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)
  scaled_vectors = unit_vectors * l2_norm
  # Cast to float32 as TF implementations use float32.
  return list(scaled_vectors.astype(np.float32))


def compute_dp_average(client_data, dp_query, is_compressed, bits):
  """Aggregate client data with DPQuery's interface and take average."""
  global_state = dp_query.initial_global_state()
  sample_params = dp_query.derive_sample_params(global_state)

  client_template = tf.zeros_like(client_data[0])
  sample_state = dp_query.initial_sample_state(client_template)

  if is_compressed:
    # Achieve compression via modular clipping. Upper bound is exclusive.
    clip_lo, clip_hi = -(2**(bits - 1)), 2**(bits - 1)

    # 1. Client pre-processing stage.
    for x in client_data:
      record = tf.convert_to_tensor(x)
      prep_record = dp_query.preprocess_record(sample_params, record)
      # Client applies modular clip on the preprocessed record.
      prep_record = modular_clip_by_value(prep_record, clip_lo, clip_hi)
      sample_state = dp_query.accumulate_preprocessed_record(
          sample_state, prep_record)

    # 2. Server applies modular clip on the aggregate.
    sample_state = modular_clip_by_value(sample_state, clip_lo, clip_hi)

  else:
    for x in client_data:
      record = tf.convert_to_tensor(x)
      sample_state = dp_query.accumulate_record(
          sample_params, sample_state, record=record)

  # Apply server post-processing.
  agg_result, _ = dp_query.get_noised_result(sample_state, global_state)

  # The agg_result should have the same input type as client_data.
  assert agg_result.shape == client_data[0].shape
  assert agg_result.dtype == client_data[0].dtype

  # Take the average on the aggregate.
  return agg_result / len(client_data)


def build_quantized_dp_query(mechanism, c, padded_d, gamma, stddev, beta,
                             client_template):
  """Construct a DPQuery with quantization operations."""
  # The implementation adds noise on scaled records, and takes integer scales
  gamma, stddev, padded_d, c, beta = map(np.float64,
                                         [gamma, stddev, padded_d, c, beta])
  # Take ceil as DGauss impl only supports integer scales for now.
  scaled_stddev = np.ceil(stddev / gamma)

  # L2 norm bound: The DPQuery implementations expect scaled norm bounds.
  sq_l2_norm_bound = compute_l2_sensitivy_squared(
      l2_clip_norm=c, gamma=gamma, beta=beta, dimension=padded_d)
  # Add some norm leeway to peacefully allow for numerical/precision errors.
  scaled_l2_norm_bound = (np.sqrt(sq_l2_norm_bound) + 1e-4) / gamma

  # Construct DPQuery
  discrete_query = get_dp_query(
      mechanism, l2_norm_bound=scaled_l2_norm_bound, noise_scale=scaled_stddev)

  # Wrap the discrete DPQuery with CompressionSumQuery.
  quantize_scale = 1.0 / gamma
  quantization_params = compression_query.ScaledQuantizationParams(
      stochastic=True,
      conditional=True,
      l2_norm_bound=c,
      beta=beta,
      quantize_scale=quantize_scale)

  compression_dp_query = compression_query.CompressionSumQuery(
      quantization_params=quantization_params,
      inner_query=discrete_query,
      record_template=client_template)

  return compression_dp_query
