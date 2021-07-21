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
"""Utils for constructing DP queries from TF Privacy."""

from absl import logging

from distributed_dp import accounting_utils
from distributed_dp import compression_query
from distributed_dp import distributed_discrete_gaussian_query


def _ddp_query_factory(mechanism, local_stddev, l1_norm_bound, l2_norm_bound):
  """Factory for distributed discrete DPQuery objects from TF Privacy."""
  del l1_norm_bound  # Unused.
  mechanism = mechanism.lower()
  if mechanism == 'ddgauss':
    return distributed_discrete_gaussian_query.DistributedDiscreteGaussianSumQuery(
        l2_norm_bound=l2_norm_bound, local_scale=local_stddev)
  else:
    raise ValueError(f'Unsupported mechanism: "{mechanism}".')


def build_ddp_query(mechanism, local_stddev, l2_norm_bound, beta, padded_dim,
                    scale, client_template):
  """Construct a DDP query object wrapped with quantization operations."""
  beta = beta or 0
  conditional = beta > 0
  logging.info('Conditional rounding set to %s (beta = %f)', conditional, beta)

  # Add some post-rounding norm leeway to peacefully allow for precision issues.
  scaled_rounded_l2 = accounting_utils.rounded_l2_norm_bound(
      (l2_norm_bound + 1e-5) * scale, beta=beta, dim=padded_dim)
  scaled_rounded_l1 = accounting_utils.rounded_l1_norm_bound(
      scaled_rounded_l2, padded_dim)
  ddp_query = _ddp_query_factory(
      mechanism=mechanism,
      local_stddev=local_stddev * scale,
      l1_norm_bound=scaled_rounded_l1,
      l2_norm_bound=scaled_rounded_l2)

  # Wrap DDP query with quantization operations.
  quantization_params = compression_query.QuantizationParams(
      stochastic=True,
      conditional=conditional,
      l2_norm_bound=l2_norm_bound,
      beta=beta,
      quantize_scale=scale)
  quantized_ddp_query = compression_query.CompressionSumQuery(
      quantization_params=quantization_params,
      inner_query=ddp_query,
      record_template=client_template)

  return quantized_ddp_query
