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
"""Utils for running experiments with discrete DP and compression."""

import pprint

from absl import logging
import numpy as np
import tensorflow_federated as tff

from distributed_discrete_gaussian import accounting_utils
from distributed_discrete_gaussian import compression_query
from distributed_discrete_gaussian import distributed_discrete_gaussian_query
from distributed_discrete_gaussian import modular_clipping_factory


def get_total_dim(client_template):
  """Returns the dimension of the client template as a single vector."""
  return sum(np.prod(x.shape) for x in client_template)


def pad_dim(dim):
  return np.math.pow(2, np.ceil(np.log2(dim)))


def dp_query_factory(mechanism, noise_stddev, quantized_l2_norm_bound):
  """Factory for instantiating discrete DP mechanisms from TF Privacy."""
  mechanism = mechanism.lower()
  if mechanism == 'ddgauss':
    return distributed_discrete_gaussian_query.DistributedDiscreteGaussianSumQuery(
        l2_norm_bound=quantized_l2_norm_bound, local_scale=noise_stddev)
  else:
    raise ValueError(f'Unsupported mechanism: "{mechanism}".')


def build_compressed_dp_query(mechanism, clip, padded_dim, gamma, stddev, beta,
                              client_template):
  """Construct a DPQuery with quantization operations."""
  # Scaling up the noise for the quantized query.
  scaled_stddev = np.ceil(stddev / gamma)
  # Compute the (scaled) inflated norm bound after random Hadamard transform.
  sq_l2_norm_bound = accounting_utils.compute_l2_sensitivy_squared(
      l2_clip_norm=clip, gamma=gamma, beta=beta, dimension=padded_dim)
  # Add some norm leeway to peacefully allow for numerical/precision errors.
  scaled_l2_norm_bound = (np.sqrt(sq_l2_norm_bound) + 1e-4) / gamma

  discrete_query = dp_query_factory(
      mechanism=mechanism,
      noise_stddev=scaled_stddev,
      quantized_l2_norm_bound=scaled_l2_norm_bound)

  quantize_scale = 1.0 / gamma
  beta = beta or 0
  conditional = beta > 0
  logging.info('Conditional rounding set to %s (beta = %f)', conditional, beta)

  quantization_params = compression_query.ScaledQuantizationParams(
      stochastic=True,
      conditional=conditional,
      l2_norm_bound=clip,
      beta=beta,
      quantize_scale=quantize_scale)

  # Wrap the discrete query with compression operations.
  compressed_query = compression_query.CompressionSumQuery(
      quantization_params=quantization_params,
      inner_query=discrete_query,
      record_template=client_template)

  return compressed_query


def build_aggregator(compression_flags, dp_flags, num_clients,
                     num_clients_per_round, num_rounds, client_template):
  """Create a `tff.aggregator` containing all aggregation operations."""

  clip, epsilon = dp_flags['l2_norm_clip'], dp_flags['epsilon']
  # No DP (but still do the clipping if necessary).
  if epsilon is None:
    agg_factory = tff.aggregators.UnweightedMeanFactory()
    if clip is not None:
      assert clip > 0, 'Norm clip must be positive.'
      agg_factory = tff.aggregators.clipping_factory(clip, agg_factory)
    logging.info('Using vanilla sum aggregation with clipping %s', clip)
    return agg_factory

  # Parameters for DP
  assert epsilon > 0, f'Epsilon should be positive, found {epsilon}.'
  assert clip is not None and clip > 0, f'Clip must be positive, found {clip}.'
  sampling_rate = float(num_clients_per_round) / num_clients
  delta = dp_flags['delta'] or 1.0 / num_clients  # Default to delta = 1 / n.
  dim = get_total_dim(client_template)

  logging.info('Shared DP Parameters:')
  logging.info(
      pprint.pformat({
          'epsilon': epsilon,
          'delta': delta,
          'clip': clip,
          'dim': dim,
          'sampling_rate': sampling_rate,
          'num_clients': num_clients,
          'num_clients_per_round': num_clients_per_round,
          'num_rounds': num_rounds
      }))

  # Baseline: continuous Gaussian
  if dp_flags['dp_mechanism'] == 'gaussian':
    noise_mult = accounting_utils.get_gauss_noise_multiplier(
        target_eps=epsilon,
        target_delta=delta,
        target_sampling_rate=sampling_rate,
        steps=num_rounds)
    # Operations include clipping on client and noising + averaging on server;
    # No MeanFactory and ClippingFactory needed.
    agg_factory = tff.aggregators.DifferentiallyPrivateFactory.gaussian_fixed(
        noise_multiplier=noise_mult,
        clients_per_round=num_clients_per_round,
        clip=clip)
    logging.info('Gaussian Parameters:')
    logging.info({'noise_mult': noise_mult})

  # Distributed Discrete Gaussian
  elif dp_flags['dp_mechanism'] == 'ddgauss':
    padded_dim = pad_dim(dim)

    k_stddevs = compression_flags['k_stddevs'] or 2
    beta = compression_flags['beta']
    bits = compression_flags['num_bits']

    # Modular clipping has exclusive upper bound.
    mod_clip_lo, mod_clip_hi = -(2**(bits - 1)), 2**(bits - 1)

    gamma = accounting_utils.get_ddgauss_gamma(
        q=sampling_rate,
        epsilon=epsilon,
        l2_clip_norm=clip,
        bits=bits,
        num_clients=num_clients_per_round,
        dimension=padded_dim,
        delta=delta,
        beta=beta,
        steps=num_rounds,
        k=k_stddevs,
        sqrtn_norm_growth=False)

    local_stddev = accounting_utils.get_ddgauss_noise_stddev(
        q=sampling_rate,
        epsilon=epsilon,
        l2_clip_norm=clip,
        gamma=gamma,
        beta=beta,
        steps=num_rounds,
        num_clients=num_clients_per_round,
        dimension=padded_dim,
        delta=delta)

    logging.info('DDGauss Parameters:')
    logging.info(
        pprint.pformat({
            'bits': bits,
            'beta': beta,
            'dim': dim,
            'padded_dim': padded_dim,
            'gamma': gamma,
            'k_stddevs': k_stddevs,
            'local_stddev': local_stddev
        }))

    # Build nested aggregators.
    agg_factory = tff.aggregators.SumFactory()
    # 1. Modular clipping.
    agg_factory = modular_clipping_factory.ModularClippingSumFactory(
        clip_range_lower=mod_clip_lo,
        clip_range_upper=mod_clip_hi,
        inner_agg_factory=agg_factory)

    # 2. DPFactory that uses the compressed_query.
    compressed_query = build_compressed_dp_query(
        mechanism='ddgauss',
        clip=clip,
        padded_dim=padded_dim,
        gamma=gamma,
        stddev=local_stddev,
        beta=beta,
        client_template=client_template)

    agg_factory = tff.aggregators.DifferentiallyPrivateFactory(
        query=compressed_query, record_aggregation_factory=agg_factory)

    # 3. L2 norm clipping as the first step.
    agg_factory = tff.aggregators.clipping_factory(
        clipping_norm=clip, inner_agg_factory=agg_factory)

    # 4. Apply a MeanFactory at last (mean can't be part of the discrete
    # DPQueries (like the case of Gaussian) as the records may become floats
    # and hence break the decompression process).
    agg_factory = tff.aggregators.UnweightedMeanFactory(
        value_sum_factory=agg_factory)

  else:
    raise ValueError(f'Unsupported mechanism: {dp_flags["dp_mechanism"]}')

  return agg_factory
