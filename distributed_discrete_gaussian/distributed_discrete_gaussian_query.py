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
"""Implements DPQuery interface for distributed discrete Gaussian mechanism."""

import collections

import tensorflow as tf
import tensorflow_privacy as tfp

from distributed_discrete_gaussian import discrete_gaussian_utils


class DistributedDiscreteGaussianSumQuery(tfp.SumAggregationDPQuery):
  """Implements DPQuery for discrete distributed Gaussian sum queries.

  For each local record, we check the L2 norm bound and add discrete Gaussian
  noise. In particular, this DPQuery does not perform L2 norm clipping and the
  norms of the input records are expected to be bounded.
  """

  # pylint: disable=invalid-name
  _GlobalState = collections.namedtuple('_GlobalState',
                                        ['l2_norm_bound', 'local_scale'])

  # pylint: disable=invalid-name
  _SampleParams = collections.namedtuple('_SampleParams',
                                         ['l2_norm_bound', 'local_scale'])

  def __init__(self, l2_norm_bound, local_scale):
    """Initializes the DistributedDiscreteGaussianSumQuery.

    Args:
      l2_norm_bound: The L2 norm bound to verify for each record.
      local_scale: The scale (stddev) of the local discrete Gaussian noise.
    """
    self._l2_norm_bound = l2_norm_bound
    self._local_scale = local_scale

  def set_ledger(self, ledger):
    raise NotImplementedError('Ledger has not yet been implemented for'
                              'DistributedDiscreteGaussianSumQuery!')

  def initial_global_state(self):
    return self._GlobalState(
        tf.cast(self._l2_norm_bound, tf.float32),
        tf.cast(self._local_scale, tf.int32))  # Only integer scales for now.

  def derive_sample_params(self, global_state):
    return self._SampleParams(global_state.l2_norm_bound,
                              global_state.local_scale)

  def _add_local_noise(self, record, local_scale, shares=1):
    """Add local discrete Gaussian noise to the record.

    Args:
      record: The record to which we generate and add local noise.
      local_scale: The scale (stddev) of the local discrete Gaussian noise.
      shares: Number of shares of local noise to generate. Should be 1 for each
        record. This can be useful when we want to generate multiple noise
        shares at once.

    Returns:
      The record with local noise added.
    """

    def add_noise(v):
      # Adds an extra dimension for `shares` number of draws.
      shape = tf.concat([[shares], tf.shape(v)], axis=0)
      dgauss_noise = discrete_gaussian_utils.sample_discrete_gaussian(
          scale=local_scale, shape=shape, dtype=v.dtype)
      # Sum across the number of noise shares and add it.
      return v + tf.reduce_sum(dgauss_noise, axis=0)

    return tf.nest.map_structure(add_noise, record)

  def preprocess_record(self, params, record):
    """Check record norm and add noise to the record."""
    record_as_list = tf.nest.flatten(record)
    record_as_float_list = [tf.cast(x, tf.float32) for x in record_as_list]
    tf.nest.map_structure(lambda x: tf.compat.v1.assert_type(x, tf.int32),
                          record_as_list)
    dependencies = [
        tf.compat.v1.assert_less_equal(
            tf.linalg.global_norm(record_as_float_list),
            params.l2_norm_bound,
            message=f'Global L2 norm exceeds {params.l2_norm_bound}.')
    ]
    with tf.control_dependencies(dependencies):
      result = tf.cond(
          tf.equal(params.local_scale, 0), lambda: record,
          lambda: self._add_local_noise(record, params.local_scale))
      return result

  def get_noised_result(self, sample_state, global_state):
    """The noise was added locally, so simply return the aggregate."""
    # Note that this assumes we won't have clients dropping out (thus missing
    # local noise shares) for experiments.
    return sample_state, global_state
