# Copyright 2023, Google LLC.
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
"""Tests for tff_aggregator."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from multi_epoch_dp_matrix_factorization import matrix_constructors
from multi_epoch_dp_matrix_factorization import matrix_factorization_query
from multi_epoch_dp_matrix_factorization import tff_aggregator


def _make_prefix_sum_matrix(dim: int) -> tf.Tensor:
  return tf.constant(np.tril(np.ones(shape=[dim] * 2)), dtype=tf.float32)


class PrefixSumAggregatorTest(tf.test.TestCase):

  def test_aggregator_factory_constructs(self):
    dim = 3
    tensor_specs = tf.TensorSpec(dtype=tf.float32, shape=[])
    l2_norm_clip = 1.0
    noise_multiplier = 0.0
    w_matrix = _make_prefix_sum_matrix(dim)
    h_matrix = tf.eye(dim)
    clients_per_round = 1
    seed = 0

    agg_factory = tff_aggregator.create_residual_prefix_sum_dp_factory(
        tensor_specs=tensor_specs,
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        w_matrix=w_matrix,
        h_matrix=h_matrix,
        clients_per_round=clients_per_round,
        seed=seed,
    )
    self.assertIsInstance(
        agg_factory, tff.aggregators.UnweightedAggregationFactory
    )

  def test_aggregator_raises_with_mismatched_type_structure(self):
    dim = 3
    tensor_specs = tf.TensorSpec(dtype=tf.float32, shape=[])
    l2_norm_clip = 1.0
    noise_multiplier = 0.0
    w_matrix = _make_prefix_sum_matrix(dim)
    h_matrix = tf.eye(dim)
    clients_per_round = 1
    seed = 0

    agg_factory = tff_aggregator.create_residual_prefix_sum_dp_factory(
        tensor_specs=tensor_specs,
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        w_matrix=w_matrix,
        h_matrix=h_matrix,
        clients_per_round=clients_per_round,
        seed=seed,
    )
    with self.assertRaises(ValueError):
      agg_factory.create(
          value_type=tff.types.type_from_tensors([tf.zeros(shape=[])] * 2)
      )

  def test_unnoised_prefix_sum_aggregator_performs_federated_mean(self):
    dim = 3
    tensor_specs = tf.TensorSpec(dtype=tf.float32, shape=[])
    # Set the l2 norm clip large enough so that none of the incoming values are
    # clipped.
    l2_norm_clip = 1.0 * dim
    noise_multiplier = 0.0
    w_matrix = _make_prefix_sum_matrix(dim)
    h_matrix = tf.cast(tf.eye(dim), w_matrix.dtype)
    clients_per_round = 2
    seed = 0

    agg_factory = tff_aggregator.create_residual_prefix_sum_dp_factory(
        tensor_specs=tensor_specs,
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        w_matrix=w_matrix,
        h_matrix=h_matrix,
        clients_per_round=clients_per_round,
        seed=seed,
    )

    aggregator_process = agg_factory.create(
        value_type=tff.TensorType(dtype=tf.float32, shape=[])
    )
    agg_state = aggregator_process.initialize()
    for i in range(dim):
      client_1_value = 0.5 * float(i)
      client_2_value = 1.5 * float(i)
      output = aggregator_process.next(
          agg_state, [client_1_value, client_2_value]
      )
      result = output.result
      agg_state = output.state
      expected_mean = 0.5 * (client_1_value + client_2_value)
      # Since values are scalar and positive, the expected mean
      # is also the average_client_norm.
      self.assertEqual(
          output.measurements['average_client_norm'], expected_mean
      )
      self.assertEqual(result, expected_mean)

  def test_noised_prefix_sum_outputs_residuals(self):
    dim = 10
    tensor_specs = tf.TensorSpec(dtype=tf.float32, shape=[])
    # Set the l2 norm clip large enough so that none of the incoming values are
    # clipped.
    l2_norm_clip = 1.0 * dim
    noise_multiplier = 1.0 / l2_norm_clip
    w_matrix = tf.constant(np.tril(np.ones(shape=[dim, dim])))
    h_matrix = tf.cast(tf.eye(dim), w_matrix.dtype)
    clients_per_round = 2
    seed = 0

    # Technically, the assertions to follow rely on the fact that seed is passed
    # directly to the `OnTheFlyNoiseMechanism` from the call below.
    underlying_mechanism = (
        matrix_factorization_query.OnTheFlyFactorizedNoiseMechanism(
            tensor_specs=tensor_specs,
            stddev=noise_multiplier * l2_norm_clip,
            w_matrix=w_matrix,
            seed=seed,
        )
    )
    mech_state = underlying_mechanism.initialize()

    agg_factory = tff_aggregator.create_residual_prefix_sum_dp_factory(
        tensor_specs=tensor_specs,
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        w_matrix=w_matrix,
        h_matrix=h_matrix,
        clients_per_round=clients_per_round,
        seed=seed,
    )

    aggregator_process = agg_factory.create(
        value_type=tff.TensorType(dtype=tf.float32, shape=[])
    )
    agg_state = aggregator_process.initialize()
    previous_noise = tf.zeros(shape=[], dtype=tf.float32)
    for i in range(dim):
      output = aggregator_process.next(agg_state, [float(i)] * 2)
      noise_at_index, mech_state = underlying_mechanism.compute_noise(
          mech_state
      )
      result = output.result
      agg_state = output.state
      # We added the noise before computing the mean, so we divide by the number
      # of clients per round here.
      self.assertEqual(
          result, i + (noise_at_index - previous_noise) / clients_per_round
      )
      previous_noise = noise_at_index


class MomentumMatrixResidualTest(parameterized.TestCase, tf.test.TestCase):

  def test_aggregator_factory_constructs(self):
    dim = 3
    tensor_specs = tf.TensorSpec(dtype=tf.float32, shape=[])
    l2_norm_clip = 1.0
    noise_multiplier = 0.0
    momentum_value = 0.9
    w_matrix = tf.constant(
        matrix_constructors.momentum_sgd_matrix(
            num_iters=dim, momentum=momentum_value
        ),
        dtype=tf.float32,
    )
    h_matrix = tf.eye(dim)
    clients_per_round = 1
    seed = 0

    agg_factory = tff_aggregator.create_residual_momentum_dp_factory(
        tensor_specs=tensor_specs,
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        w_matrix=w_matrix,
        h_matrix=h_matrix,
        clients_per_round=clients_per_round,
        seed=seed,
        momentum_value=momentum_value,
    )
    self.assertIsInstance(
        agg_factory, tff.aggregators.UnweightedAggregationFactory
    )

  def test_aggregator_raises_with_mismatched_type_structure(self):
    dim = 3
    tensor_specs = tf.TensorSpec(dtype=tf.float32, shape=[])
    l2_norm_clip = 1.0
    noise_multiplier = 0.0
    momentum_value = 0.9
    w_matrix = tf.constant(
        matrix_constructors.momentum_sgd_matrix(
            num_iters=dim, momentum=momentum_value
        ),
        dtype=tf.float32,
    )
    h_matrix = tf.eye(dim)
    clients_per_round = 1
    seed = 0

    agg_factory = tff_aggregator.create_residual_momentum_dp_factory(
        tensor_specs=tensor_specs,
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        w_matrix=w_matrix,
        h_matrix=h_matrix,
        clients_per_round=clients_per_round,
        seed=seed,
        momentum_value=momentum_value,
    )
    with self.assertRaises(ValueError):
      agg_factory.create(
          value_type=tff.types.type_from_tensors([tf.zeros(shape=[])] * 2)
      )

  def test_unnoised_prefix_sum_aggregator_performs_momentum_federated_mean(
      self,
  ):
    dim = 3
    tensor_specs = tf.TensorSpec(dtype=tf.float32, shape=[])
    # Set the l2 norm clip large enough so that none of the incoming values are
    # clipped.
    l2_norm_clip = 1.0 * dim
    noise_multiplier = 0.0
    momentum_value = 0.9
    w_matrix = tf.constant(
        matrix_constructors.momentum_sgd_matrix(
            num_iters=dim, momentum=momentum_value
        ),
        dtype=tf.float32,
    )
    h_matrix = tf.cast(tf.eye(dim), w_matrix.dtype)
    clients_per_round = 2
    seed = 0

    agg_factory = tff_aggregator.create_residual_momentum_dp_factory(
        tensor_specs=tensor_specs,
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        w_matrix=w_matrix,
        h_matrix=h_matrix,
        clients_per_round=clients_per_round,
        seed=seed,
        momentum_value=momentum_value,
    )

    aggregator_process = agg_factory.create(
        value_type=tff.TensorType(dtype=tf.float32, shape=[])
    )
    agg_state = aggregator_process.initialize()
    momentum_accumulator = 0.0
    for i in range(dim):
      # Average client values; no noise is added, so we should just get these
      # values back.
      output = aggregator_process.next(agg_state, [float(i)] * 2)
      result = output.result
      agg_state = output.state

      expected_result = i + momentum_accumulator * momentum_value
      momentum_accumulator = expected_result
      self.assertAllClose(result, expected_result)

  def test_noised_outputs_residuals(self):
    dim = 10
    tensor_specs = tf.TensorSpec(dtype=tf.float32, shape=[])
    # Set the l2 norm clip large enough so that none of the incoming values are
    # clipped.
    l2_norm_clip = 1.0 * dim
    noise_multiplier = 1.0 / l2_norm_clip
    momentum_value = 0.9
    w_matrix = tf.constant(
        matrix_constructors.momentum_sgd_matrix(
            num_iters=dim, momentum=momentum_value
        ),
        dtype=tf.float32,
    )
    h_matrix = tf.cast(tf.eye(dim), w_matrix.dtype)

    clients_per_round = 2
    seed = 0

    # Technically, the assertions to follow rely on the fact that seed is passed
    # directly to the `OnTheFlyNoiseMechanism` from the call below.
    underlying_mechanism = (
        matrix_factorization_query.OnTheFlyFactorizedNoiseMechanism(
            tensor_specs=tensor_specs,
            stddev=noise_multiplier * l2_norm_clip,
            w_matrix=w_matrix,
            seed=seed,
        )
    )
    mech_state = underlying_mechanism.initialize()

    agg_factory = tff_aggregator.create_residual_momentum_dp_factory(
        tensor_specs=tensor_specs,
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        w_matrix=w_matrix,
        h_matrix=h_matrix,
        clients_per_round=clients_per_round,
        seed=seed,
        momentum_value=momentum_value,
    )

    aggregator_process = agg_factory.create(
        value_type=tff.TensorType(dtype=tf.float32, shape=[])
    )
    agg_state = aggregator_process.initialize()
    previous_noise = tf.zeros(shape=[], dtype=tf.float32)
    momentum_accumulator = 0.0
    for i in range(dim):
      output = aggregator_process.next(agg_state, [float(i)] * 2)
      noise_at_index, mech_state = underlying_mechanism.compute_noise(
          mech_state
      )
      result = output.result
      agg_state = output.state

      expected_unnoised_result = i + momentum_accumulator * momentum_value
      momentum_accumulator = expected_unnoised_result
      # We added the noise before computing the mean, so we divide by the number
      # of clients per round here.
      self.assertAllClose(
          result,
          expected_unnoised_result
          + (noise_at_index - previous_noise) / clients_per_round,
      )
      previous_noise = noise_at_index

  def _create_5_round_process(self, **kwargs):
    dim = 5
    tensor_specs = tf.TensorSpec(dtype=tf.float32, shape=[])
    momentum_value = 0.9
    w_matrix = tf.constant(
        matrix_constructors.momentum_sgd_matrix(
            num_iters=dim, momentum=momentum_value
        ),
        dtype=tf.float32,
    )
    h_matrix = tf.cast(tf.eye(dim), w_matrix.dtype)
    clients_per_round = 2
    agg_factory = tff_aggregator.create_residual_momentum_dp_factory(
        tensor_specs=tensor_specs,
        l2_norm_clip=float(dim),
        noise_multiplier=1,
        w_matrix=w_matrix,
        h_matrix=h_matrix,
        clients_per_round=clients_per_round,
        seed=0,
        momentum_value=momentum_value,
        **kwargs,
    )
    return agg_factory.create(
        value_type=tff.TensorType(dtype=tf.float32, shape=[])
    )

  def test_raises_on_extra_rounds(self):
    dim = 5
    aggregator_process = self._create_5_round_process(
        emit_zeros_after_last_round=False
    )
    agg_state = aggregator_process.initialize()
    for i in range(dim):
      output = aggregator_process.next(agg_state, [float(i)] * 2)
      agg_state = output.state

    # An extra round raises a TF error:
    with self.assertRaisesRegex(
        Exception, 'can therefore only support 5 rounds'
    ):
      aggregator_process.next(agg_state, [float(1)] * 2)

  def test_zeros_on_extra_rounds(self):
    dim = 5
    aggregator_process = self._create_5_round_process(
        emit_zeros_after_last_round=True
    )
    agg_state = aggregator_process.initialize()
    for i in range(dim):
      output = aggregator_process.next(agg_state, [float(2 * i)] * 2)
      agg_state = output.state

    # Extra rounds produce zeros:
    for _ in range(3):
      output = aggregator_process.next(agg_state, [float(10)] * 2)
      agg_state = output.state
      self.assertAllClose(output.result, 0.0)


if __name__ == '__main__':
  tf.test.main()
