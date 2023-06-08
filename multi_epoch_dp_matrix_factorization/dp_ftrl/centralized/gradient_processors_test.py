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
"""Tests for gradient_processors."""
import collections

from jax import numpy as jnp
import tensorflow as tf
import tensorflow_federated as tff

from multi_epoch_dp_matrix_factorization.dp_ftrl.centralized import gradient_processors


def _build_no_privacy_scalar_tree_agg(clip_norm=1.0):
  record_specs = tf.TensorSpec(shape=[], dtype=tf.float32)
  aggregator = tff.aggregators.DifferentiallyPrivateFactory.tree_aggregation(
      noise_multiplier=0.0,
      # Our implementation only passes *one* value through to the
      # aggregator, and does not want any normalization inside the aggregator.
      clients_per_round=1,
      l2_norm_clip=clip_norm,
      record_specs=record_specs,
      noise_seed=0,
      use_efficient=True,
  )
  return aggregator.create(tff.to_type(tf.float32))


class GradientProcessorsTest(tf.test.TestCase):

  def test_noprivacy_processor_builds_and_runs(self):
    num_microbatches = 10
    grad_processor = gradient_processors.NoPrivacyGradientProcessor()
    state = grad_processor.init()
    batched_grads = jnp.array([1.0] * num_microbatches)
    state, grad_estimate = grad_processor.apply(state, batched_grads)
    self.assertAllClose(grad_estimate, 1.0)

  def test_constructs_and_runs_from_tree_aggregator(self):
    num_microbatches = 10
    aggregator = _build_no_privacy_scalar_tree_agg()
    grad_processor = gradient_processors.DPAggregatorBackedGradientProcessor(
        aggregator, l2_norm_clip=1.0
    )
    state = grad_processor.init()
    batched_grads = jnp.array([1.0] * num_microbatches)
    state, grad_estimate = grad_processor.apply(state, batched_grads)
    self.assertAllClose(grad_estimate, 1.0)

  def test_constructs_and_runs_tree_agg_with_model_structure(self):
    num_microbatches = 10
    record_specs = collections.OrderedDict(
        a=tf.TensorSpec(shape=[100, 100], dtype=tf.float32),
        b=tf.TensorSpec(shape=[100], dtype=tf.float32),
    )
    factory = tff.aggregators.DifferentiallyPrivateFactory.tree_aggregation(
        noise_multiplier=0.0,
        clients_per_round=1,
        l2_norm_clip=1.0,
        record_specs=record_specs,
        noise_seed=0,
        use_efficient=True,
    )
    aggregator = factory.create(tff.to_type(record_specs))
    grad_processor = gradient_processors.DPAggregatorBackedGradientProcessor(
        aggregator, l2_norm_clip=1.0
    )
    state = grad_processor.init()
    # Jax returns our per-example gradients as dicts, so we test against this
    # here.
    batched_grads = dict(
        a=jnp.array([jnp.ones(shape=[100, 100])] * num_microbatches),
        b=jnp.array([jnp.ones(shape=[100])] * num_microbatches),
    )
    # We compute the l2 norm of each 'per-example' gradient; since the clipping
    # norm is 1.0, dividing by this will yield the per-element expected value of
    # the result.
    argument_l2_norm = (100**2 + 100) ** 0.5
    expected_result = dict(
        a=jnp.array(jnp.ones(shape=[100, 100]) / argument_l2_norm),
        b=jnp.array(jnp.ones(shape=[100])) / argument_l2_norm,
    )
    state, grad_estimate = grad_processor.apply(state, batched_grads)
    self.assertAllClose(grad_estimate, expected_result)

  def test_aggregator_clips_and_sums(self):
    num_microbatches = 10
    clip_norm = 0.5
    aggregator = _build_no_privacy_scalar_tree_agg(clip_norm=clip_norm)
    grad_processor = gradient_processors.DPAggregatorBackedGradientProcessor(
        aggregator, l2_norm_clip=clip_norm
    )
    state = grad_processor.init()
    batched_grads = jnp.array([2.0] * num_microbatches)
    state, grad_estimate = grad_processor.apply(state, batched_grads)
    # The incoming values should have been clipped and then averaged.
    self.assertAllClose(grad_estimate, clip_norm)

  def test_only_clips_values_over_norm(self):
    num_microbatches = 10
    clip_norm = 0.5
    aggregator = _build_no_privacy_scalar_tree_agg(clip_norm=clip_norm)
    grad_processor = gradient_processors.DPAggregatorBackedGradientProcessor(
        aggregator, l2_norm_clip=clip_norm
    )
    state = grad_processor.init()
    batched_grads = jnp.array(
        [0.25] * (num_microbatches // 2) + [2.0] * (num_microbatches // 2)
    )
    state, grad_estimate = grad_processor.apply(state, batched_grads)
    expected_mean = (0.25 + 0.5) / 2
    # The incoming values should have been clipped and then averaged.
    self.assertAllClose(grad_estimate, expected_mean)


if __name__ == '__main__':
  tf.test.main()
