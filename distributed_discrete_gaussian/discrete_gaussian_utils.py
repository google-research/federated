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
"""Util functions for drawing discrete Gaussian samples.

The following functions implement a vectorized TF version of the sampling
algorithm described in the paper:

The Discrete Gaussian for Differential Privacy
https://arxiv.org/pdf/2004.00010.pdf

Note that the exact sampling implementation should use integer and fractional
parameters only. Here, for experimental purposes, we relax this constraint a bit
and use vectorized implementations of Bernoulli and discrete Laplace sampling
that can take float parameters.
"""

import tensorflow as tf
import tensorflow_probability as tf_prob


def _sample_discrete_laplace(t, shape):
  """Sample from discrete Laplace with scale t.

  This method is based on the observation that sampling from Z ~ Lap(t) is
  equivalent to sampling X, Y independently from Geo(1 - exp(-1/t)) and take
  Z = X - Y.

  Note also that tensorflow_probability's geometric sampler is based on floating
  operations and may possibly be inexact.

  Args:
    t: The scale of the discrete Laplace distribution.
    shape: The tensor shape of the tensors drawn.

  Returns:
    A tensor of the specified shape filled with random values.
  """
  geometric_probs = 1.0 - tf.exp(-1.0 / tf.cast(t, tf.float64))
  geo1 = tf_prob.distributions.Geometric(probs=geometric_probs).sample(shape)
  geo2 = tf_prob.distributions.Geometric(probs=geometric_probs).sample(shape)
  return tf.cast(geo1 - geo2, tf.int64)


def _sample_bernoulli(p):
  """Sample from Bernoulli(p)."""
  return tf_prob.distributions.Bernoulli(probs=p, dtype=tf.int64).sample()


def _check_input_args(scale, shape, dtype):
  """Checks the input args to the discrete Gaussian sampler."""
  if tf.as_dtype(dtype) not in (tf.int32, tf.int64):
    raise ValueError(
        f'Only tf.int32 and tf.int64 are supported. Found dtype `{dtype}`.')

  checks = [
      tf.compat.v1.assert_non_negative(scale),
      tf.compat.v1.assert_integer(scale)
  ]
  with tf.control_dependencies(checks):
    return tf.identity(scale), shape, dtype


def _sample_discrete_gaussian_helper(scale, shape, dtype):
  """Draw samples from discrete Gaussian, assuming scale >= 0."""
  scale = tf.cast(scale, tf.int64)
  sq_scale = tf.square(scale)

  # Do rejection sampling by oversampling.
  oversample_factor = 2
  # Draw at least some samples in case we got unlucky with small input shape.
  min_n = tf.cast(1000, tf.int64)
  target_n = tf.reduce_prod(tf.cast(shape, tf.int64))
  draw_n = tf.maximum(min_n, oversample_factor * target_n)

  # Scale for discrete Laplace.
  t = tf.cast(scale, tf.int64) + 1

  def draw_samples(inp_samples, inp_accept):
    """Sample with rejection."""
    y = _sample_discrete_laplace(t, shape=(draw_n,))
    z_numer = tf.pow((tf.abs(y) * t - sq_scale), 2)
    z_denom = 2 * sq_scale * t * t
    bern_probs = tf.exp(-tf.cast(z_numer, tf.float64) /
                        tf.cast(z_denom, tf.float64))
    accept = _sample_bernoulli(bern_probs)
    # Outputs from previous iterations are only used for restoring shapes.
    y.set_shape(inp_samples.get_shape())
    accept.set_shape(inp_accept.get_shape())
    return [y, accept]

  # Retry in the (extremely unlikely) case that oversampling doesn't suffice.
  samples = tf.zeros((draw_n,), dtype=tf.int64)
  accept = tf.zeros((draw_n,), dtype=tf.int64)
  samples, accept = tf.while_loop(
      cond=lambda _, accept: tf.reduce_sum(accept) < target_n,
      body=draw_samples,
      loop_vars=[samples, accept])

  accepted_samples = samples[tf.equal(accept, 1)][:target_n]
  return tf.cast(tf.reshape(accepted_samples, shape), dtype)


def sample_discrete_gaussian(scale, shape, dtype=tf.int32):
  """Draws (possibly inexact) samples from the discrete Gaussian distribution.

  We relax some integer constraints to use vectorized implementations of
  Bernoulli and discrete Laplace sampling. Integer operations are done in
  tf.int64 as TF does not have direct support for fractions.

  Args:
    scale: The scale of the discrete Gaussian distribution.
    shape: The shape of the output tensor.
    dtype: The type of the output.

  Returns:
    A tensor of the specified shape filled with random values.
  """
  scale, shape, dtype = _check_input_args(scale, shape, dtype)
  return tf.cond(
      tf.equal(scale, 0), lambda: tf.zeros(shape, dtype),
      lambda: _sample_discrete_gaussian_helper(scale, shape, dtype))
