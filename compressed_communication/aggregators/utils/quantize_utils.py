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
"""Utilities for quantizing a float tensor to an integer tensor."""

import tensorflow as tf


# Normalization
def mean_magnitude(value):
  return tf.reduce_mean(tf.abs(value))


def max_magnitude(value):
  return tf.reduce_max(tf.abs(value))


def dimensionless_norm(value):
  return tf.sqrt(tf.reduce_mean(tf.square(value)))


# Uniform Quantization
def uniform_quantize(value, step_size, seed):
  """Scales and rounds `value` by provided `step_size`."""
  del seed  # Unused.
  return tf.cast(tf.round(tf.divide(value, step_size)), tf.int32)


def uniform_dequantize(value, step_size, noise_sum):
  """Unscales quantized `value` by provided `step_size`."""
  del noise_sum  # Unused.
  return tf.cast(value, tf.float32) * step_size


# Stochastic Quantization
def stochastic_quantize(value, step_size, seed):
  """Scales `value` by provided `step_size` and randomly rounds each element."""
  scaled = tf.cast(tf.divide(value, step_size), tf.float32)
  prob = scaled - tf.cast(tf.floor(scaled), tf.float32)
  random = tf.random.stateless_uniform(value.shape, seed=seed, dtype=tf.float32)
  rounded = tf.where(
      tf.less_equal(random, prob), tf.math.ceil(scaled), tf.math.floor(scaled))
  return tf.cast(rounded, tf.int32)


# Dithered Quantization
def generate_noise(seed, shape):
  return tf.random.stateless_uniform(
      shape, minval=-0.5, maxval=0.5, seed=seed, dtype=tf.float32)


def dithered_quantize(value, step_size, seed):
  """Scales `value` by provided `step_size`, adds random noise and rounds."""
  scaled = tf.cast(tf.divide(value, step_size), tf.float32)
  noise = generate_noise(seed, value.shape)
  return tf.cast(tf.round(scaled - noise), tf.int32)


def dithered_dequantize(value, step_size, noise_sum):
  """Removes noise and unscales summed quantized `value` by provided `step_size`.

  Requires that `noise_sum` is generated using the same random seeds used to
  quantize each of the values added together to produce `value`.

  Args:
    value: The sum of several noised, scaled, and quantized inputs.
    step_size: The quantization step size used to scale the result.
    noise_sum: The sum of all the noise values used to produce the inputs to
      `value`.

  Returns:
    A de-quantized `value`.
  """
  return (tf.cast(value, tf.float32) + noise_sum) * step_size


# Quantization Step Size Decay Schedules
def linear_decay(initial_value, min_value, round_num, total_rounds):
  delta = tf.cast(round_num, tf.float32) / tf.cast(total_rounds, tf.float32) * (
      initial_value - min_value)
  return tf.maximum(initial_value - delta, min_value)


def exponential_decay(initial_value, min_value, round_num, exp):
  return (initial_value - min_value) * tf.exp(-round_num * exp) + min_value


def step_decay(initial_value, min_value, round_num, freq):
  return tf.maximum(initial_value * 0.5**(tf.floor(round_num / freq)),
                    min_value)
