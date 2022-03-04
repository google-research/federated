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

from absl.testing import absltest
import tensorflow as tf

from compressed_communication.aggregators.utils import quantize_utils


_seed = tf.stack([0, 0])
_q_step_size = 0.4


class NormalizeUtilsTest(absltest.TestCase):

  def test_mean_magnitude(self):
    value = tf.constant([0.0, 1.0, 2.0])
    mean_magnitude = quantize_utils.mean_magnitude(value)
    self.assertAlmostEqual(mean_magnitude, (0.0 + 1.0 + 2.0) / 3.0)

  def test_max_magnitude(self):
    value = tf.constant([0.0, 1.0, 2.0])
    max_magnitude = quantize_utils.max_magnitude(value)
    self.assertAlmostEqual(max_magnitude, 2.0)

  def test_dimensionless_norm(self):
    value = tf.constant([0.0, 1.0, 2.0])
    dimensionless_norm = quantize_utils.dimensionless_norm(value)
    self.assertAlmostEqual(dimensionless_norm,
                           tf.sqrt((0.0 + 1.0 + 4.0) / 3.0))


class UniformQuantizeUtilsTest(absltest.TestCase):

  def test_quantize(self):
    value = tf.ones((3,), tf.float32) * 2.0
    quantized = quantize_utils.uniform_quantize(value,
                                                _q_step_size,
                                                _seed)
    expected_quantized = tf.ones((3,), tf.int32) * 5
    self.assertSequenceAlmostEqual(quantized, expected_quantized)

  def test_dequantize(self):
    value = tf.ones((3,), tf.int32) * 5
    noise = quantize_utils.generate_noise(_seed, value.shape)
    dequantized = quantize_utils.uniform_dequantize(value, _q_step_size, noise)
    expected_dequantized = tf.ones((3,), tf.float32) * 2.0
    self.assertSequenceAlmostEqual(dequantized, expected_dequantized)


class StochasticQuantizeUtilsTest(absltest.TestCase):

  def test_diff_random_seeds_diff_result(self):
    value = tf.random.uniform(
        (1000,), minval=-5.0, maxval=5.0, dtype=tf.float32)
    quantized = quantize_utils.stochastic_quantize(value,
                                                   _q_step_size,
                                                   _seed)
    requantized = quantize_utils.stochastic_quantize(value,
                                                     _q_step_size,
                                                     _seed + 1)
    self.assertFalse(tf.reduce_all(tf.equal(quantized, requantized)))

  def test_noop_quantize(self):
    value = tf.cast(tf.random.uniform(
        (3,), minval=-5, maxval=5, dtype=tf.int32), tf.float32)
    quantized = quantize_utils.stochastic_quantize(value, 1.0, _seed)
    self.assertSequenceAlmostEqual(value, tf.cast(quantized, tf.float32))

  def test_quantized_to_valid_value(self):
    value = tf.random.uniform(
        (3,), minval=-5.0, maxval=5.0, dtype=tf.float32)
    floor = tf.cast(tf.math.floor(value / _q_step_size), tf.int32)
    ceil = tf.cast(tf.math.ceil(value / _q_step_size), tf.int32)
    quantized = quantize_utils.stochastic_quantize(value, _q_step_size, _seed)
    for i in range(value.shape[0]):
      self.assertIn(quantized[i], (floor[i], ceil[i]))

  def test_expected_stochasticity(self):
    zeros = tf.zeros((1000,), dtype=tf.float32)
    round_down = quantize_utils.stochastic_quantize(zeros, _q_step_size, _seed)
    self.assertTrue(tf.reduce_all(tf.equal(round_down,
                                           tf.cast(zeros, tf.int32))))
    ones = tf.zeros((1000,), dtype=tf.float32)
    round_up = quantize_utils.stochastic_quantize(ones, 0.9999, _seed)
    self.assertTrue(tf.reduce_all(tf.equal(round_up, tf.cast(ones, tf.int32))))


class DitheredQuantizeUtilsTest(absltest.TestCase):

  def test_diff_random_seeds_diff_result(self):
    value = tf.random.uniform(
        (1000,), minval=-5.0, maxval=5.0, dtype=tf.float32)
    quantized = quantize_utils.dithered_quantize(
        value, _q_step_size, _seed)
    requantized = quantize_utils.dithered_quantize(
        value, _q_step_size, _seed + 1)
    self.assertFalse(tf.reduce_all(tf.equal(quantized, requantized)))

  def test_noop_quantize(self):
    value = tf.cast(tf.random.uniform(
        (3,), minval=-5, maxval=5, dtype=tf.int32), tf.float32)
    quantized = quantize_utils.dithered_quantize(value, 1.0, _seed)
    self.assertSequenceAlmostEqual(value, tf.cast(quantized, tf.float32))

  def test_quantized_to_valid_value(self):
    value = tf.random.uniform(
        (3,), minval=-5.0, maxval=5.0, dtype=tf.float32)
    floor = tf.cast(tf.math.floor(value / _q_step_size), tf.int32)
    ceil = tf.cast(tf.math.ceil(value / _q_step_size), tf.int32)
    quantized = quantize_utils.dithered_quantize(value, _q_step_size, _seed)
    for i in range(value.shape[0]):
      self.assertIn(quantized[i], (floor[i], ceil[i]))

  def test_dequantize(self):
    value = tf.random.uniform(
        (3,), minval=-5.0, maxval=5.0, dtype=tf.float32)
    quantized = quantize_utils.dithered_quantize(
        value, _q_step_size, _seed)
    noise = quantize_utils.generate_noise(_seed, value.shape)
    dequantized = quantize_utils.dithered_dequantize(quantized, _q_step_size,
                                                     noise)
    self.assertLessEqual(tf.reduce_max(tf.abs(dequantized - value)),
                         0.5 * _q_step_size)

  def test_dequantize_summed_value(self):
    num_values = 2
    values = [
        tf.random.uniform((3,), minval=-5.0, maxval=5.0, dtype=tf.float32)
        for _ in range(num_values)
    ]
    summed_values = tf.reduce_sum(values, axis=0)
    quantized_values = [quantize_utils.dithered_quantize(
        x, _q_step_size, _seed) for x in values]
    summed_quantized = tf.reduce_sum(quantized_values, axis=0)
    noise = quantize_utils.generate_noise(_seed, values[0].shape)
    summed_dequantized = quantize_utils.dithered_dequantize(
        summed_quantized, _q_step_size, noise * num_values)
    self.assertLessEqual(
        tf.reduce_max(tf.abs(summed_dequantized - summed_values)),
        0.5 * _q_step_size * num_values)


class DecayScheduleUtilsTest(absltest.TestCase):

  def test_linear_decay(self):
    initial_value = 2.
    min_value = 0.
    total_rounds = 4
    decayed_value = []
    for round_num in [0, 1, 2, 3]:
      decayed_value.append(quantize_utils.linear_decay(
          initial_value, min_value, round_num, total_rounds))
    self.assertEqual(decayed_value, [2., 1.5, 1., 0.5])

  def test_exponential_decay(self):
    initial_value = 2.
    min_value = 0.
    exp = 1.
    decayed_value = []
    for round_num in [0, 1, 2, 3]:
      decayed_value.append(quantize_utils.exponential_decay(
          initial_value, min_value, round_num, exp))
    self.assertEqual(decayed_value,
                     [2., 2. * tf.exp(-1.), 2. * tf.exp(-2.), 2. * tf.exp(-3.)])

  def test_step_decay(self):
    initial_value = 2.
    min_value = 0.
    freq = 2
    decayed_value = []
    for round_num in [0, 1, 2, 3]:
      decayed_value.append(quantize_utils.step_decay(
          initial_value, min_value, round_num, freq))
    self.assertEqual(decayed_value, [2., 2., 1., 1.])


if __name__ == "__main__":
  absltest.main()
