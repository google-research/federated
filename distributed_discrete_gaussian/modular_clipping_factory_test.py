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
"""Tests for ModularClippingSumFactory."""

import collections

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_federated as tff

from distributed_discrete_gaussian import modular_clipping_factory


DEFAULT_CLIP_LOWER = -2
DEFAULT_CLIP_UPPER = 2

_test_struct_type = [(tf.int32, (3,)), tf.int32]

_int_at_server = tff.type_at_server(tf.int32)
_int_at_clients = tff.type_at_clients(tf.int32)


def _make_test_struct_value(x):
  return [tf.constant(x, dtype=tf.int32, shape=(3,)), x]


def _clipped_sum(clip_lower=DEFAULT_CLIP_LOWER, clip_upper=DEFAULT_CLIP_UPPER):
  return modular_clipping_factory.ModularClippingSumFactory(
      clip_lower, clip_upper, tff.aggregators.SumFactory())


class ModularClippingSumFactoryComputationTest(tff.test.TestCase,
                                               parameterized.TestCase):

  def test_raise_on_invalid_clip_range(self):
    with self.assertRaises(ValueError):
      _ = _clipped_sum(-1, -2)
    with self.assertRaises(ValueError):
      _ = _clipped_sum(3, 2)
    with self.assertRaises(ValueError):
      _ = _clipped_sum(0, 2**31)
    with self.assertRaises(ValueError):
      _ = _clipped_sum(-2**31 - 1, 0)
    with self.assertRaises(ValueError):
      _ = _clipped_sum(-2**30, 2**30 + 5)
    with self.assertRaises(TypeError):
      _ = _clipped_sum(tf.constant(0), tf.constant(1))

  @parameterized.named_parameters(
      ('int', tf.int32),
      ('struct', _test_struct_type))
  def test_clip_type_properties_simple(self, value_type):
    factory = _clipped_sum()
    value_type = tff.to_type(value_type)
    process = factory.create(value_type)

    self.assertIsInstance(process, tff.templates.AggregationProcess)

    server_state_type = tff.type_at_server(())  # Inner SumFactory has no state

    expected_initialize_type = tff.FunctionType(
        parameter=None, result=server_state_type)
    self.assertTrue(
        process.initialize.type_signature.is_equivalent_to(
            expected_initialize_type))

    expected_measurements_type = tff.type_at_server(
        collections.OrderedDict(agg_process=()))

    expected_next_type = tff.FunctionType(
        parameter=collections.OrderedDict(
            state=server_state_type,
            value=tff.type_at_clients(value_type)),
        result=tff.templates.MeasuredProcessOutput(
            state=server_state_type,
            result=tff.type_at_server(value_type),
            measurements=expected_measurements_type))
    self.assertTrue(
        process.next.type_signature.is_equivalent_to(expected_next_type))


class ClippingFactoryExecutionTest(tff.test.TestCase, parameterized.TestCase):

  def _check_result(self, expected, result):
    for exp, res in zip(_make_test_struct_value(expected), result):
      self.assertAllClose(exp, res, atol=0)

  @parameterized.named_parameters([
      ('in_range', -5, 10, [5], [5]),
      ('out_of_range_left', -5, 10, [-15], [0]),
      ('out_of_range_right', -5, 10, [20], [5]),
      ('boundary_left', -5, 10, [-5], [-5]),
      ('boundary_right', -5, 10, [10], [-5]),
      ('all_negative_bound_in', -20, -10, [-15], [-15]),
      ('all_negative_bound_out_left', -20, -10, [-25], [-15]),
      ('all_negative_bound_out_right', -20, -10, [-5], [-15]),
      ('all_positive_bound_in', 20, 40, [30], [30]),
      ('all_positive_bound_out_left', 20, 40, [10], [30]),
      ('all_positive_bound_out_right', 20, 40, [50], [30]),
      ('large_range_symmetric', -2**30, 2**30 - 1, [2**30 + 5], [-2**30 + 6]),
      ('large_range_left', -2**31 + 1, 0, [5], [-2**31 + 6]),
      ('large_range_right', 0, 2**31 - 1, [-5], [2**31 - 6])])
  def test_clip_individual_values(self, clip_range_lower, clip_range_upper,
                                  client_data, expected_sum):
    factory = _clipped_sum(clip_range_lower, clip_range_upper)

    value_type = tff.to_type(tf.int32)
    process = factory.create(value_type)

    state = process.initialize()
    output = process.next(state, client_data)
    self.assertEqual(output.result, expected_sum)

  @parameterized.named_parameters([
      ('in_range_clip', -3, 3, [1, -2, 1, -2], -2),
      ('boundary_clip', -3, 3, [-3, 3, 3, 3], 0),
      ('out_of_range_clip', -2, 2, [-3, 3, 5], 1),
      ('mixed_clip', -2, 2, [-4, -2, 1, 2, 7], 0)])
  def test_clip_sum(self, clip_range_lower, clip_range_upper,
                    client_data, expected_sum):
    factory = _clipped_sum(clip_range_lower, clip_range_upper)

    value_type = tff.to_type(tf.int32)
    process = factory.create(value_type)

    state = process.initialize()
    output = process.next(state, client_data)
    self.assertEqual(output.result, expected_sum)

  @parameterized.named_parameters([
      ('in_range_clip', -3, 3, [1, -2, 1, -2], -2),
      ('boundary_clip', -3, 3, [-3, 3, 3, 3], 0),
      ('out_of_range_clip', -2, 2, [-3, 3, 5], 1),
      ('mixed_clip', -2, 2, [-4, -2, 1, 2, 7], 0)])
  def test_clip_sum_struct(self, clip_range_lower, clip_range_upper,
                           client_data, expected_sum):
    factory = _clipped_sum(clip_range_lower, clip_range_upper)

    value_type = tff.to_type(_test_struct_type)
    process = factory.create(value_type)

    state = process.initialize()
    client_struct_data = [_make_test_struct_value(v) for v in client_data]

    output = process.next(state, client_struct_data)
    self._check_result(expected_sum, output.result)


if __name__ == '__main__':
  tff.test.main()
