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

import collections
from absl.testing import parameterized
import tensorflow as tf
import tensorflow_federated as tff

from compressed_communication.aggregators import group


_measurement = 1.0
_measurement_fn = lambda _: tff.federated_value(_measurement, tff.SERVER)
_measurement_aggregator = tff.aggregators.add_measurements(
    tff.aggregators.SumFactory(), client_measurement_fn=_measurement_fn)


class GroupComputationTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("two_groups",
       dict(
           kernel=[0, 2],
           bias=[1]),
       dict(
           kernel=tff.aggregators.SumFactory(),
           bias=_measurement_aggregator),
       [(tf.float32, (2,)), (tf.float32, (3,)), (tf.float32, (2,))],
       tff.StructType([("kernel", ()), ("bias", ())]),
       tff.StructType([("kernel", ()), ("bias", tf.float32)])))
  def test_group_properties(self, grouped_indices, inner_agg_factories,
                            value_type, states_type, measurements_type):
    factory = group.GroupFactory(grouped_indices, inner_agg_factories)
    value_type = tff.to_type(value_type)
    process = factory.create(value_type)
    self.assertIsInstance(process, tff.templates.AggregationProcess)

    server_state_type = tff.type_at_server(states_type)
    expected_initialize_type = tff.FunctionType(
        parameter=None, result=server_state_type)
    tff.test.assert_types_equivalent(process.initialize.type_signature,
                                     expected_initialize_type)

    expected_measurements_type = tff.type_at_server(measurements_type)
    expected_next_type = tff.FunctionType(
        parameter=collections.OrderedDict(
            state=server_state_type, value=tff.type_at_clients(value_type)),
        result=tff.templates.MeasuredProcessOutput(
            state=server_state_type,
            result=tff.type_at_server(value_type),
            measurements=expected_measurements_type))
    tff.test.assert_types_equivalent(process.next.type_signature,
                                     expected_next_type)

  @parameterized.named_parameters(
      ("group_name_mismatch",
       dict(
           x=[0, 2],
           y=[1]),
       dict(
           kernel=tff.aggregators.SumFactory(),
           bias=_measurement_aggregator),
       [(tf.float32, (2,)), (tf.float32, (3,)), (tf.float32, (2,))]),
      )
  def test_group_init_raises(self, grouped_indices, inner_agg_factories,
                             value_type):
    del value_type  # Unused.
    self.assertRaises(ValueError, group.GroupFactory, grouped_indices,
                      inner_agg_factories)

  @parameterized.named_parameters(
      ("integer_tensors",
       dict(
           kernel=[0, 2],
           bias=[1]),
       dict(
           kernel=tff.aggregators.SumFactory(),
           bias=_measurement_aggregator),
       [(tf.int32, (2,)), (tf.int32, (3,)), (tf.int32, (2,))]),
      ("single_float",
       dict(
           kernel=[0]),
       dict(
           kernel=tff.aggregators.SumFactory()),
       tf.float32),
      )
  def test_group_create_raises(self, grouped_indices, inner_agg_factories,
                               value_type):
    factory = group.GroupFactory(grouped_indices, inner_agg_factories)
    value_type = tff.to_type(value_type)
    self.assertRaises(ValueError, factory.create, value_type)


class GroupExecutionTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ("two_groups",
       dict(
           kernel=[1, 2],
           bias=[0]),
       dict(
           kernel=tff.aggregators.SumFactory(),
           bias=_measurement_aggregator),
       [(tf.float32, (2,)), (tf.float32, (3,)), (tf.float32, (2,))]))
  def test_group_impl_first_permutation(self, grouped_indices,
                                        inner_agg_factories, value_type):

    factory = group.GroupFactory(grouped_indices, inner_agg_factories)
    value_type = tff.to_type(value_type)
    process = factory.create(value_type)
    state = process.initialize()

    client_values = [[tf.ones(t.shape) for t in value_type] for _ in range(2)]
    expected_result = [tf.ones(t.shape) * 2 for t in value_type]

    measurements = process.next(state, client_values).measurements
    self.assertAllEqual(measurements,
                        collections.OrderedDict(kernel=(), bias=_measurement))
    result = process.next(state, client_values).result
    self.assertAllClose(result, expected_result)

  @parameterized.named_parameters(
      ("two_groups",
       dict(
           kernel=[2, 0],
           bias=[1]),
       dict(
           kernel=tff.aggregators.SumFactory(),
           bias=_measurement_aggregator),
       [(tf.float32, (2,)), (tf.float32, (3,)), (tf.float32, (2,))]))
  def test_group_impl_second_permutation(self, grouped_indices,
                                         inner_agg_factories, value_type):

    factory = group.GroupFactory(grouped_indices, inner_agg_factories)
    value_type = tff.to_type(value_type)
    process = factory.create(value_type)
    state = process.initialize()

    client_values = [[tf.ones(t.shape) for t in value_type] for _ in range(2)]
    expected_result = [tf.ones(t.shape) * 2 for t in value_type]

    measurements = process.next(state, client_values).measurements
    self.assertAllEqual(measurements,
                        collections.OrderedDict(kernel=(), bias=_measurement))
    result = process.next(state, client_values).result
    self.assertAllClose(result, expected_result)

  @parameterized.named_parameters(
      ("two_groups",
       dict(
           kernel=[0, 1],
           bias=[2]),
       dict(
           kernel=tff.aggregators.SumFactory(),
           bias=_measurement_aggregator),
       [(tf.float32, (2,)), (tf.float32, (3,)), (tf.float32, (2,))]))
  def test_group_impl_third_permutation(self, grouped_indices,
                                        inner_agg_factories, value_type):

    factory = group.GroupFactory(grouped_indices, inner_agg_factories)
    value_type = tff.to_type(value_type)
    process = factory.create(value_type)
    state = process.initialize()

    client_values = [[tf.ones(t.shape) for t in value_type] for _ in range(2)]
    expected_result = [tf.ones(t.shape) * 2 for t in value_type]

    measurements = process.next(state, client_values).measurements
    self.assertAllEqual(measurements,
                        collections.OrderedDict(kernel=(), bias=_measurement))
    result = process.next(state, client_values).result
    self.assertAllClose(result, expected_result)


if __name__ == "__main__":
  tf.test.main()
