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

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_federated as tff

from data_poor_fl import coordinate_aggregators


class CoordinateAggregatorsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('num_coordinates1', 1),
      ('num_coordinates2', 2),
      ('num_coordinates3', 3),
      ('num_coordinates5', 5),
  )
  def test_builds_with_expected_state_length(self, num_coordinates):
    mean_factory = tff.aggregators.MeanFactory()
    mean_aggregator = mean_factory.create(
        value_type=tff.TensorType(tf.float32),
        weight_type=tff.TensorType(tf.float32))
    coordinate_agg = coordinate_aggregators.build_coordinate_aggregator(
        mean_aggregator, num_coordinates=num_coordinates)
    agg_state = coordinate_agg.initialize()
    self.assertLen(agg_state, num_coordinates)

  def test_single_coordinate_matches_base_aggregator(self):
    base_factory = tff.aggregators.MeanFactory()
    base_aggregator = base_factory.create(
        value_type=tff.TensorType(tf.float32),
        weight_type=tff.TensorType(tf.float32))
    coordinate_aggregator = coordinate_aggregators.build_coordinate_aggregator(
        base_aggregator, num_coordinates=1)

    base_state = base_aggregator.initialize()
    coordinate_state = coordinate_aggregator.initialize()
    self.assertAllClose(base_state, coordinate_state[0])

    client_values = [1.5, 3.5, 7.2]
    coordinate_client_values = [[a] for a in client_values]
    client_weights = [1, 3, 2]
    coordinate_client_weights = [[a] for a in client_weights]
    base_output = base_aggregator.next(base_state, client_values,
                                       client_weights)
    coordinate_output = coordinate_aggregator.next(coordinate_state,
                                                   coordinate_client_values,
                                                   coordinate_client_weights)
    self.assertAllClose(base_output.state, coordinate_output.state[0])
    self.assertAllClose(base_output.result, coordinate_output.result[0])
    self.assertAllClose(base_output.measurements,
                        coordinate_output.measurements[0])

  def test_coordinate_mean_aggregator_with_three_coordinates(self):
    mean_factory = tff.aggregators.MeanFactory()
    mean_aggregator = mean_factory.create(
        value_type=tff.TensorType(tf.float32),
        weight_type=tff.TensorType(tf.float32))
    coordinate_agg = coordinate_aggregators.build_coordinate_aggregator(
        mean_aggregator, num_coordinates=3)
    client_values = [[0, 1, 2], [1, 2, 3], [2, 3, 4]]
    client_weights = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

    coordinate_agg_state = coordinate_agg.initialize()
    actual_result = coordinate_agg.next(coordinate_agg_state, client_values,
                                        client_weights).result
    mean_state = mean_aggregator.initialize()
    mean_states = [mean_state, mean_state, mean_state]
    expected_result = [
        mean_aggregator.next(a).result
        for a in zip(mean_states, client_values, client_weights)
    ]
    self.assertEqual(actual_result, expected_result)

  def test_raises_on_unweighted_aggregator(self):
    sum_factory = tff.aggregators.SumFactory()
    sum_aggregator = sum_factory.create(value_type=tff.TensorType(tf.float32))
    with self.assertRaises(ValueError):
      coordinate_aggregators.build_coordinate_aggregator(
          sum_aggregator, num_coordinates=3)


if __name__ == '__main__':
  tf.test.main()
