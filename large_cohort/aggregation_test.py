# Copyright 2021, Google LLC.
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

import tensorflow as tf
import tensorflow_federated as tff

from large_cohort import aggregation


class AggregationTest(tf.test.TestCase):

  def test_experiment_aggregator_matches_tff_learning(self):
    test_model_weights = tff.to_type(
        collections.OrderedDict(
            w=tff.TensorType(shape=[10], dtype=tf.float32),
            b=tff.TensorType(shape=[1], dtype=tf.float32)))
    experiment_aggregation_process = aggregation.create_aggregator().create(
        test_model_weights, weight_type=tff.TensorType(tf.float32))
    learning_aggregation_process = tff.learning.robust_aggregator().create(
        test_model_weights, weight_type=tff.TensorType(tf.float32))
    # `initialize` function should be identical.
    tff.test.assert_types_identical(
        experiment_aggregation_process.initialize.type_signature,
        learning_aggregation_process.initialize.type_signature)
    # `next` is slightly different because of measurements, so we assert
    # piecewise.
    tff.test.assert_types_identical(
        experiment_aggregation_process.next.type_signature.parameter,
        learning_aggregation_process.next.type_signature.parameter)
    experiment_result_type = experiment_aggregation_process.next.type_signature.result
    learning_result_type = learning_aggregation_process.next.type_signature.result
    tff.test.assert_types_identical(experiment_result_type.state,
                                    learning_result_type.state)
    tff.test.assert_types_identical(experiment_result_type.result,
                                    learning_result_type.result)
    tff.test.assert_types_identical(
        experiment_result_type.measurements.member.zeroing.clipping,
        tff.to_type(
            collections.OrderedDict(
                mean_of_norm_of_client_update=tf.float32,
                norm_of_mean_client_update=tf.float32,
                average_cosine_similarity=tf.float32,
                mean_value=(),
                mean_weight=())))

  def test_mean_computation(self):
    aggregation_process = aggregation.create_aggregator(
        clipping=False, zeroing=False).create(
            tff.TensorType(shape=[3], dtype=tf.float32),
            weight_type=tff.TensorType(tf.float32))
    state = aggregation_process.initialize()
    client_values = [[1, 2, 7], [2, 4, 2], [-3, 0, -3]]
    client_weights = [1.0, 2.0, 3.0]
    output = aggregation_process.next(state, client_values, client_weights)

    @tff.federated_computation(
        tff.type_at_clients(tff.TensorType(shape=[3], dtype=tf.float32)),
        tff.type_at_clients(tff.TensorType(tf.float32)))
    def test_mean(value, weight):
      return tff.federated_mean(value, weight)

    expected_output = test_mean(client_values, client_weights)
    self.assertAllClose(output.result, expected_output)

  def test_measurements_output(self):
    aggregation_process = aggregation.create_aggregator(
        clipping=False, zeroing=False).create(
            tff.TensorType(shape=[3], dtype=tf.float32),
            weight_type=tff.TensorType(tf.float32))
    state = aggregation_process.initialize()
    client_values = [[1, 2, 7], [2, 4, 2], [-3, 0, -3]]
    output = aggregation_process.next(
        state,
        client_values,
        [1, 1, 1]  # uniform weighting
    )
    self.assertAllClose(output.result, [0, 2, 2])
    self.assertAllClose(
        output.measurements,
        collections.OrderedDict(
            mean_of_norm_of_client_update=5.496,
            norm_of_mean_client_update=2.828,
            average_cosine_similarity=-0.227,
            mean_value=(),
            mean_weight=()),
        atol=1e-3)

  def test_measurements_with_orthogonal_vectors(self):
    aggregation_process = aggregation.create_aggregator(
        clipping=False, zeroing=False).create(
            tff.TensorType(shape=[4], dtype=tf.float32),
            weight_type=tff.TensorType(tf.float32))
    state = aggregation_process.initialize()
    client_values = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    output = aggregation_process.next(
        state,
        client_values,
        [1, 1, 1, 1]  # uniform weighting
    )
    self.assertAllClose(output.result, [1 / 4, 1 / 4, 1 / 4, 1 / 4])
    self.assertAllClose(
        output.measurements,
        collections.OrderedDict(
            mean_of_norm_of_client_update=1.0,
            norm_of_mean_client_update=0.5,
            average_cosine_similarity=0,
            mean_value=(),
            mean_weight=()),
        atol=1e-3)

  def test_measurements_with_parallel_vectors(self):
    aggregation_process = aggregation.create_aggregator(
        clipping=False, zeroing=False).create(
            tff.TensorType(shape=[3], dtype=tf.float32),
            weight_type=tff.TensorType(tf.float32))
    state = aggregation_process.initialize()
    client_values = [[1, 0, 0], [1, 0, 0], [1, 0, 0]]
    output = aggregation_process.next(
        state,
        client_values,
        [3, 4, 6],  # The weighting is irrelevant since clients are identical.
    )
    self.assertAllClose(output.result, [1, 0, 0])
    self.assertAllClose(
        output.measurements,
        collections.OrderedDict(
            mean_of_norm_of_client_update=1.0,
            norm_of_mean_client_update=1.0,
            average_cosine_similarity=1.0,
            mean_value=(),
            mean_weight=()),
        atol=1e-3)


if __name__ == '__main__':
  tf.test.main()
