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

from personalization_benchmark.cross_device.algorithms import hypcluster_utils

MODEL_WEIGHTS_TYPE = tff.type_at_server(
    tff.to_type(tff.learning.ModelWeights(tf.float32, ())))


class CoordinateFinalizersTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('num_coordinates1', 1),
      ('num_coordinates2', 2),
      ('num_coordinates3', 3),
      ('num_coordinates5', 5),
  )
  def test_build_with_expected_state_length(self, num_coordinates):
    server_optimizer = tff.learning.optimizers.build_sgdm()
    base_finalizer = tff.learning.templates.build_apply_optimizer_finalizer(
        server_optimizer, MODEL_WEIGHTS_TYPE.member)
    finalizer = hypcluster_utils.build_coordinate_finalizer(
        base_finalizer, num_coordinates=num_coordinates)
    state = finalizer.initialize()
    self.assertLen(state, num_coordinates)

  def test_single_coordinate_matches_base_finalizer(self):
    server_optimizer = tff.learning.optimizers.build_sgdm()
    base_finalizer = tff.learning.templates.build_apply_optimizer_finalizer(
        server_optimizer, MODEL_WEIGHTS_TYPE.member)
    coordinate_finalizer = hypcluster_utils.build_coordinate_finalizer(
        base_finalizer, num_coordinates=1)

    base_state = base_finalizer.initialize()
    coordinate_state = coordinate_finalizer.initialize()
    self.assertAllClose(base_state, coordinate_state[0])

    weights = tff.learning.ModelWeights(1.0, ())
    update = 0.1
    base_output = base_finalizer.next(base_state, weights, update)
    coordinate_output = coordinate_finalizer.next(coordinate_state, [weights],
                                                  [update])
    self.assertAllClose(base_output.state, coordinate_output.state[0])
    self.assertAllClose(base_output.result.trainable,
                        coordinate_output.result[0].trainable)
    self.assertAllClose(base_output.measurements,
                        coordinate_output.measurements[0])

  def test_coordinate_finalizer_with_three_coordinates(self):
    server_optimizer = tff.learning.optimizers.build_sgdm()
    base_finalizer = tff.learning.templates.build_apply_optimizer_finalizer(
        server_optimizer, MODEL_WEIGHTS_TYPE.member)
    coordinate_finalizer = hypcluster_utils.build_coordinate_finalizer(
        base_finalizer, num_coordinates=3)
    weights = [
        tff.learning.ModelWeights(1.0, ()),
        tff.learning.ModelWeights(2.0, ()),
        tff.learning.ModelWeights(3.0, ())
    ]
    updates = [4.0, 5.0, 6.0]

    coordinate_state = coordinate_finalizer.initialize()
    coordinate_output = coordinate_finalizer.next(coordinate_state, weights,
                                                  updates)
    actual_result = coordinate_output.result
    base_state = base_finalizer.initialize()
    list_of_base_state = [base_state, base_state, base_state]
    expected_result = [
        base_finalizer.next(a).result
        for a in zip(list_of_base_state, weights, updates)
    ]

    for a, b in zip(actual_result, expected_result):
      self.assertAllClose(a.trainable, b.trainable)


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
    coordinate_agg = hypcluster_utils.build_coordinate_aggregator(
        mean_aggregator, num_coordinates=num_coordinates)
    agg_state = coordinate_agg.initialize()
    self.assertLen(agg_state, num_coordinates)

  def test_single_coordinate_matches_base_aggregator(self):
    base_factory = tff.aggregators.MeanFactory()
    base_aggregator = base_factory.create(
        value_type=tff.TensorType(tf.float32),
        weight_type=tff.TensorType(tf.float32))
    coordinate_aggregator = hypcluster_utils.build_coordinate_aggregator(
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
    coordinate_agg = hypcluster_utils.build_coordinate_aggregator(
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
      hypcluster_utils.build_coordinate_aggregator(
          sum_aggregator, num_coordinates=3)


def create_dataset():
  # Create data satisfying y = x + 1
  x = [[1.0], [2.0], [3.0]]
  y = [[2.0], [3.0], [4.0]]
  return tf.data.Dataset.from_tensor_slices((x, y)).batch(1)


def get_input_spec():
  return create_dataset().element_spec


def model_fn(initializer='zeros'):
  keras_model = tf.keras.Sequential([
      tf.keras.layers.Dense(
          1,
          kernel_initializer=initializer,
          bias_initializer=initializer,
          input_shape=(1,))
  ])
  return tff.learning.from_keras_model(
      keras_model=keras_model,
      input_spec=get_input_spec(),
      loss=tf.keras.losses.MeanSquaredError())


@tff.tf_computation
def create_nested_structure():
  return [
      dict(a=tf.zeros((2, 2), dtype=tf.int32), b=1, c=3),
      dict(a=tf.ones((2, 2), dtype=tf.int32), b=2, c=4),
      dict(a=2 * tf.ones((2, 2), dtype=tf.int32), b=3, c=5),
  ]


def create_initial_models(num_models: int):
  model = model_fn(initializer='ones')
  model_weights_tensors = tf.nest.map_structure(
      lambda var: var.numpy(), tff.learning.ModelWeights.from_model(model))
  return [model_weights_tensors for _ in range(num_models)]


class GatherTest(tf.test.TestCase):

  def test_gather_type_signature(self):
    list_type = create_nested_structure.type_signature.result
    list_element_type = tff.to_type(list_type[0])
    gather_fn = hypcluster_utils.build_gather_fn(
        list_element_type, num_indices=3)
    self.assertEqual(gather_fn.type_signature.parameter[0], list_type)
    self.assertEqual(gather_fn.type_signature.parameter[1],
                     tff.TensorType(dtype=tf.int32))
    self.assertEqual(gather_fn.type_signature.result, list_element_type)

  def test_gather_on_list_of_tensors(self):
    list_element_type = tff.TensorType(dtype=tf.int32)
    gather_fn = hypcluster_utils.build_gather_fn(
        list_element_type, num_indices=5)
    gather_structure = [5, 1, 16, -1, 10042]
    for i in range(5):
      actual_result = gather_fn(gather_structure, i)
      expected_result = gather_structure[i]
      self.assertAllEqual(actual_result, expected_result)

  def test_gather_on_nested_structure(self):
    list_type = create_nested_structure.type_signature.result
    list_element_type = tff.to_type(list_type[0])
    gather_fn = hypcluster_utils.build_gather_fn(
        list_element_type, num_indices=3)
    gather_structure = create_nested_structure()
    for i in range(3):
      actual_result = gather_fn(gather_structure, i)
      expected_result = gather_structure[i]
      self.assertDictEqual(actual_result, expected_result)


class ScatterTest(tf.test.TestCase):

  def test_scatter_type_signature(self):
    list_type = create_nested_structure.type_signature.result
    list_element_type = tff.to_type(list_type[0])
    scatter_fn = hypcluster_utils.build_scatter_fn(
        list_element_type, num_indices=3)
    self.assertEqual(scatter_fn.type_signature.parameter[0], list_element_type)
    self.assertEqual(scatter_fn.type_signature.parameter[1],
                     tff.TensorType(tf.int32))
    self.assertEqual(scatter_fn.type_signature.parameter[2],
                     tff.TensorType(tf.float32))
    self.assertEqual(scatter_fn.type_signature.result[0], list_type)
    expected_result_weight_type = tff.StructWithPythonType(
        [tff.TensorType(tf.float32)] * 3, list)
    self.assertEqual(scatter_fn.type_signature.result[1],
                     expected_result_weight_type)

  def test_scatter_tensor(self):
    value_type = tff.TensorType(dtype=tf.int32)
    scatter_fn = hypcluster_utils.build_scatter_fn(value_type, num_indices=5)
    for i in range(5):
      actual_value, actual_weight = scatter_fn(7, i, 0.5)
      expected_value = [0] * 5
      expected_value[i] = 7
      self.assertEqual(actual_value, expected_value)
      expected_weight = [0.0] * 5
      expected_weight[i] = 0.5
      self.assertEqual(actual_weight, expected_weight)

  def test_scatter_nested_structure(self):
    list_type = create_nested_structure.type_signature.result
    list_element_type = tff.to_type(list_type[0])
    nested_structure = create_nested_structure()
    scatter_fn = hypcluster_utils.build_scatter_fn(
        list_element_type, num_indices=2)
    actual_value, actual_weight = scatter_fn(nested_structure[1], 0, 3.0)
    expected_value = [
        nested_structure[1],
        dict(a=tf.zeros((2, 2), dtype=tf.int32), b=0, c=0)
    ]
    for actual_dict, expected_dict in zip(actual_value, expected_value):
      self.assertDictEqual(actual_dict, expected_dict)
    expected_weight = [3.0, 0.0]
    self.assertEqual(actual_weight, expected_weight)


if __name__ == '__main__':
  tf.test.main()
