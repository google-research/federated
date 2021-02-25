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

from optimization.shared import evaluation


def keras_model_fn():
  # Create a simple linear regression model, single output.
  # We initialize at all zeros, so that we can exactly compute mean squared
  # errors for testing purposes.
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(
          1,
          kernel_initializer='zeros',
          bias_initializer='zeros',
          input_shape=(1,))
  ])
  return model


def create_balanced_client_datasets(client_id):
  if client_id == 0:
    # MSE when all weights are zero: 0.0
    x = [[1.0], [2.0], [3.0]]
    y = [[0.0], [0.0], [0.0]]
  elif client_id == 1:
    # MSE when all weights are zero: 1.75
    x = [[1.0], [2.0], [3.0]]
    y = [[1.0], [2.0], [0.5]]
  return tf.data.Dataset.from_tensor_slices(collections.OrderedDict(
      x=x, y=y)).batch(1)


def create_unbalanced_client_datasets(client_id):
  if client_id == 0:
    # MSE when all weights are zero: 0.0
    x = [[1.0], [2.0]]
    y = [[0.0], [0.0]]
  elif client_id == 1:
    # MSE when all weights are zero: 3.0
    x = [[1.0], [2.0], [3.0]]
    y = [[0.0], [0.0], [3.0]]
  return tf.data.Dataset.from_tensor_slices(collections.OrderedDict(
      x=x, y=y)).batch(1)


def create_client_data(balanced=True):
  if balanced:
    create_dataset_fn = create_balanced_client_datasets
  else:
    create_dataset_fn = create_unbalanced_client_datasets
  return [create_dataset_fn(0), create_dataset_fn(1)]


def get_input_spec():
  return create_balanced_client_datasets(0).element_spec


def tff_model_fn():
  return tff.learning.from_keras_model(
      keras_model=keras_model_fn(),
      input_spec=get_input_spec(),
      loss=tf.keras.losses.MeanSquaredError())


class FederatedEvalTest(tf.test.TestCase):

  def test_eval_metrics_have_correct_structure(self):
    client_data = [create_balanced_client_datasets(0)]
    metrics_builder = lambda: [tf.keras.metrics.MeanSquaredError()]
    eval_fn = evaluation.build_federated_evaluation(tff_model_fn,
                                                    metrics_builder)
    model_weights = tff.learning.ModelWeights.from_model(tff_model_fn())
    eval_metrics = eval_fn(model_weights, client_data)

    aggregation_keys = [x.value for x in evaluation.AggregationMethods]
    self.assertCountEqual(eval_metrics.keys(), aggregation_keys)

    metric_names = ['mean_squared_error', 'num_examples']
    for aggr in aggregation_keys:
      self.assertCountEqual(eval_metrics[aggr].keys(), metric_names)

  def test_mean_aggregations_equal_with_one_client(self):
    client_data = [create_unbalanced_client_datasets(1)]
    metrics_builder = lambda: [tf.keras.metrics.MeanSquaredError()]
    eval_fn = evaluation.build_federated_evaluation(tff_model_fn,
                                                    metrics_builder)
    model_weights = tff.learning.ModelWeights.from_model(tff_model_fn())
    eval_metrics = eval_fn(model_weights, client_data)

    expected_uniform_metrics = collections.OrderedDict(
        mean_squared_error=3.0, num_examples=3.0)
    expected_example_metrics = expected_uniform_metrics
    expected_metrics = collections.OrderedDict(
        example_weighted=expected_example_metrics,
        uniform_weighted=expected_uniform_metrics)
    self.assertAllClose(eval_metrics, expected_metrics)

  def test_mean_metrics_equal_for_balanced_client_data(self):
    client_data = create_client_data(balanced=True)
    metrics_builder = lambda: [tf.keras.metrics.MeanSquaredError()]
    eval_fn = evaluation.build_federated_evaluation(tff_model_fn,
                                                    metrics_builder)
    model_weights = tff.learning.ModelWeights.from_model(tff_model_fn())
    eval_metrics = eval_fn(model_weights, client_data)

    expected_uniform_metrics = collections.OrderedDict(
        mean_squared_error=0.875, num_examples=3.0)
    expected_example_metrics = expected_uniform_metrics
    expected_metrics = collections.OrderedDict(
        example_weighted=expected_example_metrics,
        uniform_weighted=expected_uniform_metrics)
    self.assertAllClose(eval_metrics, expected_metrics)

  def test_mean_metrics_for_unbalanced_client_data(self):
    client_data = create_client_data(balanced=False)
    metrics_builder = lambda: [tf.keras.metrics.MeanSquaredError()]
    eval_fn = evaluation.build_federated_evaluation(tff_model_fn,
                                                    metrics_builder)
    model_weights = tff.learning.ModelWeights.from_model(tff_model_fn())
    eval_metrics = eval_fn(model_weights, client_data)

    expected_uniform_metrics = collections.OrderedDict(
        mean_squared_error=1.5, num_examples=2.5)
    expected_example_metrics = collections.OrderedDict(
        mean_squared_error=1.8, num_examples=2.6)
    expected_metrics = collections.OrderedDict(
        example_weighted=expected_example_metrics,
        uniform_weighted=expected_uniform_metrics)
    self.assertAllClose(eval_metrics, expected_metrics)

  def test_eval_fn_has_correct_type_signature(self):
    metrics_builder = lambda: [tf.keras.metrics.MeanSquaredError()]
    eval_fn = evaluation.build_federated_evaluation(tff_model_fn,
                                                    metrics_builder)
    actual_type = eval_fn.type_signature

    model_type = tff.FederatedType(
        tff.learning.ModelWeights(
            trainable=(
                tff.TensorType(tf.float32, [1, 1]),
                tff.TensorType(tf.float32, [1]),
            ),
            non_trainable=(),
        ), tff.SERVER)

    dataset_type = tff.FederatedType(
        tff.SequenceType(
            collections.OrderedDict(
                x=tff.TensorType(tf.float32, [None, 1]),
                y=tff.TensorType(tf.float32, [None, 1]))), tff.CLIENTS)

    metrics_type = tff.FederatedType(
        collections.OrderedDict(
            mean_squared_error=tff.TensorType(tf.float32),
            num_examples=tff.TensorType(tf.float32)), tff.SERVER)

    aggregated_metrics_type = collections.OrderedDict(
        example_weighted=metrics_type, uniform_weighted=metrics_type)

    expected_type = tff.FunctionType(
        parameter=collections.OrderedDict(
            model_weights=model_type, federated_dataset=dataset_type),
        result=aggregated_metrics_type)

    actual_type.check_assignable_from(expected_type)


class CentralizedEvalTest(tf.test.TestCase):

  def test_eval_metrics_have_correct_structure(self):
    client_data = create_balanced_client_datasets(0)
    metrics_builder = lambda: [tf.keras.metrics.MeanSquaredError()]
    eval_fn = evaluation.build_centralized_evaluation(tff_model_fn,
                                                      metrics_builder)
    model_weights = tff.learning.ModelWeights.from_model(tff_model_fn())
    eval_metrics = eval_fn(model_weights, client_data)
    metric_names = ['mean_squared_error', 'num_examples']
    self.assertCountEqual(eval_metrics.keys(), metric_names)

  def test_eval_metrics_return_correct_values(self):
    metrics_builder = lambda: [tf.keras.metrics.MeanSquaredError()]
    eval_fn = evaluation.build_centralized_evaluation(tff_model_fn,
                                                      metrics_builder)
    model_weights = tff.learning.ModelWeights.from_model(tff_model_fn())

    eval_metrics1 = eval_fn(model_weights, create_balanced_client_datasets(0))
    expected_metrics1 = collections.OrderedDict(
        mean_squared_error=0.0, num_examples=3.0)
    self.assertAllClose(eval_metrics1, expected_metrics1)

    eval_metrics2 = eval_fn(model_weights, create_balanced_client_datasets(1))
    expected_metrics2 = collections.OrderedDict(
        mean_squared_error=1.75, num_examples=3.0)
    self.assertAllClose(eval_metrics2, expected_metrics2)

  def test_eval_fn_has_correct_type_signature(self):
    metrics_builder = lambda: [tf.keras.metrics.MeanSquaredError()]
    eval_fn = evaluation.build_centralized_evaluation(tff_model_fn,
                                                      metrics_builder)
    actual_type = eval_fn.type_signature

    model_type = tff.FederatedType(
        tff.learning.ModelWeights(
            trainable=(
                tff.TensorType(tf.float32, [1, 1]),
                tff.TensorType(tf.float32, [1]),
            ),
            non_trainable=(),
        ), tff.SERVER)

    dataset_type = tff.FederatedType(
        tff.SequenceType(
            collections.OrderedDict(
                x=tff.TensorType(tf.float32, [None, 1]),
                y=tff.TensorType(tf.float32, [None, 1]))), tff.SERVER)

    metrics_type = tff.FederatedType(
        collections.OrderedDict(
            mean_squared_error=tff.TensorType(tf.float32),
            num_examples=tff.TensorType(tf.float32)), tff.SERVER)

    expected_type = tff.FunctionType(
        parameter=collections.OrderedDict(
            model_weights=model_type, centralized_dataset=dataset_type),
        result=metrics_type)

    actual_type.check_assignable_from(expected_type)


if __name__ == '__main__':
  tf.test.main()
