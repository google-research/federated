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

import tensorflow as tf
import tensorflow_federated as tff

from large_cohort import eval_utils
from large_cohort import simulation_specs


def get_model_spec():

  def keras_model_builder():
    # Create a simple linear regression model, single output.
    # We initialize all weights to zero.
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(
            1,
            kernel_initializer='zeros',
            bias_initializer='zeros',
            input_shape=(1,))
    ])
    return model

  return simulation_specs.ModelSpec(
      keras_model_builder=keras_model_builder,
      loss_builder=tf.keras.losses.MeanSquaredError,
      metrics_builder=lambda: [tf.keras.metrics.MeanSquaredError()])


def keras_model_builder_with_ones():
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(
          1,
          kernel_initializer='ones',
          bias_initializer='ones',
          input_shape=(1,))
  ])
  return model


def create_dataset():
  # Create data satisfying y = 2*x + 1
  x = [[1.0], [2.0], [3.0]]
  y = [[3.0], [5.0], [7.0]]
  return tf.data.Dataset.from_tensor_slices((x, y)).batch(1)


def create_federated_dataset():
  x1 = [[1.0]]
  y1 = [[3.0]]
  dataset1 = tf.data.Dataset.from_tensor_slices((x1, y1)).batch(1)
  x2 = [[2.0]]
  y2 = [[5.0]]
  dataset2 = tf.data.Dataset.from_tensor_slices((x2, y2)).batch(1)
  x3 = [[3.0]]
  y3 = [[7.0]]
  dataset3 = tf.data.Dataset.from_tensor_slices((x3, y3)).batch(1)
  return [dataset1, dataset2, dataset3]


def get_input_spec():
  return create_dataset().element_spec


def tff_model_builder():
  model_spec = get_model_spec()
  return tff.learning.from_keras_model(
      keras_model=model_spec.keras_model_builder(),
      input_spec=get_input_spec(),
      loss=model_spec.loss_builder(),
      metrics=model_spec.metrics_builder())


class CentralizedEvalFnTest(tf.test.TestCase):

  def test_evaluation_on_simple_model(self):
    model_spec = get_model_spec()
    dataset = create_dataset()
    tff_model = tff_model_builder()
    model_weights = tff.learning.ModelWeights.from_model(tff_model)
    centralized_eval_fn = eval_utils.create_centralized_eval_fn(model_spec)
    eval_metrics = centralized_eval_fn(model_weights, dataset)
    self.assertCountEqual(eval_metrics.keys(), ['loss', 'mean_squared_error'])
    self.assertEqual(eval_metrics['loss'], eval_metrics['mean_squared_error'])
    self.assertNear(eval_metrics['loss'], (9.0 + 25.0 + 49.0) / 3.0, err=1e-6)

  def test_evaluation_uses_tff_model_weights(self):
    dataset = create_dataset()
    tff_model = tff_model_builder()
    # Note that by construction, these model weights are all zero.
    model_weights = tff.learning.ModelWeights.from_model(tff_model)

    model_spec_with_ones = simulation_specs.ModelSpec(
        keras_model_builder=keras_model_builder_with_ones,
        loss_builder=tf.keras.losses.MeanSquaredError,
        metrics_builder=lambda: [tf.keras.metrics.MeanSquaredError()])

    centralized_eval_fn = eval_utils.create_centralized_eval_fn(
        model_spec_with_ones)
    eval_metrics = centralized_eval_fn(model_weights, dataset)
    # We check to make sure the evaluation occurred at the all zero weights.
    self.assertCountEqual(eval_metrics.keys(), ['loss', 'mean_squared_error'])
    self.assertEqual(eval_metrics['loss'], eval_metrics['mean_squared_error'])
    self.assertNear(eval_metrics['loss'], (9.0 + 25.0 + 49.0) / 3.0, err=1e-6)

  def test_centralized_eval_agrees_with_federated_eval(self):
    model_spec = get_model_spec()
    dataset = create_dataset()
    tff_model = tff_model_builder()
    model_weights = tff.learning.ModelWeights.from_model(tff_model)

    centralized_eval_fn = eval_utils.create_centralized_eval_fn(model_spec)
    centralized_eval_metrics = centralized_eval_fn(model_weights, dataset)

    federated_eval_fn = tff.learning.build_federated_evaluation(
        tff_model_builder)
    federated_eval_metrics = federated_eval_fn(model_weights, [dataset])

    self.assertEqual(federated_eval_metrics['mean_squared_error'],
                     centralized_eval_metrics['mean_squared_error'])


class FederatedEvalFnTest(tf.test.TestCase):

  def test_federated_eval_on_one_client(self):
    tff_model = tff_model_builder()
    model_weights = tff.learning.ModelWeights.from_model(tff_model)
    metrics_builder = lambda: [tf.keras.metrics.MeanSquaredError()]

    federated_evaluate_fn = eval_utils.create_federated_eval_fn(
        tff_model_builder, metrics_builder)
    dataset = create_dataset()
    federated_metrics = federated_evaluate_fn(model_weights, [dataset])

    self.assertLen(federated_metrics, 1)
    client_metrics = federated_metrics[0]
    self.assertCountEqual(client_metrics.keys(),
                          ['mean_squared_error', 'num_examples'])

    # Note that our model uses all zeros for weights, therefore predicting
    # y = 0. We manually compute the MSE under this model.
    expected_loss = (3.0**2 + 5.0**2 + 7.0**2) / 3.0
    self.assertNear(
        client_metrics['mean_squared_error'], expected_loss, err=1e-6)
    self.assertNear(client_metrics['num_examples'], 3.0, err=1e-8)

  def test_federated_eval_uses_input_weights(self):
    keras_model = keras_model_builder_with_ones()
    model_weights = tff.learning.ModelWeights(keras_model.trainable_weights,
                                              keras_model.non_trainable_weights)
    metrics_builder = lambda: [tf.keras.metrics.MeanSquaredError()]

    federated_evaluate_fn = eval_utils.create_federated_eval_fn(
        tff_model_builder, metrics_builder)
    dataset = create_dataset()
    federated_metrics = federated_evaluate_fn(model_weights, [dataset])

    self.assertLen(federated_metrics, 1)
    client_metrics = federated_metrics[0]

    # Using the all ones weights, our model predicts y = x + 1
    expected_loss = ((2.0 - 3.0)**2 + (3.0 - 5.0)**2 + (4.0 - 7.0)**2) / 3.0

    self.assertNear(
        client_metrics['mean_squared_error'], expected_loss, err=1e-6)
    self.assertNear(client_metrics['num_examples'], 3.0, err=1e-8)

  def test_federated_eval_on_one_client_repeated(self):
    tff_model = tff_model_builder()
    model_weights = tff.learning.ModelWeights.from_model(tff_model)
    metrics_builder = lambda: [tf.keras.metrics.MeanSquaredError()]

    federated_evaluate_fn = eval_utils.create_federated_eval_fn(
        tff_model_builder, metrics_builder)
    dataset = create_dataset()
    federated_metrics = federated_evaluate_fn(model_weights, [dataset, dataset])

    self.assertLen(federated_metrics, 2)
    self.assertAllClose(federated_metrics[0], federated_metrics[1])

  def test_federated_eval_on_different_clients(self):
    tff_model = tff_model_builder()
    model_weights = tff.learning.ModelWeights.from_model(tff_model)
    metrics_builder = lambda: [tf.keras.metrics.MeanSquaredError()]

    federated_evaluate_fn = eval_utils.create_federated_eval_fn(
        tff_model_builder, metrics_builder)
    federated_dataset = create_federated_dataset()
    federated_metrics = federated_evaluate_fn(model_weights, federated_dataset)

    self.assertLen(federated_metrics, 3)
    actual_mse_values = sorted(
        [x['mean_squared_error'] for x in federated_metrics])
    expected_mse_values = [9.0, 25.0, 49.0]
    self.assertAllClose(actual_mse_values, expected_mse_values)
    actual_num_examples = [x['num_examples'] for x in federated_metrics]
    expected_num_examples = [1.0, 1.0, 1.0]
    self.assertAllClose(actual_num_examples, expected_num_examples)


if __name__ == '__main__':
  tf.test.main()
