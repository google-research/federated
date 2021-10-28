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

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_federated as tff

from generalization.utils import eval_metric_distribution


def keras_model_builder_with_zeros():
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
  return tff.learning.from_keras_model(
      keras_model=keras_model_builder_with_zeros(),
      input_spec=get_input_spec(),
      loss=tf.keras.losses.MeanSquaredError(),
      metrics=[tf.keras.metrics.MeanSquaredError()])


class CommonStatUtilsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('unweighted_avg', eval_metric_distribution.unweighted_avg, 3.0),
      ('weighted_avg', eval_metric_distribution.weighted_avg, 2.0),
      ('unweighted_var', eval_metric_distribution.unweighted_var, 14.0 / 3.0),
      ('unweighted_std', eval_metric_distribution.unweighted_std,
       tf.sqrt(14.0 / 3)),
      ('median', eval_metric_distribution.median, 2.0),
      ('pct75', eval_metric_distribution.pct75, 6.0),
      ('pct25', eval_metric_distribution.pct25, 1.0),
  )
  def test_eval_metric_distribution(self, tested_fn, expected_output):
    input_tensor = tf.constant([1.0, 2.0, 6.0])
    weights_tensor = tf.constant([8.0, 5.0, 2.0])
    self.assertAlmostEqual(
        tested_fn(input_tensor, weights_tensor), expected_output)


if __name__ == '__main__':
  tf.test.main()


class FedEvalFnTest(tf.test.TestCase):

  def test_fed_eval_on_one_client(self):
    tff_model = tff_model_builder()
    model_weights = tff.learning.ModelWeights.from_model(tff_model)
    metrics_builder = lambda: [tf.keras.metrics.MeanSquaredError()]

    fed_eval_fn = eval_metric_distribution.create_federated_eval_distribution_computation(
        tff_model_builder, metrics_builder)
    dataset = create_dataset()
    fed_metrics = fed_eval_fn(model_weights, [dataset])

    self.assertLen(fed_metrics, 1)
    client_metrics = fed_metrics[0]
    self.assertCountEqual(client_metrics.keys(),
                          ['mean_squared_error', 'num_examples'])

    # Note that our model uses all zeros for weights, therefore predicting
    # y = 0. We manually compute the MSE under this model.
    expected_loss = (3.0**2 + 5.0**2 + 7.0**2) / 3.0
    self.assertNear(
        client_metrics['mean_squared_error'], expected_loss, err=1e-6)
    self.assertNear(client_metrics['num_examples'], 3.0, err=1e-8)

  def test_fed_eval_uses_input_weights(self):
    keras_model = keras_model_builder_with_ones()
    model_weights = tff.learning.ModelWeights(keras_model.trainable_weights,
                                              keras_model.non_trainable_weights)
    metrics_builder = lambda: [tf.keras.metrics.MeanSquaredError()]

    fed_eval_fn = eval_metric_distribution.create_federated_eval_distribution_computation(
        tff_model_builder, metrics_builder)
    dataset = create_dataset()
    fed_metrics = fed_eval_fn(model_weights, [dataset])

    self.assertLen(fed_metrics, 1)
    client_metrics = fed_metrics[0]

    # Using the all ones weights, our model predicts y = x + 1
    expected_loss = ((2.0 - 3.0)**2 + (3.0 - 5.0)**2 + (4.0 - 7.0)**2) / 3.0

    self.assertNear(
        client_metrics['mean_squared_error'], expected_loss, err=1e-6)
    self.assertNear(client_metrics['num_examples'], 3.0, err=1e-8)

  def test_fed_eval_on_one_client_repeated(self):
    tff_model = tff_model_builder()
    model_weights = tff.learning.ModelWeights.from_model(tff_model)
    metrics_builder = lambda: [tf.keras.metrics.MeanSquaredError()]

    fed_eval_fn = eval_metric_distribution.create_federated_eval_distribution_computation(
        tff_model_builder, metrics_builder)
    dataset = create_dataset()
    fed_metrics = fed_eval_fn(model_weights, [dataset, dataset])

    self.assertLen(fed_metrics, 2)
    self.assertAllClose(fed_metrics[0], fed_metrics[1])

  def test_fed_eval_on_different_clients(self):
    tff_model = tff_model_builder()
    model_weights = tff.learning.ModelWeights.from_model(tff_model)
    metrics_builder = lambda: [tf.keras.metrics.MeanSquaredError()]

    fed_eval_fn = eval_metric_distribution.create_federated_eval_distribution_computation(
        tff_model_builder, metrics_builder)
    federated_dataset = create_federated_dataset()
    fed_metrics = fed_eval_fn(model_weights, federated_dataset)

    self.assertLen(fed_metrics, 3)
    actual_mse_values = sorted([x['mean_squared_error'] for x in fed_metrics])
    expected_mse_values = [9.0, 25.0, 49.0]
    self.assertAllClose(actual_mse_values, expected_mse_values)
    actual_num_examples = [x['num_examples'] for x in fed_metrics]
    expected_num_examples = [1.0, 1.0, 1.0]
    self.assertAllClose(actual_num_examples, expected_num_examples)


class FedEvalCustomStatFnTest(tf.test.TestCase):

  def test_eval_metric_distribution_on_one_client(self):
    tff_model = tff_model_builder()
    model_weights = tff.learning.ModelWeights.from_model(tff_model)
    metrics_builder = lambda: [tf.keras.metrics.MeanSquaredError()]

    eval_metric_distribution_fn = eval_metric_distribution.create_federated_eval_distribution_fn(
        tff_model_builder, metrics_builder,
        {'avg': eval_metric_distribution.unweighted_avg})
    dataset = create_dataset()
    distribution_metrics = eval_metric_distribution_fn(model_weights, [dataset])

    self.assertLen(distribution_metrics, 1)
    self.assertCountEqual(distribution_metrics.keys(),
                          ['mean_squared_error_avg'])

    # Note that our model uses all zeros for weights, therefore predicting
    # y = 0. We manually compute the MSE under this model.
    expected_loss = (3.0**2 + 5.0**2 + 7.0**2) / 3.0
    self.assertNear(
        distribution_metrics['mean_squared_error_avg'], expected_loss, err=1e-6)

  def test_eval_metric_distribution_uses_input_weights(self):
    keras_model = keras_model_builder_with_ones()
    model_weights = tff.learning.ModelWeights(keras_model.trainable_weights,
                                              keras_model.non_trainable_weights)
    metrics_builder = lambda: [tf.keras.metrics.MeanSquaredError()]

    eval_metric_distribution_fn = eval_metric_distribution.create_federated_eval_distribution_fn(
        tff_model_builder, metrics_builder,
        {'avg': eval_metric_distribution.unweighted_avg})
    dataset = create_dataset()

    # The model weights from `tff_model_builder` will *not* be used.
    distribution_metrics = eval_metric_distribution_fn(model_weights, [dataset])

    self.assertLen(distribution_metrics, 1)

    # Using the all ones weights, our model predicts y = x + 1
    expected_loss = ((2.0 - 3.0)**2 + (3.0 - 5.0)**2 + (4.0 - 7.0)**2) / 3.0

    self.assertNear(
        distribution_metrics['mean_squared_error_avg'], expected_loss, err=1e-6)

  def test_eval_metric_distribution_on_one_client_repeated(self):
    tff_model = tff_model_builder()
    model_weights = tff.learning.ModelWeights.from_model(tff_model)
    metrics_builder = lambda: [tf.keras.metrics.MeanSquaredError()]

    eval_metric_distribution_fn = eval_metric_distribution.create_federated_eval_distribution_fn(
        tff_model_builder, metrics_builder,
        {'avg': eval_metric_distribution.unweighted_avg})
    dataset = create_dataset()
    distribution_metrics = eval_metric_distribution_fn(model_weights,
                                                       [dataset, dataset])

    self.assertLen(distribution_metrics, 1)
    self.assertCountEqual(distribution_metrics.keys(),
                          ['mean_squared_error_avg'])
    expected_loss = (3.0**2 + 5.0**2 + 7.0**2) / 3.0

    self.assertNear(
        distribution_metrics['mean_squared_error_avg'], expected_loss, err=1e-6)

  def test_fed_eval_multiple_stats_on_different_clients(self):
    tff_model = tff_model_builder()
    model_weights = tff.learning.ModelWeights.from_model(tff_model)
    metrics_builder = lambda: [tf.keras.metrics.MeanSquaredError()]

    eval_metric_distribution_fn = eval_metric_distribution.create_federated_eval_distribution_fn(
        tff_model_builder, metrics_builder, {
            'avg': eval_metric_distribution.unweighted_avg,
            'wavg': eval_metric_distribution.weighted_avg,
            'median': eval_metric_distribution.median,
            'var': eval_metric_distribution.unweighted_var,
            'std': eval_metric_distribution.unweighted_std,
        })
    federated_dataset = create_federated_dataset()
    distribution_metrics = eval_metric_distribution_fn(model_weights,
                                                       federated_dataset)

    self.assertLen(distribution_metrics, 5)
    expected_mse_avg = (9.0 + 25.0 + 49.0) / 3
    expected_mse_wavg = (9.0 + 25.0 + 49.0) / 3
    expected_mse_median = 25.0

    expected_mse_var = ((9.0 - expected_mse_avg)**2 +
                        (25.0 - expected_mse_avg)**2 +
                        (49.0 - expected_mse_avg)**2) / 3.0
    expected_mse_std = tf.sqrt(expected_mse_var)

    self.assertNear(
        distribution_metrics['mean_squared_error_avg'],
        expected_mse_avg,
        err=1e-4)
    self.assertNear(
        distribution_metrics['mean_squared_error_wavg'],
        expected_mse_wavg,
        err=1e-4)
    self.assertNear(
        distribution_metrics['mean_squared_error_median'],
        expected_mse_median,
        err=1e-4)
    self.assertNear(
        distribution_metrics['mean_squared_error_var'],
        expected_mse_var,
        err=1e-4)
    self.assertNear(
        distribution_metrics['mean_squared_error_std'],
        expected_mse_std,
        err=1e-4)


if __name__ == '__main__':
  tf.test.main()
