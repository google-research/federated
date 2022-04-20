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

from data_poor_fl import hypcluster_eval


def create_dataset():
  # Create data satisfying y = x + 1
  x = [[1.0], [2.0], [3.0]]
  y = [[2.0], [3.0], [4.0]]
  return tf.data.Dataset.from_tensor_slices((x, y)).batch(1)


def create_dataset_with_zeros_y():
  # Create data with y = 0
  x = [[1.0], [2.0]]
  y = [[0.0], [0.0]]
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


def weight_tensors_from_model(
    model: tff.learning.Model) -> tff.learning.ModelWeights:
  return tf.nest.map_structure(lambda var: var.numpy(),
                               tff.learning.ModelWeights.from_model(model))


def create_initial_models(num_models: int):
  model = model_fn(initializer='ones')
  model_weights_tensors = weight_tensors_from_model(model)
  return [model_weights_tensors for _ in range(num_models)]


class HypClusterEvalTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('clusters1', 1),
      ('clusters2', 2),
      ('clusters3', 3),
      ('clusters5', 5),
  )
  def test_returns_expected_metrics_keys(self, num_clusters):
    hyp_eval = hypcluster_eval.build_hypcluster_eval(
        model_fn=model_fn, num_clusters=num_clusters)
    self.assertLen(hyp_eval.type_signature.parameter[0].member, num_clusters)
    model_weights = tff.learning.ModelWeights.from_model(model_fn())
    federated_data = [create_dataset(), create_dataset()]
    hyp_metrics = hyp_eval([model_weights] * num_clusters, federated_data)
    choose_metrics_keys = ['choose_' + str(i) for i in range(num_clusters)]
    model_metrics_keys = ['model_' + str(i) for i in range(num_clusters)]
    expected_keys = choose_metrics_keys + model_metrics_keys + ['best']
    self.assertCountEqual(list(hyp_metrics.keys()), expected_keys)

  def test_matches_federated_eval_with_one_cluster(self):
    hyp_eval = hypcluster_eval.build_hypcluster_eval(
        model_fn=model_fn, num_clusters=1)
    federated_eval = tff.learning.build_federated_evaluation(model_fn)
    model_weights = tff.learning.ModelWeights.from_model(model_fn())
    federated_data = [create_dataset(), create_dataset()]
    hyp_metrics = hyp_eval([model_weights], federated_data)
    self.assertCountEqual(
        list(hyp_metrics.keys()), ['best', 'choose_0', 'model_0'])
    self.assertEqual(hyp_metrics['choose_0'], 1.0)
    best_hyp_metrics = hyp_metrics['best']
    model_0_metrics = hyp_metrics['model_0']
    reference_metrics = federated_eval(model_weights, federated_data)['eval']
    self.assertAllClose(best_hyp_metrics, reference_metrics)
    self.assertAllClose(best_hyp_metrics, model_0_metrics)

  def test_selects_best_model(self):
    hyp_eval = hypcluster_eval.build_hypcluster_eval(
        model_fn=model_fn, num_clusters=2)
    zero_model = model_fn(initializer='zeros')
    ones_model = model_fn(initializer='ones')
    zero_weights = tff.learning.ModelWeights.from_model(zero_model)
    ones_weights = tff.learning.ModelWeights.from_model(ones_model)
    federated_data = [create_dataset()]
    hyp_metrics = hyp_eval([zero_weights, ones_weights], federated_data)
    self.assertEqual(hyp_metrics['choose_0'], 0.0)
    self.assertEqual(hyp_metrics['choose_1'], 1.0)
    federated_eval = tff.learning.build_federated_evaluation(model_fn)
    reference_metrics = federated_eval(ones_weights, federated_data)['eval']
    self.assertAllClose(hyp_metrics['best'], hyp_metrics['model_1'])
    self.assertAllClose(hyp_metrics['best'], reference_metrics)


class HypClusterEvalWithDatasetSplitTest(tf.test.TestCase,
                                         parameterized.TestCase):

  def test_get_metrics_for_select_and_test_returns_correct_metrics(self):
    model_list = [model_fn(), model_fn()]
    weights_list = [
        weight_tensors_from_model(model_fn('zeros')),
        weight_tensors_from_model(model_fn('ones')),
    ]
    client_data = collections.OrderedDict(
        selection_data=create_dataset(),
        test_data=create_dataset_with_zeros_y())
    selection_metrics, test_metrics = (
        hypcluster_eval._get_metrics_for_select_and_test(
            model_list, weights_list, client_data))
    # The first model will predict y=0 for all examples. On the selection data
    # (given by `create_dataset()`), the sum of squared error is 2*2+3*3+4*4=29.
    self.assertEqual(selection_metrics[0]['loss'], [29.0, 3.0])
    # On the test data (given by `create_dataset_with_zeros_y()`), the sum of
    # squared error is 0.
    self.assertEqual(test_metrics[0]['loss'], [0.0, 2.0])
    # The second model will predict y=x+1. On the selection data, the sum of
    # squared error is 0
    self.assertEqual(selection_metrics[1]['loss'], [0.0, 3.0])
    # On the test data, the sum of squared error is 2*2+3*3=13.
    self.assertEqual(test_metrics[1]['loss'], [13.0, 2.0])
    for model_i in [0, 1]:
      # The selection data is given by `create_dataset()`.
      self.assertEqual(selection_metrics[model_i]['num_examples'], [3])
      self.assertEqual(selection_metrics[model_i]['num_batches'], [3])
      # The test data is given by `create_dataset_with_zeros_y()`.
      self.assertEqual(test_metrics[model_i]['num_examples'], [2])
      self.assertEqual(test_metrics[model_i]['num_batches'], [2])

  def test_hypcluster_eval_returns_correct_metrics_with_single_cluster(self):
    eval_comp = hypcluster_eval.build_hypcluster_eval_with_dataset_split(
        model_fn=model_fn, num_clusters=1)
    model_weights_list = [weight_tensors_from_model(model_fn('ones'))]
    first_client_data = collections.OrderedDict(
        selection_data=create_dataset(),
        test_data=create_dataset_with_zeros_y())
    second_client_data = collections.OrderedDict(
        selection_data=create_dataset_with_zeros_y(),
        test_data=create_dataset())
    eval_metrics = eval_comp(model_weights_list,
                             [first_client_data, second_client_data])
    # `num_clusters=1`, so both clients have to select the model with all ones,
    # and evaluate it on their test data.
    self.assertEqual(list(eval_metrics.keys()), ['best', 'model_0', 'choose_0'])
    self.assertEqual(eval_metrics['choose_0'], 1.0)
    for key in ['best', 'model_0']:
      # The sum of squared error is 2*2+3*3=13 on the first client's test data,
      # and is 0 on the second client's test data.
      self.assertAlmostEqual(eval_metrics[key]['loss'], 13.0 / 5.0, places=6)
      self.assertEqual(eval_metrics[key]['num_examples'], 5)

  @parameterized.named_parameters(
      ('clusters2', 2),
      ('clusters3', 3),
      ('clusters5', 5),
  )
  def test_hypcluster_eval_returns_correct_metrics_with_multi_clusters(
      self, num_clusters):
    eval_comp = hypcluster_eval.build_hypcluster_eval_with_dataset_split(
        model_fn=model_fn, num_clusters=num_clusters)
    model_weights_list = [
        weight_tensors_from_model(model_fn(tf.constant_initializer(i)))
        for i in range(num_clusters)
    ]  # The i-th model has weights equal to i.
    first_client_data = collections.OrderedDict(
        selection_data=create_dataset(),
        test_data=create_dataset_with_zeros_y())
    second_client_data = collections.OrderedDict(
        selection_data=create_dataset_with_zeros_y(),
        test_data=create_dataset())
    eval_metrics = eval_comp(model_weights_list,
                             [first_client_data, second_client_data])
    expected_keys = ['best'] + [f'model_{i}' for i in range(num_clusters)
                               ] + [f'choose_{i}' for i in range(num_clusters)]
    self.assertEqual(list(eval_metrics.keys()), expected_keys)
    for i in range(num_clusters):
      if i == 0 or i == 1:
        # There are two clients: Half of the clients (i.e., the second client)
        # select `model_0`, the rest half select `model_1`.
        self.assertEqual(eval_metrics[f'choose_{i}'], 0.5)
      else:
        self.assertEqual(eval_metrics[f'choose_{i}'], 0.0)
    # The sum of squared error is 2*2+3*3=13 on the first client's test data,
    # and is 2*2+3*3+4*4=29 on the second client's test data.
    self.assertAlmostEqual(eval_metrics['best']['loss'], 42.0 / 5.0, places=5)
    for i in range(num_clusters):
      # The i-th model has weights equal to i, so for input x, it will predict
      # y = x*i + i. Both clients need to evaluate this model on its test data.
      expected_sum_squared_error = ((i + i)**2 + (2 * i + i)**2 +
                                    (i + i - 2)**2 + (2 * i + i - 3)**2 +
                                    (3 * i + i - 4)**2)
      self.assertAlmostEqual(
          eval_metrics[f'model_{i}']['loss'],
          float(expected_sum_squared_error) / 5.0,
          places=5)
      self.assertAlmostEqual(eval_metrics[f'model_{i}']['num_examples'], 5)


if __name__ == '__main__':
  tf.test.main()
