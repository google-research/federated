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
"""Integration tests for q-FFL against a simple regression model."""

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_federated as tff

from large_cohort.processes import q_ffl


def create_dataset1():
  # Create data satisfying y = 2*x + 1
  x = [[1.0], [2.0]]
  y = [[3.0], [5.0]]
  return tf.data.Dataset.from_tensor_slices((x, y)).batch(1)


def create_dataset2():
  # Create data satisfying y = 3*x + 5
  x = [[1.0], [2.0]]
  y = [[8.0], [11.0]]
  return tf.data.Dataset.from_tensor_slices((x, y)).batch(1)


def create_dataset3():
  # Create data satisfying y = x - 1
  x = [[1.0], [2.0], [3.0]]
  y = [[0.0], [1.0], [2.0]]
  return tf.data.Dataset.from_tensor_slices((x, y)).batch(1)


def get_input_spec():
  return create_dataset1().element_spec


def model_fn():
  keras_model = tf.keras.Sequential([
      tf.keras.layers.Dense(
          1,
          kernel_initializer='zeros',
          bias_initializer='zeros',
          input_shape=(1,))
  ])
  return tff.learning.from_keras_model(
      keras_model=keras_model,
      input_spec=get_input_spec(),
      loss=tf.keras.losses.MeanSquaredError())


def get_output_converter():
  return q_ffl.build_keras_output_to_loss_fn(tf.keras.metrics.MeanSquaredError)


class KerasMetricConverterTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('config1', 1.0, 3.0, tf.keras.metrics.MeanSquaredError),
      ('config2', 2.01, 4.0, tf.keras.metrics.SparseCategoricalAccuracy),
      ('config3', 7.3, 1.0, tf.keras.metrics.Mean))
  def test_with_mean_metric(self, total, count, mean_metric_builder):
    expected_loss = total / count
    convert_variables_to_mse_fn = q_ffl.build_keras_output_to_loss_fn(
        mean_metric_builder)
    actual_loss = convert_variables_to_mse_fn({'loss': [total, count]})
    self.assertNear(expected_loss, actual_loss, err=1e-5)

  @parameterized.named_parameters(('loss1', 6.0), ('loss2', 1.5),
                                  ('loss3', -7.36))
  def test_with_sum_metric(self, expected_loss):
    convert_variables_to_mse_fn = q_ffl.build_keras_output_to_loss_fn(
        tf.keras.metrics.Sum)
    actual_loss = convert_variables_to_mse_fn({'loss': [expected_loss]})
    self.assertNear(expected_loss, actual_loss, err=1e-5)


class QFFLTest(tf.test.TestCase, parameterized.TestCase):

  def _run_rounds(self, process, federated_data, num_rounds):
    train_outputs = []
    initial_state = process.initialize()
    state = initial_state
    for _ in range(num_rounds):
      state, metrics = process.next(state, federated_data)
      train_outputs.append(metrics)
    return state, train_outputs, initial_state

  def test_qffl_create_iter_proc(self):
    process = q_ffl.build_q_ffl_process(
        model_fn=model_fn,
        fairness_parameter=1.0,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        output_to_loss_fn=get_output_converter())
    self.assertIsInstance(process, tff.templates.IterativeProcess)

  def test_qffl_process_types_match_fedavg_process_types(self):
    qffl_process = q_ffl.build_q_ffl_process(
        model_fn=model_fn,
        fairness_parameter=1.0,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        output_to_loss_fn=get_output_converter())
    fedavg_process = tff.learning.build_federated_averaging_process(
        model_fn=model_fn, client_optimizer_fn=tf.keras.optimizers.SGD)

    self.assertEqual(qffl_process.initialize.type_signature,
                     fedavg_process.initialize.type_signature)
    self.assertEqual(qffl_process.next.type_signature,
                     fedavg_process.next.type_signature)
    self.assertEqual(qffl_process.get_model_weights.type_signature,
                     fedavg_process.get_model_weights.type_signature)

  def test_qffl_decreases_loss(self):
    federated_data = [create_dataset1()]
    process = q_ffl.build_q_ffl_process(
        model_fn=model_fn,
        fairness_parameter=1.0,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        output_to_loss_fn=get_output_converter())
    _, train_outputs, _ = self._run_rounds(process, federated_data, 3)
    self.assertLess(train_outputs[-1]['train']['loss'],
                    train_outputs[0]['train']['loss'])

  def test_qffl_with_single_client_matches_fedavg(self):
    federated_data = [create_dataset1()]
    q_ffl_process = q_ffl.build_q_ffl_process(
        model_fn=model_fn,
        fairness_parameter=3.0,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        output_to_loss_fn=get_output_converter())
    fedavg_process = tff.learning.build_federated_averaging_process(
        model_fn=model_fn, client_optimizer_fn=tf.keras.optimizers.SGD)
    q_ffl_state = q_ffl_process.initialize()
    fedavg_state = fedavg_process.initialize()
    tf.nest.map_structure(self.assertAllClose, q_ffl_state, fedavg_state)
    for _ in range(3):
      q_ffl_state, q_ffl_results = q_ffl_process.next(q_ffl_state,
                                                      federated_data)
      fedavg_state, fedavg_results = fedavg_process.next(
          fedavg_state, federated_data)
      tf.nest.map_structure(self.assertAllClose, q_ffl_state, fedavg_state)
      tf.nest.map_structure(self.assertAllClose, q_ffl_results, fedavg_results)

  def test_qffl_with_fairness_0_matches_weighted_fedavg_on_balanced_data(self):
    federated_data = [create_dataset1(), create_dataset2()]
    q_ffl_process = q_ffl.build_q_ffl_process(
        model_fn=model_fn,
        fairness_parameter=0.0,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        output_to_loss_fn=get_output_converter())
    fedavg_process = tff.learning.build_federated_averaging_process(
        model_fn=model_fn,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        client_weighting=tff.learning.ClientWeighting.NUM_EXAMPLES)
    q_ffl_state = q_ffl_process.initialize()
    fedavg_state = fedavg_process.initialize()
    tf.nest.map_structure(self.assertAllClose, q_ffl_state, fedavg_state)
    for _ in range(3):
      q_ffl_state, q_ffl_results = q_ffl_process.next(q_ffl_state,
                                                      federated_data)
      fedavg_state, fedavg_results = fedavg_process.next(
          fedavg_state, federated_data)
      tf.nest.map_structure(self.assertAllClose, q_ffl_state, fedavg_state)
      tf.nest.map_structure(self.assertAllClose, q_ffl_results, fedavg_results)

  def test_qffl_with_fairness_0_matches_uniform_fedavg_on_unbalanced_data(self):
    federated_data = [create_dataset1(), create_dataset2()]
    q_ffl_process = q_ffl.build_q_ffl_process(
        model_fn=model_fn,
        fairness_parameter=0.0,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        output_to_loss_fn=get_output_converter())
    fedavg_process = tff.learning.build_federated_averaging_process(
        model_fn=model_fn,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        client_weighting=tff.learning.ClientWeighting.UNIFORM)
    q_ffl_state = q_ffl_process.initialize()
    fedavg_state = fedavg_process.initialize()
    tf.nest.map_structure(self.assertAllClose, q_ffl_state, fedavg_state)
    for _ in range(3):
      q_ffl_state, q_ffl_results = q_ffl_process.next(q_ffl_state,
                                                      federated_data)
      fedavg_state, fedavg_results = fedavg_process.next(
          fedavg_state, federated_data)
      tf.nest.map_structure(self.assertAllClose, q_ffl_state, fedavg_state)
      tf.nest.map_structure(self.assertAllClose, q_ffl_results, fedavg_results)

  @parameterized.named_parameters(('param1', 0.1), ('param2', 1.0),
                                  ('param3', 10.0))
  def test_qffl_with_fairness_larger_than_0_does_not_match_fedavg(
      self, fairness_parameter):
    federated_data = [create_dataset1(), create_dataset2()]
    q_ffl_process = q_ffl.build_q_ffl_process(
        model_fn=model_fn,
        fairness_parameter=fairness_parameter,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        output_to_loss_fn=get_output_converter())
    fedavg_process = tff.learning.build_federated_averaging_process(
        model_fn=model_fn, client_optimizer_fn=tf.keras.optimizers.SGD)
    q_ffl_state = q_ffl_process.initialize()
    fedavg_state = fedavg_process.initialize()
    tf.nest.map_structure(self.assertAllClose, q_ffl_state, fedavg_state)

    q_ffl_state, _ = q_ffl_process.next(q_ffl_state, federated_data)
    fedavg_state, _ = fedavg_process.next(fedavg_state, federated_data)
    self.assertNotAllClose(q_ffl_state.model.trainable,
                           fedavg_state.model.trainable)


if __name__ == '__main__':
  tf.test.main()
