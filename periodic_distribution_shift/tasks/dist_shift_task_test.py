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

import attr
import tensorflow as tf
import tensorflow_federated as tff

from periodic_distribution_shift.tasks import dist_shift_task
from periodic_distribution_shift.tasks import dist_shift_task_data


def create_dataset(client_id):
  del client_id
  # Create data satisfying y = 2*x + 1
  x = [[1.0], [2.0]]
  y = [[3.0], [5.0]]
  return tf.data.Dataset.from_tensor_slices((x, y)).batch(1)


def create_client_data():
  return tff.simulation.datasets.ClientData.from_clients_and_tf_fn(
      client_ids=['1', '2', '3'], serializable_dataset_fn=create_dataset)


def create_task_data():
  return dist_shift_task_data.DistShiftDatasets(
      train_data=create_client_data(), test_data=create_client_data())


def keras_model_builder():
  inputs = tf.keras.layers.Input(shape=(3,), name='input_layer')
  outputs = tf.keras.layers.Dense(
      2, kernel_initializer='ones', use_bias=False, name='dense_layer')(
          inputs)
  return tf.keras.models.Model(inputs=inputs, outputs=outputs, name='model')


def tff_model_builder():
  x_type = tf.TensorSpec(shape=(None, 1), dtype=tf.float32)
  input_spec = (x_type, x_type)
  keras_model = keras_model_builder()
  loss = tf.keras.losses.MeanSquaredError()
  return tff.learning.keras_utils.from_keras_model(
      keras_model, loss, input_spec=input_spec)


class TaskUtilsTest(tf.test.TestCase):

  def test_constructs_with_matching_dataset_and_model_spec(self):
    baseline_task_spec = dist_shift_task.DistShiftTask(create_task_data(),
                                                       tff_model_builder)
    self.assertIsInstance(baseline_task_spec, dist_shift_task.DistShiftTask)

  def test_construct_raises_on_non_baseline_task_spec_datasets(self):
    with self.assertRaises(TypeError):
      dist_shift_task.DistShiftTask(create_dataset(0), tff_model_builder)

  def test_construct_raises_on_non_callable_model_fn(self):
    with self.assertRaises(attr.exceptions.NotCallableError):
      dist_shift_task.DistShiftTask(create_task_data(), 'bad_input')

  def test_construct_raises_on_non_tff_model(self):
    with self.assertRaisesRegex(
        TypeError, 'Expected model_fn to output a tff.learning.Model'):
      dist_shift_task.DistShiftTask(create_task_data(), keras_model_builder)

  def test_raises_on_different_data_and_model_spec(self):
    baseline_task_spec_data = create_task_data()
    model_input_spec = (
        tf.TensorSpec(shape=(None, 4, 2), dtype=tf.float32, name=None),
        tf.TensorSpec(shape=(None, 4, 2), dtype=tf.float32, name=None),
    )
    inputs = tf.keras.layers.Input(shape=(4, 2))
    outputs = tf.keras.layers.Dense(
        2, kernel_initializer='ones', use_bias=False)(
            inputs)
    keras_model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    def tff_model_fn():
      return tff.learning.keras_utils.from_keras_model(
          keras_model,
          loss=tf.keras.losses.MeanSquaredError(),
          input_spec=model_input_spec)

    with self.assertRaisesRegex(
        ValueError, 'Dataset element spec and model input spec do not match'):
      dist_shift_task.DistShiftTask(baseline_task_spec_data, tff_model_fn)


if __name__ == '__main__':
  tf.test.main()
