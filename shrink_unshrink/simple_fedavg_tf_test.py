# Copyright 2021, The TensorFlow Federated Authors.
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
"""Tests for simple_fedavg_tf."""
import collections

import tensorflow as tf
import tensorflow_federated as tff

from shrink_unshrink import simple_fedavg_tf


def _create_test_small_dense_model() -> tf.keras.Model:
  model = tf.keras.Sequential()
  model.add(tf.keras.Input(shape=(2,)))
  model.add(
      tf.keras.layers.Dense(
          3,
          activation="relu",
          name="layer1",
          kernel_initializer="ones",
          bias_initializer="ones"))
  model.add(
      tf.keras.layers.Dense(
          4,
          activation="relu",
          name="layer2",
          kernel_initializer="ones",
          bias_initializer="ones"))
  model.add(
      tf.keras.layers.Dense(
          5, name="layer3", kernel_initializer="ones", bias_initializer="ones"))
  return model


def _small_dense_model_fn() -> tff.learning.Model:
  keras_model = _create_test_small_dense_model()
  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  input_spec = collections.OrderedDict(
      x=tf.TensorSpec([None, 2], tf.int32),
      y=tf.TensorSpec([None, 5], tf.int32))
  return tff.learning.from_keras_model(
      keras_model=keras_model, input_spec=input_spec, loss=loss)


def _create_test_big_dense_model() -> tf.keras.Model:
  model = tf.keras.Sequential()
  model.add(tf.keras.Input(shape=(2,)))
  model.add(
      tf.keras.layers.Dense(
          30,
          activation="relu",
          name="layer1",
          kernel_initializer="ones",
          bias_initializer="ones"))
  model.add(
      tf.keras.layers.Dense(
          40,
          activation="relu",
          name="layer2",
          kernel_initializer="ones",
          bias_initializer="ones"))
  model.add(
      tf.keras.layers.Dense(
          5, name="layer3", kernel_initializer="ones", bias_initializer="ones"))
  return model


def _big_dense_model_fn() -> tff.learning.Model:
  keras_model = _create_test_big_dense_model()
  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  input_spec = collections.OrderedDict(
      x=tf.TensorSpec([None, 2], tf.int32),
      y=tf.TensorSpec([None, 5], tf.int32))
  return tff.learning.from_keras_model(
      keras_model=keras_model, input_spec=input_spec, loss=loss)


class SimpleFedavgTfTest(tf.test.TestCase):

  def test_projection(self):
    # Create projection_matrix, a 3 by 2 matrix; call it P for short.
    projection_matrix = tf.convert_to_tensor([[1.0, 4.0], [2.0, 5.0],
                                              [3.0, 6.0]])

    # Create 3 dimensional input vector to be projected.
    input_vec = tf.convert_to_tensor([7.0, 8.0, 9.0])

    # Project input_vec down into 2D space
    # then project resulting 2D vector back into 3D space
    # i.e., left multiply input_vec by PP^T
    my_output_vec = simple_fedavg_tf.projection(projection_matrix, input_vec)

    # This is what the ouput should look like
    output_vec = tf.convert_to_tensor([538.0, 710.0, 882.0])

    self.assertAllEqual(output_vec, my_output_vec)

  def test_flatten_and_reshape_list_of_tensors(self):
    model = _big_dense_model_fn()
    weights_list = simple_fedavg_tf.get_model_weights(model).trainable
    flat_weights, sizes, shapes = simple_fedavg_tf.flatten_list_of_tensors(
        weights_list)
    new_weights_list = simple_fedavg_tf.reshape_flattened_tensor(
        flat_weights, sizes, shapes)

    for x, y in zip(weights_list, new_weights_list):
      self.assertAllEqual(x, y)


if __name__ == "__main__":
  tf.test.main()
