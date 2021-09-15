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

from absl import logging
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from shrink_unshrink import models
from shrink_unshrink import simple_fedavg_tf
from shrink_unshrink import simple_fedavg_tff


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

  def test_build_qr_projection_matrix(self):
    seed = 1
    idx = 2

    desired_shape = tf.shape(tf.ones(shape=(3, 5)))
    projection_matrix = simple_fedavg_tf.build_qr_projection_matrix(
        seed=(seed, idx), desired_shape=desired_shape, is_left_multiply=True)
    self.assertAllEqual(tf.shape(projection_matrix), desired_shape)
    self.assertDTypeEqual(projection_matrix, np.float32)
    self.assertAllClose(projection_matrix @ tf.transpose(projection_matrix),
                        tf.eye(3))

    desired_shape = tf.shape(tf.ones(shape=(5, 3)))
    projection_matrix = simple_fedavg_tf.build_qr_projection_matrix(
        seed=(seed, idx), desired_shape=desired_shape, is_left_multiply=False)
    self.assertAllEqual(tf.shape(projection_matrix), desired_shape)
    self.assertDTypeEqual(projection_matrix, np.float32)
    self.assertAllClose(
        tf.transpose(projection_matrix) @ projection_matrix, tf.eye(3))

  def test_build_normal_projection_matrix(self):
    seed = 1
    idx = 2

    desired_shape = tf.shape(tf.ones(shape=(3, 5)))
    projection_matrix = simple_fedavg_tf.build_normal_projection_matrix(
        seed=(seed, idx), desired_shape=desired_shape, is_left_multiply=True)

    latent_dim = tf.cast(desired_shape[0], tf.float32)
    og_projection_matrix = tf.random.stateless_normal(
        shape=desired_shape, seed=(seed, idx), stddev=1 / tf.sqrt(latent_dim))
    self.assertAllEqual(projection_matrix, og_projection_matrix)
    self.assertDTypeEqual(projection_matrix, np.float32)
    self.assertAllEqual(tf.shape(projection_matrix), desired_shape)

    desired_shape = tf.shape(tf.ones(shape=(5, 3)))
    projection_matrix = simple_fedavg_tf.build_normal_projection_matrix(
        seed=(seed, idx), desired_shape=desired_shape, is_left_multiply=False)

    latent_dim = tf.cast(desired_shape[1], tf.float32)
    og_projection_matrix = tf.random.stateless_normal(
        shape=desired_shape, seed=(seed, idx), stddev=1 / tf.sqrt(latent_dim))
    self.assertAllEqual(projection_matrix, og_projection_matrix)
    self.assertDTypeEqual(projection_matrix, np.float32)
    self.assertAllEqual(tf.shape(projection_matrix), desired_shape)

  def test_build_dropout_projection_matrix(self):
    seed = 1
    idx = 2

    desired_shape = tf.shape(tf.ones(shape=(3, 5)))
    projection_matrix = simple_fedavg_tf.build_dropout_projection_matrix(
        seed=(seed, idx), desired_shape=desired_shape, is_left_multiply=True)
    self.assertAllLessEqual(tf.reduce_sum(projection_matrix, axis=0), 1)
    self.assertAllLessEqual(tf.reduce_sum(projection_matrix, axis=1), 1)
    self.assertAllGreaterEqual(tf.reduce_sum(projection_matrix, axis=1), 1)
    self.assertEqual(tf.reduce_sum(projection_matrix), 3)
    self.assertDTypeEqual(projection_matrix, np.float32)
    self.assertAllEqual(tf.shape(projection_matrix), desired_shape)

    desired_shape = tf.shape(tf.ones(shape=(5, 3)))
    projection_matrix = simple_fedavg_tf.build_dropout_projection_matrix(
        seed=(seed, idx), desired_shape=desired_shape, is_left_multiply=False)
    self.assertAllLessEqual(tf.reduce_sum(projection_matrix, axis=0), 1)
    self.assertAllGreaterEqual(tf.reduce_sum(projection_matrix, axis=0), 1)
    self.assertAllLessEqual(tf.reduce_sum(projection_matrix, axis=1), 1)
    self.assertEqual(tf.reduce_sum(projection_matrix), 3)
    self.assertDTypeEqual(projection_matrix, np.float32)
    self.assertAllEqual(tf.shape(projection_matrix), desired_shape)

  def test_make_big_and_small_emnist_cnn_model(self):
    train_client_spec = tff.simulation.baselines.ClientSpec(
        num_epochs=3, batch_size=32, max_elements=1000)
    my_task = tff.simulation.baselines.emnist.create_character_recognition_task(
        train_client_spec, use_synthetic_data=True, model_id="cnn")
    big_model_fn, small_model_fn = models.make_big_and_small_emnist_cnn_model_fn(
        my_task,
        big_conv1_filters=32,
        big_conv2_filters=64,
        big_dense_size=512,
        small_conv1_filters=24,
        small_conv2_filters=48,
        small_dense_size=384)

    model = big_model_fn()
    model_weights = simple_fedavg_tf.get_model_weights(model)
    server_optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)
    simple_fedavg_tff._initialize_optimizer_vars(model, server_optimizer)
    server_state = simple_fedavg_tf.ServerState(
        model_weights=model_weights,
        optimizer_state=server_optimizer.variables(),
        round_num=0)

    whimsy_server_weights = simple_fedavg_tf.get_model_weights(
        big_model_fn()).trainable
    whimsy_client_weights = simple_fedavg_tf.get_model_weights(
        small_model_fn()).trainable

    # left_mask = [-1, -1, 0, -1, 2, -1, 3, -1]
    left_mask = [-1, -1, 0, -1, 1000, -1, 3, -1]
    right_mask = [0, 0, 1, 1, 3, 3, -1, -1]

    left_maskval_to_projmat_dict = simple_fedavg_tf.create_left_maskval_to_projmat_dict(
        seed=1,
        whimsy_server_weights=whimsy_server_weights,
        whimsy_client_weights=whimsy_client_weights,
        left_mask=left_mask,
        right_mask=right_mask,
        build_projection_matrix=simple_fedavg_tf.build_normal_projection_matrix)

    new_server_state = simple_fedavg_tf.project_server_weights(
        server_state, left_maskval_to_projmat_dict, left_mask, right_mask)

    weights_delta = new_server_state.model_weights.trainable
    client_ouput = simple_fedavg_tf.ClientOutput(weights_delta, 1, 1, 1)
    final_client_output = simple_fedavg_tf.unproject_client_weights(
        client_ouput, left_maskval_to_projmat_dict, left_mask, right_mask)

    for _, z in enumerate(
        zip(final_client_output.weights_delta,
            server_state.model_weights.trainable)):
      x, y = z
      self.assertAllEqual(tf.shape(x), tf.shape(y))

  def test_make_big_and_small_stackoverflow_model(self):
    train_client_spec = tff.simulation.baselines.ClientSpec(
        num_epochs=3, batch_size=32, max_elements=1000)
    my_task = tff.simulation.baselines.stackoverflow.create_word_prediction_task(
        train_client_spec, use_synthetic_data=True)
    big_rnn_model, small_rnn_model = models.make_big_and_small_stackoverflow_model_fn(
        my_task)

    model = big_rnn_model()
    model_weights = simple_fedavg_tf.get_model_weights(model)
    server_optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)
    simple_fedavg_tff._initialize_optimizer_vars(model, server_optimizer)
    server_state = simple_fedavg_tf.ServerState(
        model_weights=model_weights,
        optimizer_state=server_optimizer.variables(),
        round_num=0)

    whimsy_server_weights = simple_fedavg_tf.get_model_weights(
        big_rnn_model()).trainable
    whimsy_client_weights = simple_fedavg_tf.get_model_weights(
        small_rnn_model()).trainable

    left_mask = [-1, 0, 2, -1, 2, -1, 0, -1]
    right_mask = [0, 1, 1, 1, 0, 0, -1, -1]

    left_maskval_to_projmat_dict = simple_fedavg_tf.create_left_maskval_to_projmat_dict(
        seed=1,
        whimsy_server_weights=whimsy_server_weights,
        whimsy_client_weights=whimsy_client_weights,
        left_mask=left_mask,
        right_mask=right_mask,
        build_projection_matrix=simple_fedavg_tf.build_normal_projection_matrix)

    new_server_state = simple_fedavg_tf.project_server_weights(
        server_state, left_maskval_to_projmat_dict, left_mask, right_mask)

    weights_delta = new_server_state.model_weights.trainable
    client_ouput = simple_fedavg_tf.ClientOutput(weights_delta, 1, 1, 1)
    final_client_output = simple_fedavg_tf.unproject_client_weights(
        client_ouput, left_maskval_to_projmat_dict, left_mask, right_mask)

    for idx, z in enumerate(
        zip(final_client_output.weights_delta,
            server_state.model_weights.trainable)):
      x, y = z
      self.assertAllEqual(tf.shape(x), tf.shape(y))
      if idx == 3:  # this lstm bias term is not initialized at 0
        self.assertNotAllClose(x, tf.zeros_like(x))
        self.assertNotAllClose(y, tf.zeros_like(y))
        self.assertNotAllClose(x, y)
      elif idx == 5:  # bias terms are initialized at 0
        self.assertAllEqual(x, y)
        self.assertAllEqual(x, tf.zeros_like(x))
      elif idx == 7:  # bias terms are initialized at 0
        self.assertAllEqual(x, y)
        self.assertAllEqual(x, tf.zeros_like(x))
      else:  # the projection
        self.assertNotAllClose(x, y)

  def test_make_big_and_big_stackoverflow_model(self):
    train_client_spec = tff.simulation.baselines.ClientSpec(
        num_epochs=3, batch_size=32, max_elements=1000)
    my_task = tff.simulation.baselines.stackoverflow.create_word_prediction_task(
        train_client_spec, use_synthetic_data=True)
    big_rnn_model, _ = models.make_big_and_small_stackoverflow_model_fn(my_task)

    model = big_rnn_model()
    model_weights = simple_fedavg_tf.get_model_weights(model)
    server_optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)
    simple_fedavg_tff._initialize_optimizer_vars(model, server_optimizer)
    server_state = simple_fedavg_tf.ServerState(
        model_weights=model_weights,
        optimizer_state=server_optimizer.variables(),
        round_num=0)

    whimsy_server_weights = simple_fedavg_tf.get_model_weights(
        big_rnn_model()).trainable
    whimsy_client_weights = simple_fedavg_tf.get_model_weights(
        big_rnn_model()).trainable

    left_mask = [-1, 0, 2, -1, 2, -1, 0, -1]
    right_mask = [0, 1, 1, 1, 0, 0, -1, -1]

    left_maskval_to_projmat_dict = simple_fedavg_tf.create_left_maskval_to_projmat_dict(
        seed=1,
        whimsy_server_weights=whimsy_server_weights,
        whimsy_client_weights=whimsy_client_weights,
        left_mask=left_mask,
        right_mask=right_mask,
        build_projection_matrix=simple_fedavg_tf.build_normal_projection_matrix)

    new_server_state = simple_fedavg_tf.project_server_weights(
        server_state, left_maskval_to_projmat_dict, left_mask, right_mask)

    weights_delta = new_server_state.model_weights.trainable
    client_ouput = simple_fedavg_tf.ClientOutput(weights_delta, 1, 1, 1)
    final_client_output = simple_fedavg_tf.unproject_client_weights(
        client_ouput, left_maskval_to_projmat_dict, left_mask, right_mask)

    for idx, z in enumerate(
        zip(final_client_output.weights_delta,
            server_state.model_weights.trainable)):
      x, y = z
      self.assertAllEqual(tf.shape(x), tf.shape(y))
      if idx == 3:  # this lstm bias term is not initialized at 0
        self.assertNotAllClose(x, tf.zeros_like(x))
        self.assertNotAllClose(y, tf.zeros_like(y))
        self.assertNotAllClose(x, y)
      elif idx == 5:  # bias terms are initialized at 0
        self.assertAllEqual(x, y)
        self.assertAllEqual(x, tf.zeros_like(x))
      elif idx == 7:  # bias terms are initialized at 0
        self.assertAllEqual(x, y)
        self.assertAllEqual(x, tf.zeros_like(x))
      else:  # the projection
        self.assertNotAllClose(x, y)

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

    train_client_spec = tff.simulation.baselines.ClientSpec(
        num_epochs=3, batch_size=32, max_elements=1000)
    my_task = tff.simulation.baselines.stackoverflow.create_word_prediction_task(
        train_client_spec, use_synthetic_data=True)
    big_model_fn, small_model_fn = models.make_big_and_small_stackoverflow_model_fn(
        my_task)
    big_model = big_model_fn()
    small_model = small_model_fn()

    weights_list = simple_fedavg_tf.get_model_weights(big_model).trainable
    flat_weights, sizes, shapes = simple_fedavg_tf.flatten_list_of_tensors(
        weights_list)
    new_weights_list = simple_fedavg_tf.reshape_flattened_tensor(
        flat_weights, sizes, shapes)

    for x, y in zip(weights_list, new_weights_list):
      self.assertAllEqual(x, y)

    weights_list = simple_fedavg_tf.get_model_weights(small_model).trainable
    flat_weights, sizes, shapes = simple_fedavg_tf.flatten_list_of_tensors(
        weights_list)
    new_weights_list = simple_fedavg_tf.reshape_flattened_tensor(
        flat_weights, sizes, shapes)

    for x, y in zip(weights_list, new_weights_list):
      self.assertAllEqual(x, y)

  def test_project_and_unproject_server_weights(self):
    model = _big_dense_model_fn()
    model_weights = simple_fedavg_tf.get_model_weights(model)
    server_optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)
    simple_fedavg_tff._initialize_optimizer_vars(model, server_optimizer)
    server_state = simple_fedavg_tf.ServerState(
        model_weights=model_weights,
        optimizer_state=server_optimizer.variables(),
        round_num=0)

    whimsy_server_weights = simple_fedavg_tf.get_model_weights(
        _big_dense_model_fn()).trainable
    whimsy_client_weights = simple_fedavg_tf.get_model_weights(
        _small_dense_model_fn()).trainable
    left_mask = [-1, -1, 0, -1, 1, -1]
    right_mask = [0, 0, 1, 1, -1, -1]
    left_maskval_to_projmat_dict = simple_fedavg_tf.create_left_maskval_to_projmat_dict(
        seed=1,
        whimsy_server_weights=whimsy_server_weights,
        whimsy_client_weights=whimsy_client_weights,
        left_mask=left_mask,
        right_mask=right_mask,
        build_projection_matrix=simple_fedavg_tf.build_normal_projection_matrix)
    ones_left_maskval_to_projmat_dict = {
        k: tf.ones_like(v) for k, v in left_maskval_to_projmat_dict.items()
    }
    _ = simple_fedavg_tf.project_server_weights(server_state,
                                                left_maskval_to_projmat_dict,
                                                left_mask, right_mask)

    ones_server_state = simple_fedavg_tf.project_server_weights(
        server_state, ones_left_maskval_to_projmat_dict, left_mask, right_mask)

    ones_weight_lst = ones_server_state.model_weights.trainable
    self.assertAllEqual(ones_weight_lst[0], tf.ones((2, 3)) * 30)
    self.assertAllEqual(ones_weight_lst[1], tf.ones((3)) * 30)
    self.assertAllEqual(ones_weight_lst[2], tf.ones((3, 4)) * 1200)
    self.assertAllEqual(ones_weight_lst[3], tf.ones((4)) * 40)
    self.assertAllEqual(ones_weight_lst[4], tf.ones((4, 5)) * 40)
    self.assertAllEqual(ones_weight_lst[5], tf.ones((5)))

    weights_delta = ones_weight_lst
    client_ouput = simple_fedavg_tf.ClientOutput(weights_delta, 1, 1, 1)
    _ = simple_fedavg_tf.unproject_client_weights(client_ouput,
                                                  left_maskval_to_projmat_dict,
                                                  left_mask, right_mask)

    ones_client_output = simple_fedavg_tf.unproject_client_weights(
        client_ouput, ones_left_maskval_to_projmat_dict, left_mask, right_mask)
    ones_weight_delta = ones_client_output.weights_delta
    self.assertAllEqual(ones_weight_delta[0], tf.ones((2, 30)) * 90)
    self.assertAllEqual(ones_weight_delta[1], tf.ones((30)) * 90)
    self.assertAllEqual(ones_weight_delta[2], tf.ones((30, 40)) * 14400)
    self.assertAllEqual(ones_weight_delta[3], tf.ones((40)) * 160)
    self.assertAllEqual(ones_weight_delta[4], tf.ones((40, 5)) * 160)
    self.assertAllEqual(ones_weight_delta[5], tf.ones((5)))

    for x, y in zip(ones_weight_delta, whimsy_server_weights):
      self.assertAllEqual(tf.shape(x), tf.shape(y))

  def test_project_and_unproject_server_weights_rnn(self):
    train_client_spec = tff.simulation.baselines.ClientSpec(
        num_epochs=3, batch_size=32, max_elements=1000)
    my_task = tff.simulation.baselines.stackoverflow.create_word_prediction_task(
        train_client_spec, use_synthetic_data=True)
    big_model_fn, small_model_fn = models.make_big_and_small_stackoverflow_model_fn(
        my_task)

    big_model = big_model_fn()
    small_model = small_model_fn()

    model_weights = simple_fedavg_tf.get_model_weights(big_model)
    server_optimizer = tf.keras.optimizers.SGD(learning_rate=1.0)
    simple_fedavg_tff._initialize_optimizer_vars(big_model, server_optimizer)
    server_state = simple_fedavg_tf.ServerState(
        model_weights=model_weights,
        optimizer_state=server_optimizer.variables(),
        round_num=0)

    whimsy_server_weights = simple_fedavg_tf.get_model_weights(
        big_model).trainable
    whimsy_client_weights = simple_fedavg_tf.get_model_weights(
        small_model).trainable
    left_mask = [-1, 0, 2, -1, 2, -1, 0, -1]
    right_mask = [0, 1, 1, 1, 0, 0, -1, -1]
    left_maskval_to_projmat_dict = simple_fedavg_tf.create_left_maskval_to_projmat_dict(
        seed=1,
        whimsy_server_weights=whimsy_server_weights,
        whimsy_client_weights=whimsy_client_weights,
        left_mask=left_mask,
        right_mask=right_mask,
        build_projection_matrix=simple_fedavg_tf.build_normal_projection_matrix)

    new_server_state = simple_fedavg_tf.project_server_weights(
        server_state, left_maskval_to_projmat_dict, left_mask, right_mask)
    new_weights = new_server_state.model_weights.trainable

    for x, y in zip(whimsy_client_weights, new_weights):
      self.assertAllEqual(tf.shape(x), tf.shape(y))

    weights_delta = new_weights
    client_ouput = simple_fedavg_tf.ClientOutput(weights_delta, 1, 1, 1)
    next_client_ouput = simple_fedavg_tf.unproject_client_weights(
        client_ouput, left_maskval_to_projmat_dict, left_mask, right_mask)

    next_weights_delta = next_client_ouput.weights_delta

    for x, y in zip(next_weights_delta, whimsy_server_weights):
      self.assertAllEqual(tf.shape(x), tf.shape(y))

  def test_flattening_and_reshaping(self):
    weights_lst = simple_fedavg_tf.get_model_weights(
        _big_dense_model_fn()).trainable
    flat_mask = simple_fedavg_tf.get_flat_mask(weights_lst)
    self.assertAllEqual(flat_mask, [0, 1, 0, 1, 0, 1])
    temp_weights_lst = simple_fedavg_tf.reshape_flattened_weights(
        weights_lst, flat_mask)

    self.assertEqual(len(weights_lst), len(temp_weights_lst))
    for idx in range(len(temp_weights_lst)):
      self.assertEqual(tf.rank(temp_weights_lst[idx]), 2)

    new_weights_lst = simple_fedavg_tf.flatten_reshaped_weights(
        temp_weights_lst, flat_mask)

    self.assertEqual(len(weights_lst), len(new_weights_lst))
    for idx in range(len(weights_lst)):
      self.assertAllEqual(weights_lst[idx], new_weights_lst[idx])

    train_client_spec = tff.simulation.baselines.ClientSpec(
        num_epochs=3, batch_size=32, max_elements=1000)
    my_task = tff.simulation.baselines.stackoverflow.create_word_prediction_task(
        train_client_spec, use_synthetic_data=True)
    big_model_fn, small_model_fn = models.make_big_and_small_stackoverflow_model_fn(
        my_task)
    big_model = big_model_fn()
    small_model = small_model_fn()
    del small_model

    weights_lst = simple_fedavg_tf.get_model_weights(big_model).trainable
    flat_mask = simple_fedavg_tf.get_flat_mask(weights_lst)
    self.assertAllEqual(flat_mask, [0, 0, 0, 1, 0, 1, 0, 1])
    temp_weights_lst = simple_fedavg_tf.reshape_flattened_weights(
        weights_lst, flat_mask)

    self.assertEqual(len(weights_lst), len(temp_weights_lst))
    for idx in range(len(temp_weights_lst)):
      self.assertEqual(tf.rank(temp_weights_lst[idx]), 2)
      self.assertEqual(len(tf.shape(temp_weights_lst[idx])), 2)

    new_weights_lst = simple_fedavg_tf.flatten_reshaped_weights(
        temp_weights_lst, flat_mask)

    self.assertEqual(len(weights_lst), len(new_weights_lst))
    for idx in range(len(weights_lst)):
      self.assertAllEqual(weights_lst[idx], new_weights_lst[idx])

  def test_create_left_maskval_to_projmat_dict(self):
    logging.info("starting test case")
    whimsy_server_weights = simple_fedavg_tf.get_model_weights(
        _big_dense_model_fn()).trainable
    whimsy_client_weights = simple_fedavg_tf.get_model_weights(
        _small_dense_model_fn()).trainable

    left_mask = [-1, -1, 0, -1, 1, -1]
    right_mask = [0, 0, 1, 1, -1, -1]
    logging.info("starting to retreive left_maskval_to_projmat_dict")
    left_maskval_to_projmat_dict = simple_fedavg_tf.create_left_maskval_to_projmat_dict(
        seed=1,
        whimsy_server_weights=whimsy_server_weights,
        whimsy_client_weights=whimsy_client_weights,
        left_mask=left_mask,
        right_mask=right_mask,
        build_projection_matrix=simple_fedavg_tf.build_normal_projection_matrix)
    logging.info("retreived left_maskval_to_projmat_dict")
    self.assertEqual(left_maskval_to_projmat_dict[-1], 1)
    self.assertAllEqual(
        tf.shape(left_maskval_to_projmat_dict[0]), tf.convert_to_tensor([3,
                                                                         30]))
    self.assertAllEqual(
        tf.shape(left_maskval_to_projmat_dict[1]), tf.convert_to_tensor([4,
                                                                         40]))

    left_mask = [-1, -1, 3, -1, 6, -1]
    right_mask = [1, 2, 4, 5, -1, -1]
    left_maskval_to_projmat_dict = simple_fedavg_tf.create_left_maskval_to_projmat_dict(
        seed=1,
        whimsy_server_weights=whimsy_server_weights,
        whimsy_client_weights=whimsy_client_weights,
        left_mask=left_mask,
        right_mask=right_mask,
        build_projection_matrix=simple_fedavg_tf.build_normal_projection_matrix)
    assert left_maskval_to_projmat_dict[-1] == 1
    self.assertAllEqual(
        tf.shape(left_maskval_to_projmat_dict[1]), tf.convert_to_tensor([3,
                                                                         30]))
    self.assertAllEqual(
        tf.shape(left_maskval_to_projmat_dict[2]), tf.convert_to_tensor([3,
                                                                         30]))
    self.assertAllEqual(
        tf.shape(left_maskval_to_projmat_dict[3]), tf.convert_to_tensor([3,
                                                                         30]))
    self.assertNotAllClose(left_maskval_to_projmat_dict[1],
                           left_maskval_to_projmat_dict[2])
    self.assertNotAllClose(left_maskval_to_projmat_dict[1],
                           left_maskval_to_projmat_dict[3])
    self.assertNotAllClose(left_maskval_to_projmat_dict[2],
                           left_maskval_to_projmat_dict[3])

    self.assertAllEqual(
        tf.shape(left_maskval_to_projmat_dict[4]), tf.convert_to_tensor([4,
                                                                         40]))
    self.assertAllEqual(
        tf.shape(left_maskval_to_projmat_dict[5]), tf.convert_to_tensor([4,
                                                                         40]))
    self.assertAllEqual(
        tf.shape(left_maskval_to_projmat_dict[6]), tf.convert_to_tensor([4,
                                                                         40]))
    self.assertNotAllClose(left_maskval_to_projmat_dict[4],
                           left_maskval_to_projmat_dict[5])
    self.assertNotAllClose(left_maskval_to_projmat_dict[4],
                           left_maskval_to_projmat_dict[6])
    self.assertNotAllClose(left_maskval_to_projmat_dict[5],
                           left_maskval_to_projmat_dict[6])


if __name__ == "__main__":
  tf.test.main()
