# Copyright 2020, The TensorFlow Federated Authors.
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
"""Tests for models."""

import tensorflow as tf
import tensorflow_federated as tff

from shrink_unshrink import models


class SimpleFedavgTfTest(tf.test.TestCase):

  def test_make_big_and_small_stackoverflow_model_fn_model_equality(self):
    train_client_spec = tff.simulation.baselines.ClientSpec(
        num_epochs=3, batch_size=32, max_elements=1000)
    my_task = tff.simulation.baselines.stackoverflow.create_word_prediction_task(
        train_client_spec, use_synthetic_data=False)
    big_model_fn, small_model_fn = models.make_big_and_small_stackoverflow_model_fn(
        my_task,
        big_embedding_size=96,
        big_lstm_size=670,
        small_embedding_size=72,
        small_lstm_size=503)

    tf.random.set_seed(1)
    og_model = my_task.model_fn()
    tf.random.set_seed(1)
    big_model = big_model_fn()
    tf.random.set_seed(1)
    small_model = small_model_fn()

    for x, y in zip(big_model.trainable_variables,
                    og_model.trainable_variables):
      self.assertAllEqual(x.shape, y.shape)

    for x, y in zip(big_model.non_trainable_variables,
                    og_model.non_trainable_variables):
      self.assertAllEqual(x.shape, y.shape)

    self.assertEqual(big_model.input_spec, og_model.input_spec)
    self.assertEqual(small_model.input_spec, og_model.input_spec)

  def test_make_big_and_small_emnist_cnn_model_equality(self):
    train_client_spec = tff.simulation.baselines.ClientSpec(
        num_epochs=3, batch_size=32, max_elements=1000)
    my_task = tff.simulation.baselines.emnist.create_character_recognition_task(
        train_client_spec, use_synthetic_data=True, model_id='cnn')
    big_model_fn, small_model_fn = models.make_big_and_small_emnist_cnn_model_fn(
        my_task,
        big_conv1_filters=32,
        big_conv2_filters=64,
        big_dense_size=512,
        small_conv1_filters=24,
        small_conv2_filters=48,
        small_dense_size=384)

    tf.random.set_seed(1)
    og_model = my_task.model_fn()
    tf.random.set_seed(1)
    big_model = big_model_fn()
    tf.random.set_seed(1)
    small_model = small_model_fn()

    for x, y in zip(big_model.trainable_variables,
                    og_model.trainable_variables):
      self.assertAllEqual(x.shape, y.shape)

    for x, y in zip(big_model.non_trainable_variables,
                    og_model.non_trainable_variables):
      self.assertAllEqual(x.shape, y.shape)

    self.assertEqual(big_model.input_spec, og_model.input_spec)
    self.assertEqual(small_model.input_spec, og_model.input_spec)

  def test_make_big_and_small_emnist_cnn_dropout_model_equality(self):
    train_client_spec = tff.simulation.baselines.ClientSpec(
        num_epochs=3, batch_size=32, max_elements=1000)
    my_task = tff.simulation.baselines.emnist.create_character_recognition_task(
        train_client_spec, use_synthetic_data=True, model_id='cnn_dropout')
    big_model_fn, small_model_fn = models.make_big_and_small_emnist_cnn_dropout_model_fn(
        my_task,
        big_conv1_filters=32,
        big_conv2_filters=64,
        big_dense_size=128,
        small_conv1_filters=24,
        small_conv2_filters=48,
        small_dense_size=96)

    tf.random.set_seed(1)
    og_model = my_task.model_fn()
    tf.random.set_seed(1)
    big_model = big_model_fn()
    tf.random.set_seed(1)
    small_model = small_model_fn()

    for x, y in zip(big_model.trainable_variables,
                    og_model.trainable_variables):
      self.assertAllEqual(x.shape, y.shape)

    for x, y in zip(big_model.non_trainable_variables,
                    og_model.non_trainable_variables):
      self.assertAllEqual(x.shape, y.shape)

    self.assertEqual(big_model.input_spec, og_model.input_spec)
    self.assertEqual(small_model.input_spec, og_model.input_spec)

  def test_make_big_and_small_emnist_cnn_dropout_mfactor_model_equality(self):
    train_client_spec = tff.simulation.baselines.ClientSpec(
        num_epochs=3, batch_size=32, max_elements=1000)
    my_task = tff.simulation.baselines.emnist.create_character_recognition_task(
        train_client_spec, use_synthetic_data=True, model_id='cnn_dropout')
    big_model_fn, small_model_fn = models.make_big_and_small_emnist_cnn_dropout_mfactor_model_fn(
        my_task,
        big_conv1_filters=32,
        big_conv2_filters=64,
        big_dense_size=128,
        small_conv1_filters=16,
        small_conv2_filters=48,
        small_dense_size=64)

    tf.random.set_seed(1)
    og_model = my_task.model_fn()
    tf.random.set_seed(1)
    big_model = big_model_fn()
    tf.random.set_seed(1)
    small_model = small_model_fn()

    for x, y in zip(big_model.trainable_variables,
                    og_model.trainable_variables):
      self.assertAllEqual(tf.shape(x), tf.shape(y))
      self.assertAllEqual(x.shape, y.shape)

    for x, y in zip(big_model.non_trainable_variables,
                    og_model.non_trainable_variables):
      self.assertAllEqual(tf.shape(x), tf.shape(y))
      self.assertAllEqual(x.shape, y.shape)

    self.assertEqual(big_model.input_spec, og_model.input_spec)
    self.assertEqual(small_model.input_spec, og_model.input_spec)


if __name__ == '__main__':
  tf.test.main()
