# Copyright 2019, Google LLC.
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

from utils.models import emnist_models


class ModelCollectionTest(tf.test.TestCase, parameterized.TestCase):

  def test_conv_dropout_only_digits_shape(self):
    image = tf.ones([4, 28, 28, 1])
    model = emnist_models.create_conv_dropout_model(only_digits=True)
    logits = model(image)
    self.assertIsNotNone(logits)
    self.assertEqual(logits.shape, [4, 10])

  def test_conv_dropout_shape(self):
    image = tf.ones([3, 28, 28, 1])
    model = emnist_models.create_conv_dropout_model(only_digits=False)
    logits = model(image)

    self.assertIsNotNone(logits)
    self.assertEqual(logits.shape, [3, 62])

  def test_2nn_output_shape(self):
    image = tf.ones([7, 28, 28, 1])
    model = emnist_models.create_two_hidden_layer_model(
        only_digits=False, hidden_units=200)
    logits = model(image)
    self.assertIsNotNone(logits)
    self.assertEqual(logits.shape, [7, 62])

  def test_2nn_number_of_parameters(self):
    model = emnist_models.create_two_hidden_layer_model(
        only_digits=True, hidden_units=200)

    # We calculate the number of parameters based on the fact that given densely
    # connected layers of size n and m with bias units, there are (n+1)m
    # parameters between these layers. The network above should have layers of
    # size 28*28, 200, 200, and 10.
    num_model_params = (28 * 28 + 1) * 200 + 201 * 200 + 201 * 10
    self.assertEqual(model.count_params(), num_model_params)

  def test_1m_cnn_only_digits_shape(self):
    image = tf.random.normal([4, 28, 28, 1])
    model = emnist_models.create_1m_cnn_model(only_digits=True)
    logits = model(image)
    self.assertIsNotNone(logits)
    self.assertEqual(logits.shape, [4, 10])

  def test_1m_cnn_shape(self):
    image = tf.random.normal([3, 28, 28, 1])
    model = emnist_models.create_1m_cnn_model(only_digits=False)
    logits = model(image)

    self.assertIsNotNone(logits)
    self.assertEqual(logits.shape, [3, 62])

  def test_1m_cnn_number_of_parameters(self):
    model = emnist_models.create_1m_cnn_model(only_digits=False)

    # Manually calculate the number of parameters.
    num_model_params = 3 * 3 * 1 * 32 + 32  # 1st 3x3 conv (kernel + bias).
    num_model_params += 3 * 3 * 32 * 64 + 64  # 2nd 3x3 conv (kernel + bias).
    output_size = (28 - 2) // 2 - 2  # conv --> maxpool --> conv.
    output_numel = output_size * output_size * 64
    num_model_params += output_numel * 128 + 128  # 1st dense layer, 128 units.
    num_model_params += 128 * 62 + 62  # output layer, 62 units.

    self.assertEqual(model.count_params(), num_model_params)
    self.assertLess(model.count_params(), 2**20)

  @parameterized.named_parameters(
      ('cnn_dropout', emnist_models.create_conv_dropout_model, False),
      ('cnn_dropout_digits', emnist_models.create_conv_dropout_model, True),
      ('cnn', emnist_models.create_original_fedavg_cnn_model, False),
      ('cnn_digits', emnist_models.create_original_fedavg_cnn_model, True),
      ('2nn', emnist_models.create_two_hidden_layer_model, False),
      ('2nn_digits', emnist_models.create_two_hidden_layer_model, True),
      ('1m_cnn', emnist_models.create_1m_cnn_model, False),
      ('1m_cnn_digits', emnist_models.create_1m_cnn_model, True),
  )
  def test_model_initialization_uses_random_seed(self, model_function,
                                                 only_digits):
    model_1_with_seed_0 = model_function(only_digits=only_digits, seed=0)
    model_2_with_seed_0 = model_function(only_digits=only_digits, seed=0)
    model_1_with_seed_1 = model_function(only_digits=only_digits, seed=1)
    model_2_with_seed_1 = model_function(only_digits=only_digits, seed=1)
    self.assertAllClose(model_1_with_seed_0.weights,
                        model_2_with_seed_0.weights)
    self.assertAllClose(model_1_with_seed_1.weights,
                        model_2_with_seed_1.weights)
    self.assertNotAllClose(model_1_with_seed_0.weights,
                           model_1_with_seed_1.weights)


if __name__ == '__main__':
  tf.test.main()
