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
"""Tests for emnist_character."""

from absl.testing import parameterized

import tensorflow as tf
import tensorflow_federated as tff

from generalization.tasks import emnist_character


class ConvDropoutTest(tf.test.TestCase, parameterized.TestCase):
  """Test for emnist_character.create_conv_dropout_model."""

  @parameterized.product(num_classes=[10, 47, 62])
  def test_conv_dropout_shape(self, num_classes):
    image = tf.ones([4, 28, 28, 1])
    model = emnist_character.create_conv_dropout_model(num_classes=num_classes)
    logits = model(image)
    self.assertIsNotNone(logits)
    self.assertEqual(logits.shape, [4, num_classes])

  @parameterized.product(num_classes=[10, 47, 62])
  def test_conv_dropout_uses_random_seed(self, num_classes):
    model_1_with_seed_0 = emnist_character.create_conv_dropout_model(
        num_classes, seed=0)
    model_2_with_seed_0 = emnist_character.create_conv_dropout_model(
        num_classes, seed=0)
    model_1_with_seed_1 = emnist_character.create_conv_dropout_model(
        num_classes, seed=1)
    model_2_with_seed_1 = emnist_character.create_conv_dropout_model(
        num_classes, seed=1)
    self.assertAllClose(model_1_with_seed_0.weights,
                        model_2_with_seed_0.weights)
    self.assertAllClose(model_1_with_seed_1.weights,
                        model_2_with_seed_1.weights)
    self.assertNotAllClose(model_1_with_seed_0.weights,
                           model_1_with_seed_1.weights)


class EmnistCharacterClientDataTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('EMNIST-10', True), ('EMNIST-62', False))
  def test_original_emnist_character_client_data_has_same_client_ids(
      self, only_digits):
    train_cd_orig, unpart_cd_orig = tff.simulation.datasets.emnist.load_data(
        only_digits=only_digits, cache_dir=self.get_temp_dir())
    # Assert the only_digits train_cd and val_cd has the same client_ids.
    self.assertCountEqual(train_cd_orig.client_ids, unpart_cd_orig.client_ids)


if __name__ == '__main__':
  tf.test.main()
