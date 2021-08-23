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

import tensorflow as tf

from utils.models import emnist_ae_models


class ModelCollectionTest(tf.test.TestCase):

  def test_autoencoder_model_shape(self):
    image = tf.ones([4, 28 * 28])
    model = emnist_ae_models.create_autoencoder_model()
    reconstructed_image = model(image)
    num_model_params = 2837314
    self.assertIsNotNone(reconstructed_image)
    self.assertEqual(reconstructed_image.shape, [4, 28*28])
    self.assertEqual(model.count_params(), num_model_params)

  def test_model_initialization_uses_random_seed(self):

    model_1_with_seed_0 = emnist_ae_models.create_autoencoder_model(seed=0)
    model_2_with_seed_0 = emnist_ae_models.create_autoencoder_model(seed=0)
    model_1_with_seed_1 = emnist_ae_models.create_autoencoder_model(seed=1)
    model_2_with_seed_1 = emnist_ae_models.create_autoencoder_model(seed=1)
    self.assertAllClose(model_1_with_seed_0.weights,
                        model_2_with_seed_0.weights)
    self.assertAllClose(model_1_with_seed_1.weights,
                        model_2_with_seed_1.weights)
    self.assertNotAllClose(model_1_with_seed_0.weights,
                           model_1_with_seed_1.weights)


if __name__ == '__main__':
  tf.test.main()
