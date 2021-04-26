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

from utils.models import stackoverflow_lr_models


class ModelCollectionTest(tf.test.TestCase):

  def test_lr_model_constructs_with_expected_size(self):
    tokens = tf.random.normal([4, 1000])
    model = stackoverflow_lr_models.create_logistic_model(1000, 10)
    predicted_tags = model(tokens)
    num_model_params = 1000*10 + 10
    self.assertIsNotNone(predicted_tags)
    self.assertEqual(predicted_tags.shape, [4, 10])
    self.assertEqual(model.count_params(), num_model_params)

  def test_model_initialization_uses_random_seed(self):
    vocab_size = 1000
    tag_size = 10
    model_1_with_seed_0 = stackoverflow_lr_models.create_logistic_model(
        vocab_size, tag_size, seed=0)
    model_2_with_seed_0 = stackoverflow_lr_models.create_logistic_model(
        vocab_size, tag_size, seed=0)
    model_1_with_seed_1 = stackoverflow_lr_models.create_logistic_model(
        vocab_size, tag_size, seed=1)
    model_2_with_seed_1 = stackoverflow_lr_models.create_logistic_model(
        vocab_size, tag_size, seed=1)
    self.assertAllClose(model_1_with_seed_0.weights,
                        model_2_with_seed_0.weights)
    self.assertAllClose(model_1_with_seed_1.weights,
                        model_2_with_seed_1.weights)
    self.assertNotAllClose(model_1_with_seed_0.weights,
                           model_1_with_seed_1.weights)


if __name__ == '__main__':
  tf.test.main()
