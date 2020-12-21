# Copyright 2020, Google LLC.
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

from fedopt_guide.gld23k_mobilenet import mobilenet_v2


class MobileNetModelTest(tf.test.TestCase):

  def test_alpha_changes_number_parameters(self):
    model1 = mobilenet_v2.create_mobilenet_v2(
        input_shape=(224, 224, 3), num_classes=1000)
    model2 = mobilenet_v2.create_mobilenet_v2(
        input_shape=(224, 224, 3), alpha=0.5, num_classes=1000)
    model3 = mobilenet_v2.create_mobilenet_v2(
        input_shape=(224, 224, 3), alpha=2.0, num_classes=1000)
    self.assertIsInstance(model1, tf.keras.Model)
    self.assertIsInstance(model2, tf.keras.Model)
    self.assertIsInstance(model3, tf.keras.Model)
    self.assertLess(model2.count_params(), model1.count_params())
    self.assertLess(model1.count_params(), model3.count_params())

  def test_num_groups(self):
    model1 = mobilenet_v2.create_mobilenet_v2(
        input_shape=(224, 224, 3), num_classes=1000)
    model2 = mobilenet_v2.create_mobilenet_v2(
        input_shape=(224, 224, 3), num_groups=4, num_classes=1000)
    self.assertIsInstance(model1, tf.keras.Model)
    self.assertIsInstance(model2, tf.keras.Model)
    self.assertEqual(model1.count_params(), model2.count_params())

  def test_pooling_method(self):
    model1 = mobilenet_v2.create_mobilenet_v2(
        input_shape=(224, 224, 3), pooling='avg', num_classes=1000)
    model2 = mobilenet_v2.create_mobilenet_v2(
        input_shape=(224, 224, 3), pooling='max', num_classes=1000)
    self.assertIsInstance(model1, tf.keras.Model)
    self.assertIsInstance(model2, tf.keras.Model)
    self.assertEqual(model1.count_params(), model2.count_params())

  def test_dropout(self):
    model1 = mobilenet_v2.create_mobilenet_v2(
        input_shape=(224, 224, 3), dropout_prob=0.5)
    model2 = mobilenet_v2.create_mobilenet_v2(
        input_shape=(224, 224, 3), dropout_prob=0.2)
    model3 = mobilenet_v2.create_mobilenet_v2(
        input_shape=(224, 224, 3), dropout_prob=None)
    self.assertEqual(len(model1.layers), len(model2.layers))
    self.assertGreater(len(model1.layers), len(model3.layers))

    model1 = mobilenet_v2.create_small_mobilenet_v2(
        input_shape=(64, 64, 3), dropout_prob=0.9)
    model2 = mobilenet_v2.create_small_mobilenet_v2(
        input_shape=(64, 64, 3), dropout_prob=0.2)
    model3 = mobilenet_v2.create_small_mobilenet_v2(
        input_shape=(64, 64, 3), dropout_prob=None)
    self.assertEqual(len(model1.layers), len(model2.layers))
    self.assertGreater(len(model1.layers), len(model3.layers))


if __name__ == '__main__':
  tf.test.main()
