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

import functools
from absl.testing import parameterized
import tensorflow as tf
import tensorflow_federated as tff
from fedopt_guide.gld23k_mobilenet import dataset


class DatasetTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('gld23k', True), ('gld160k', False))
  def test_map_fn_returns_desired_image_shape_and_dtype(self, load_gld23k):
    train_data, test_data = tff.simulation.datasets.gldv2.load_data(
        gld23k=load_gld23k)
    client_id = train_data.client_ids[0]
    client_train_data = train_data.create_tf_dataset_for_client(client_id)
    image_size = 224
    client_train_data = client_train_data.map(
        functools.partial(
            dataset._map_fn, is_training=True, image_size=image_size))
    train_image, train_label = next(iter(client_train_data))
    self.assertListEqual(train_image.shape.as_list(),
                         [image_size, image_size, 3])
    self.assertListEqual(train_label.shape.as_list(), [1])
    self.assertEqual(train_image.dtype, tf.float32)
    self.assertEqual(train_label.dtype, tf.int64)
    test_data = test_data.map(
        functools.partial(
            dataset._map_fn, is_training=False, image_size=image_size))
    test_image, test_label = next(iter(test_data))
    self.assertListEqual(test_image.shape.as_list(),
                         [image_size, image_size, 3])
    self.assertListEqual(test_label.shape.as_list(), [1])
    self.assertEqual(test_image.dtype, tf.float32)
    self.assertEqual(test_label.dtype, tf.int64)

  @parameterized.named_parameters(
      ('gld23k_cap64', True, 64), ('gld23k_no_cap', True, -1),
      ('gld160k_cap64', False, 64), ('gld160k_no_cap', False, -1))
  def test_preprocessing_fn_returns_correct_dataset(self, load_gld23k,
                                                    max_elements):
    image_size = 224
    batch_size = 16
    preprocessing_fn = dataset.get_preprocessing_fn(
        image_size=image_size,
        batch_size=batch_size,
        num_epochs=1,
        max_elements=max_elements,
        shuffle_buffer_size=1000)
    _, test_data = tff.simulation.datasets.gldv2.load_data(gld23k=load_gld23k)
    preprocessed_test_data = preprocessing_fn(test_data)
    image_batch, label_batch = next(iter(preprocessed_test_data))
    self.assertListEqual(image_batch.shape.as_list(),
                         [batch_size, image_size, image_size, 3])
    self.assertListEqual(label_batch.shape.as_list(), [batch_size, 1])

  @parameterized.named_parameters(('gld23k', dataset.DatasetType.GLD23K),
                                  ('gld160k', dataset.DatasetType.GLD160K))
  def test_get_centralized_datasets_succeeds(self, dataset_type):
    image_size = 224
    batch_size = 16
    centralized_train, centralized_test = dataset.get_centralized_datasets(
        image_size=image_size, batch_size=batch_size, dataset_type=dataset_type)
    train_image_batch, train_label_batch = next(iter(centralized_train))
    self.assertListEqual(train_image_batch.shape.as_list(),
                         [batch_size, image_size, image_size, 3])
    self.assertListEqual(train_label_batch.shape.as_list(), [batch_size, 1])
    test_image_batch, test_label_batch = next(iter(centralized_test))
    self.assertListEqual(test_image_batch.shape.as_list(),
                         [batch_size, image_size, image_size, 3])
    self.assertListEqual(test_label_batch.shape.as_list(), [batch_size, 1])


if __name__ == '__main__':
  tf.test.main()
