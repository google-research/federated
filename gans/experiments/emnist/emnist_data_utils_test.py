# Copyright 2018, Google LLC.
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
"""Test Federated EMNIST dataset utilities."""

import collections

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from gans.experiments.emnist import emnist_data_utils

BATCH_SIZE = 7


def _summarize_model(model):
  model.summary()
  print('\n\n\n')


def _get_example_client_dataset():
  client_data = tff.simulation.datasets.emnist.get_synthetic()
  return client_data.create_tf_dataset_for_client(client_data.client_ids[0])


def _get_example_client_dataset_containing_lowercase():
  example_ds = _get_example_client_dataset()
  example_image = next(iter(example_ds))['pixels'].numpy()
  num_labels = 62
  image_list = [example_image for _ in range(num_labels)]
  label_list = list(range(num_labels))
  synthetic_data = collections.OrderedDict([
      ('label', label_list),
      ('pixels', image_list),
  ])
  return tf.data.Dataset.from_tensor_slices(synthetic_data)


def _compute_dataset_length(dataset):
  return dataset.reduce(0, lambda x, _: x + 1)


class EmnistTest(tf.test.TestCase):

  def test_preprocessed_img_inversion(self):
    raw_images_ds = _get_example_client_dataset()

    # Inversion turned off, average pixel is dark.
    standard_images_ds = emnist_data_utils.preprocess_img_dataset(
        raw_images_ds, invert_imagery=False, batch_size=BATCH_SIZE)
    for batch in iter(standard_images_ds):
      for image in batch:
        self.assertLessEqual(np.average(image), -0.7)

    # Inversion turned on, average pixel is light.
    inverted_images_ds = emnist_data_utils.preprocess_img_dataset(
        raw_images_ds, invert_imagery=True, batch_size=BATCH_SIZE)
    for batch in iter(inverted_images_ds):
      for image in batch:
        self.assertGreaterEqual(np.average(image), 0.7)

  def test_preprocessed_img_labels_are_case_agnostic(self):
    total_num_labels = 62
    raw_dataset = _get_example_client_dataset_containing_lowercase()
    raw_dataset_iterator = iter(raw_dataset)
    num_raw_images = _compute_dataset_length(raw_dataset)
    self.assertEqual(num_raw_images, total_num_labels)

    processed_dataset = emnist_data_utils.preprocess_img_dataset(
        raw_dataset, include_label=True, batch_size=None, shuffle=False)
    processed_dataset_iterator = iter(processed_dataset)
    num_processed_images = _compute_dataset_length(processed_dataset)
    self.assertEqual(num_processed_images, total_num_labels)

    for _ in range(total_num_labels):
      raw_label = next(raw_dataset_iterator)['label']
      if raw_label > 35:
        raw_label = raw_label - 26  # Convert from lowercase to capital

      processed_label = next(processed_dataset_iterator)[1]
      self.assertEqual(raw_label, processed_label)


if __name__ == '__main__':
  tf.test.main()
