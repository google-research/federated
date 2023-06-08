# Copyright 2023, Google LLC.
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
"""Tests for data_loaders."""
import math
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow_datasets as tfds

from multi_epoch_dp_matrix_factorization.dp_ftrl.centralized import data_loaders

_KEYS_TO_DATA = {
    'cifar10': {
        'n_classes': 10,
        'n_train_examples': 100,
        'image_shape': (3, 32, 32),
    },
}


def _make_cifar10_dataset():
  train = dict(
      image=np.zeros(shape=[100, 32, 32, 3]),
      label=np.ones(shape=[100], dtype=np.int64),
  )
  test = dict(
      image=np.zeros(shape=[100, 32, 32, 3]),
      label=np.ones(shape=[100], dtype=np.int64),
  )
  return {'train': train, 'test': test}


_KEYS_TO_MOCK_VALUES = {
    'cifar10': _make_cifar10_dataset(),
}


class DataLoadersTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('cifar10', 'cifar10', 5),
  )
  def test_dataset_constructs(self, key, batch_size):
    mock_ds_load = mock.MagicMock()
    mock_ds_load.return_value = _KEYS_TO_MOCK_VALUES[key]

    with mock.patch.object(tfds, 'as_numpy', mock_ds_load), mock.patch.object(
        tfds, 'load'
    ):
      # We patch laod as well to prevent a call out to the network.
      train, val, test, n_classes = data_loaders.get_tfds_data(key, batch_size)
    mock_ds_load.assert_called_once()

    self.assertEqual(n_classes, _KEYS_TO_DATA[key]['n_classes'])
    self.assertLen(
        train, math.ceil(_KEYS_TO_DATA[key]['n_train_examples'] / batch_size)
    )
    for data in [train, val, test]:
      example_batch, example_labels = next(iter(data))
      self.assertIsInstance(example_batch, np.ndarray)
      self.assertIsInstance(example_labels, np.ndarray)
      self.assertEqual(
          example_batch.shape, (batch_size, *_KEYS_TO_DATA[key]['image_shape'])
      )
      self.assertEqual(example_labels.shape, (batch_size,))


if __name__ == '__main__':
  absltest.main()
