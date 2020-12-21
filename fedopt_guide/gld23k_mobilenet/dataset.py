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
"""Prepare Google Landmark datasets for federated and centralized training."""

import collections
import enum
import functools
from typing import Tuple

import tensorflow as tf
import tensorflow_federated as tff

from tensorflow_models.slim.preprocessing import preprocessing_factory


@enum.unique
class DatasetType(enum.Enum):
  """The type of the dataset used for experiments."""
  GLD23K = 1
  GLD160K = 2


def get_dataset_stats(dataset_type: DatasetType) -> Tuple[int, int]:
  if dataset_type == DatasetType.GLD23K:
    num_classes = 203
    shuffle_buffer_size = 1000
  else:
    num_classes = 2028
    shuffle_buffer_size = 3500
  return num_classes, shuffle_buffer_size


# We have to map a single image at a time (instead of a batch) because the
# original GLD images have various shapes.
def _map_fn(element, is_training, image_size):
  """Preprocesses an image for training/eval using Inception-style networks."""
  preprocess_fn = preprocessing_factory.get_preprocessing(
      'mobilenet_v2', is_training=is_training)
  image = preprocess_fn(element['image/decoded'], image_size, image_size)
  label = element['class']
  return image, label


def _check_positive(name: str, value: int) -> bool:
  """Checks that `value` is positive."""
  if value <= 0:
    raise ValueError(f'Expected a positive value for {name}, found {value}.')


def get_preprocessing_fn(image_size: int, batch_size: int, num_epochs: int,
                         max_elements: int,
                         shuffle_buffer_size: int) -> tff.Computation:
  """Creates a preprocessing function for federated training.

  Args:
    image_size: The height and width of images after preprocessing.
    batch_size: Batch size used for training.
    num_epochs: Number of training epochs.
    max_elements: The maximum number of elements taken from the dataset. It has
      to be a positive value or -1 (which means all elements are taken).
    shuffle_buffer_size: Buffer size used in shuffling.

  Returns:
    A `tff.Computation` that transforms the raw `tf.data.Dataset` of a client
    into a `tf.data.Dataset` that is ready for training.
  """
  _check_positive('image_size', image_size)
  _check_positive('batch_size', batch_size)
  _check_positive('num_epochs', num_epochs)
  _check_positive('shuffle_buffer_size', shuffle_buffer_size)
  if max_elements <= 0 and max_elements != -1:
    raise ValueError('Expected a positive value or -1 for `max_elements`, '
                     f'found {max_elements}.')

  feature_dtypes = collections.OrderedDict()
  feature_dtypes['image/decoded'] = tff.TensorType(
      dtype=tf.uint8, shape=[None, None, None])
  feature_dtypes['class'] = tff.TensorType(dtype=tf.int64, shape=[1])

  @tff.tf_computation(tff.SequenceType(feature_dtypes))
  def preprocessing_fn(dataset: tf.data.Dataset) -> tf.data.Dataset:
    dataset = dataset.map(
        functools.partial(_map_fn, is_training=True, image_size=image_size),
        num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(
            shuffle_buffer_size).take(max_elements).repeat(num_epochs).batch(
                batch_size)
    return dataset

  return preprocessing_fn


def get_centralized_datasets(
    image_size: int,
    batch_size: int,
    shuffle_buffer_size: int = 10000,
    dataset_type: DatasetType = DatasetType.GLD23K
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
  """Creates a train dataset and a test dataset for centralized experiments."""
  train_data, test_data = tff.simulation.datasets.gldv2.load_data(
      gld23k=True if dataset_type == DatasetType.GLD23K else False)
  centralized_train = train_data.create_tf_dataset_from_all_clients().map(
      functools.partial(_map_fn, is_training=True, image_size=image_size),
      num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(
          shuffle_buffer_size).batch(batch_size)
  centralized_test = test_data.map(
      functools.partial(_map_fn, is_training=False, image_size=image_size),
      num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size)
  return centralized_train, centralized_test
