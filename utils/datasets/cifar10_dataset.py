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
"""Library for loading and preprocessing CIFAR-10 training and testing data."""

import collections
from typing import Tuple
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow.keras import datasets, layers, models

CIFAR_SHAPE = (32, 32, 3)
TOTAL_FEATURE_SIZE = 32 * 32 * 3
NUM_EXAMPLES_PER_CLIENT = 5000

def load_cifar10_federated(num_clients=10, num_classes=10, alpha=1, train_client_batch_size=20, test_client_batch_size=100):
    '''
    Loads the train dataset into a non iid distribution over clients using the
    sampling method based on LDA, taken from this paper:
    Measuring the Effects of Non-Identical Data Distribution for
    Federated Visual Classification (https://arxiv.org/pdf/1909.06335.pdf).
    '''
    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    client_train_dataset = collections.OrderedDict()
    client_test_dataset = collections.OrderedDict()

    # for each class in the dataset, there are 10 for CIFAR-10
    idx_batch = [[] for _ in range(num_clients)]

    for k in range(num_classes):
        '''
        This will form the final dataset dictionary for each client
        For each class(label) sample in a proportion for each client. 
        The proportion is determined by a dirichlet distribution.
        '''
        label_k = np.where(train_labels.squeeze()==k)[0]
        np.random.shuffle(label_k)
        proportion = np.random.dirichlet(alpha*np.ones((num_clients,)))
        proportion = proportion*(label_k.shape[0])
        proportion = np.cumsum(proportion).astype(int)
        split_labels = np.split(label_k, proportion)
        
        idx_batch = [idx_j + splitk.tolist() for idx_j, splitk in zip(idx_batch, split_labels)]

    num_per_client_test = int((len(test_labels) // (num_clients*test_client_batch_size))*test_client_batch_size)

    for i in range(num_clients):
        client_name = str(i)
        x_train = train_images[np.array(idx_batch[i])]
        y_train = train_labels[np.array(idx_batch[i])].astype('int64').squeeze()
        train_samples_per_client = (x_train.shape[0] // train_client_batch_size) * train_client_batch_size
        x_train, y_train = x_train[:train_samples_per_client], y_train[:train_samples_per_client]
        
        data = collections.OrderedDict((('image', x_train), ('label', y_train)))        
        test_data = collections.OrderedDict((('image', test_images[i*num_per_client_test:(i+1)*num_per_client_test]), ('label', test_labels[i*num_per_client_test:(i+1)*num_per_client_test].astype('int64').squeeze())))
        client_train_dataset[client_name] = data
        client_test_dataset[client_name] = test_data

    train_dataset = tff.simulation.FromTensorSlicesClientData(client_train_dataset)
    test_dataset = tff.simulation.FromTensorSlicesClientData(client_test_dataset)

    return train_dataset, test_dataset


def build_image_map(crop_shape, distort=False):
  """Builds a function that crops and normalizes CIFAR-10 elements.
  The image is first converted to a `tf.float32`, then cropped (according to
  the `distort` argument). Finally, its values are normalized via
  `tf.image.per_image_standardization`.
  Args:
    crop_shape: A tuple (crop_height, crop_width, num_channels) specifying the
      desired crop shape for pre-processing. This tuple cannot have elements
      exceeding (32, 32, 3), element-wise. The element in the last index should
      be set to 3 to maintain the RGB image structure of the elements.
    distort: A boolean indicating whether to distort the image via random crops
      and flips. If set to False, the image is resized to the `crop_shape` via
      `tf.image.resize_with_crop_or_pad`.
  Returns:
    A callable accepting a tensor of shape (32, 32, 3), and performing the
    crops and normalization discussed above.
  """

  if distort:

    def crop_fn(image):
      image = tf.image.random_crop(image, size=crop_shape)
      image = tf.image.random_flip_left_right(image)
      return image

  else:

    def crop_fn(image):
      return tf.image.resize_with_crop_or_pad(
          image, target_height=crop_shape[1], target_width=crop_shape[2])

  def image_map(example):
    image = tf.cast(example['image'], tf.float32)
    image = crop_fn(image)
    image = tf.image.per_image_standardization(image)
    return (image, example['label'])

  return image_map


def create_preprocess_fn(
    num_epochs: int,
    batch_size: int,
    shuffle_buffer_size: int,
    crop_shape: Tuple[int, int, int] = CIFAR_SHAPE,
    distort_image=False,
    num_parallel_calls: int = tf.data.experimental.AUTOTUNE) -> tff.Computation:
  """Creates a preprocessing function for CIFAR-10 client datasets.
  Args:
    num_epochs: An integer representing the number of epochs to repeat the
      client datasets.
    batch_size: An integer representing the batch size on clients.
    shuffle_buffer_size: An integer representing the shuffle buffer size on
      clients. If set to a number <= 1, no shuffling occurs.
    crop_shape: A tuple (crop_height, crop_width, num_channels) specifying the
      desired crop shape for pre-processing. This tuple cannot have elements
      exceeding (32, 32, 3), element-wise. The element in the last index should
      be set to 3 to maintain the RGB image structure of the elements.
    distort_image: A boolean indicating whether to perform preprocessing that
      includes image distortion, including random crops and flips.
    num_parallel_calls: An integer representing the number of parallel calls
      used when performing `tf.data.Dataset.map`.
  Returns:
    A `tff.Computation` performing the preprocessing described above.
  """
  if num_epochs < 1:
    raise ValueError('num_epochs must be a positive integer.')
  if shuffle_buffer_size <= 1:
    shuffle_buffer_size = 1

  feature_dtypes = collections.OrderedDict(
      image=tff.TensorType(tf.uint8, shape=(32, 32, 3)),
      label=tff.TensorType(tf.int64))

  image_map_fn = build_image_map(crop_shape, distort_image)

  @tff.tf_computation(tff.SequenceType(feature_dtypes))
  def preprocess_fn(dataset):
    return (dataset.shuffle(shuffle_buffer_size).repeat(num_epochs).batch(
        batch_size).map(image_map_fn, num_parallel_calls=num_parallel_calls))

  return preprocess_fn


def get_federated_datasets(
    train_client_batch_size: int = 20,
    test_client_batch_size: int = 100,
    train_client_epochs_per_round: int = 1,
    test_client_epochs_per_round: int = 1,
    train_shuffle_buffer_size: int = NUM_EXAMPLES_PER_CLIENT,
    test_shuffle_buffer_size: int = 1,
    crop_shape: Tuple[int, int, int] = CIFAR_SHAPE,
    serializable: bool = False):
  """Loads and preprocesses federated CIFAR100 training and testing sets.
  Args:
    train_client_batch_size: The batch size for all train clients.
    test_client_batch_size: The batch size for all test clients.
    train_client_epochs_per_round: The number of epochs each train client should
      iterate over their local dataset, via `tf.data.Dataset.repeat`. Must be
      set to a positive integer.
    test_client_epochs_per_round: The number of epochs each test client should
      iterate over their local dataset, via `tf.data.Dataset.repeat`. Must be
      set to a positive integer.
    train_shuffle_buffer_size: An integer representing the shuffle buffer size
      (as in `tf.data.Dataset.shuffle`) for each train client's dataset. By
      default, this is set to the largest dataset size among all clients. If set
      to some integer less than or equal to 1, no shuffling occurs.
    test_shuffle_buffer_size: An integer representing the shuffle buffer size
      (as in `tf.data.Dataset.shuffle`) for each test client's dataset. If set
      to some integer less than or equal to 1, no shuffling occurs.
    crop_shape: An iterable of integers specifying the desired crop
      shape for pre-processing. Must be convertable to a tuple of integers
      (CROP_HEIGHT, CROP_WIDTH, NUM_CHANNELS) which cannot have elements that
      exceed (32, 32, 3), element-wise. The element in the last index should be
      set to 3 to maintain the RGB image structure of the elements.
    serializable: Boolean indicating whether the returned datasets are intended
      to be serialized and shipped across RPC channels. If `True`, stateful
      transformations will be disallowed.
  Returns:
    A tuple (cifar_train, cifar_test) of `tff.simulation.ClientData` instances
      representing the federated training and test datasets.
  """
  if not isinstance(crop_shape, collections.Iterable):
    raise TypeError('Argument crop_shape must be an iterable.')
  crop_shape = tuple(crop_shape)
  if len(crop_shape) != 3:
    raise ValueError('The crop_shape must have length 3, corresponding to a '
                     'tensor of shape [height, width, channels].')
  if not isinstance(serializable, bool):
    raise TypeError(
        'serializable must be a Boolean; you passed {} of type {}.'.format(
            serializable, type(serializable)))
  if train_client_epochs_per_round < 1:
    raise ValueError(
        'train_client_epochs_per_round must be a positive integer.')
  if test_client_epochs_per_round < 0:
    raise ValueError('test_client_epochs_per_round must be a positive integer.')
  if train_shuffle_buffer_size <= 1:
    train_shuffle_buffer_size = 1
  if test_shuffle_buffer_size <= 1:
    test_shuffle_buffer_size = 1

  cifar_train, cifar_test = load_cifar10_federated(train_client_batch_size=train_client_batch_size, test_client_batch_size=test_client_batch_size)
  train_crop_shape = (train_client_batch_size,) + crop_shape
  test_crop_shape = (test_client_batch_size,) + crop_shape

  train_preprocess_fn = create_preprocess_fn(
      num_epochs=train_client_epochs_per_round,
      batch_size=train_client_batch_size,
      shuffle_buffer_size=train_shuffle_buffer_size,
      crop_shape=train_crop_shape,
      distort_image=not serializable)

  test_preprocess_fn = create_preprocess_fn(
      num_epochs=test_client_epochs_per_round,
      batch_size=test_client_batch_size,
      shuffle_buffer_size=test_shuffle_buffer_size,
      crop_shape=test_crop_shape,
      distort_image=False)

  cifar_train = cifar_train.preprocess(train_preprocess_fn)
  cifar_test = cifar_test.preprocess(test_preprocess_fn)
  return cifar_train, cifar_test


def get_centralized_datasets(
    train_batch_size: int = 20,
    test_batch_size: int = 20,
    train_shuffle_buffer_size: int = 10000,
    test_shuffle_buffer_size: int = 1,
    crop_shape: Tuple[int, int, int] = CIFAR_SHAPE
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
  """Loads and preprocesses centralized CIFAR100 training and testing sets.
  Args:
    train_batch_size: The batch size for the training dataset.
    test_batch_size: The batch size for the test dataset.
    train_shuffle_buffer_size: An integer specifying the buffer size used to
      shuffle the train dataset via `tf.data.Dataset.shuffle`. If set to an
      integer less than or equal to 1, no shuffling occurs.
    test_shuffle_buffer_size: An integer specifying the buffer size used to
      shuffle the test dataset via `tf.data.Dataset.shuffle`. If set to an
      integer less than or equal to 1, no shuffling occurs.
    crop_shape: An iterable of integers specifying the desired crop
      shape for pre-processing. Must be convertable to a tuple of integers
      (CROP_HEIGHT, CROP_WIDTH, NUM_CHANNELS) which cannot have elements that
      exceed (32, 32, 3), element-wise. The element in the last index should be
      set to 3 to maintain the RGB image structure of the elements.
  Returns:
    A tuple (cifar_train, cifar_test) of `tf.data.Dataset` instances
    representing the centralized training and test datasets.
  """
  try:
    crop_shape = tuple(crop_shape)
  except:
    raise ValueError(
        'Argument crop_shape must be able to coerced into a length 3 tuple.')
  if len(crop_shape) != 3:
    raise ValueError('The crop_shape must have length 3, corresponding to a '
                     'tensor of shape [height, width, channels].')
  if train_shuffle_buffer_size <= 1:
    train_shuffle_buffer_size = 1
  if test_shuffle_buffer_size <= 1:
    test_shuffle_buffer_size = 1

  cifar_train, cifar_test = load_cifar10_federated()
  cifar_train = cifar_train.create_tf_dataset_from_all_clients()
  cifar_test = cifar_test.create_tf_dataset_from_all_clients()

  train_crop_shape = (train_batch_size,) + crop_shape
  test_crop_shape = (test_batch_size,) + crop_shape

  train_preprocess_fn = create_preprocess_fn(
      num_epochs=1,
      batch_size=train_batch_size,
      shuffle_buffer_size=train_shuffle_buffer_size,
      crop_shape=train_crop_shape,
      distort_image=True)
  cifar_train = train_preprocess_fn(cifar_train)

  test_preprocess_fn = create_preprocess_fn(
      num_epochs=1,
      batch_size=test_batch_size,
      shuffle_buffer_size=test_shuffle_buffer_size,
      crop_shape=test_crop_shape,
      distort_image=False)
  cifar_test = test_preprocess_fn(cifar_test)

  return cifar_train, cifar_test