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
"""Library for loading and preprocessing CIFAR-10 training and testing data."""

import collections
from typing import Callable, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_federated as tff

CIFAR_SHAPE = (32, 32, 3)
TOTAL_FEATURE_SIZE = 32 * 32 * 3
NUM_CLASSES = 10
TRAIN_EXAMPLES = 50000
TEST_EXAMPLES = 10000
# Number of training examples per class: 50,000 / 10.
TRAIN_EXAMPLES_PER_LABEL = 5000
# Number of test examples per class: 10,000 / 10.
TEST_EXAMPLES_PER_LABEL = 1000


def load_cifar10_federated(
    dirichlet_parameter: float = 1,
    num_clients: int = 10,
) -> Tuple[tff.simulation.datasets.ClientData,
           tff.simulation.datasets.ClientData]:
  """Construct a federated dataset from the centralized CIFAR-10.

  Sampling based on Dirichlet distribution over categories, following the paper
  Measuring the Effects of Non-Identical Data Distribution for
  Federated Visual Classification (https://arxiv.org/abs/1909.06335).

  Args:
    dirichlet_parameter: Parameter of Dirichlet distribution. Each client
      samples from this Dirichlet to get a multinomial distribution over
      classes. It controls the data heterogeneity of clients. If approaches 0,
      then each client only have data from a single category label. If
      approaches infinity, then the client distribution will approach IID
      partitioning.
    num_clients: The number of clients the examples are going to be partitioned
      on.

  Returns:
    A tuple of `tff.simulation.datasets.ClientData` representing unpreprocessed
    train data and test data.
  """
  train_images, train_labels = tfds.as_numpy(
      tfds.load(
          name='cifar10',
          split='train',
          batch_size=-1,
          as_supervised=True,
      ))
  test_images, test_labels = tfds.as_numpy(
      tfds.load(
          name='cifar10',
          split='test',
          batch_size=-1,
          as_supervised=True,
      ))
  train_clients = collections.OrderedDict()
  test_clients = collections.OrderedDict()

  train_multinomial_vals = []
  test_multinomial_vals = []
  # Each client has a multinomial distribution over classes drawn from a
  # Dirichlet.
  for i in range(num_clients):
    proportion = np.random.dirichlet(dirichlet_parameter *
                                     np.ones(NUM_CLASSES,))
    train_multinomial_vals.append(proportion)
    test_multinomial_vals.append(proportion)

  train_multinomial_vals = np.array(train_multinomial_vals)
  test_multinomial_vals = np.array(test_multinomial_vals)

  train_example_indices = []
  test_indices = []
  for k in range(NUM_CLASSES):
    train_label_k = np.where(train_labels == k)[0]
    np.random.shuffle(train_label_k)
    train_example_indices.append(train_label_k)
    test_label_k = np.where(test_labels == k)[0]
    np.random.shuffle(test_label_k)
    test_indices.append(test_label_k)

  train_example_indices = np.array(train_example_indices)
  test_indices = np.array(test_indices)

  train_client_samples = [[] for _ in range(num_clients)]
  test_client_samples = [[] for _ in range(num_clients)]
  train_count = np.zeros(NUM_CLASSES).astype(int)
  test_count = np.zeros(NUM_CLASSES).astype(int)

  train_examples_per_client = int(TRAIN_EXAMPLES / num_clients)
  test_examples_per_client = int(TEST_EXAMPLES / num_clients)
  for k in range(num_clients):

    for i in range(train_examples_per_client):
      sampled_label = np.argwhere(
          np.random.multinomial(1, train_multinomial_vals[k, :]) == 1)[0][0]
      train_client_samples[k].append(
          train_example_indices[sampled_label, train_count[sampled_label]])
      train_count[sampled_label] += 1
      if train_count[sampled_label] == TRAIN_EXAMPLES_PER_LABEL:
        train_multinomial_vals[:, sampled_label] = 0
        train_multinomial_vals = (
            train_multinomial_vals /
            train_multinomial_vals.sum(axis=1)[:, None])

    for i in range(test_examples_per_client):
      sampled_label = np.argwhere(
          np.random.multinomial(1, test_multinomial_vals[k, :]) == 1)[0][0]
      test_client_samples[k].append(test_indices[sampled_label,
                                                 test_count[sampled_label]])
      test_count[sampled_label] += 1
      if test_count[sampled_label] == TEST_EXAMPLES_PER_LABEL:
        test_multinomial_vals[:, sampled_label] = 0
        test_multinomial_vals = (
            test_multinomial_vals / test_multinomial_vals.sum(axis=1)[:, None])

  for i in range(num_clients):
    client_name = str(i)
    x_train = train_images[np.array(train_client_samples[i])]
    y_train = train_labels[np.array(
        train_client_samples[i])].astype('int64').squeeze()
    train_data = collections.OrderedDict(
        (('image', x_train), ('label', y_train)))
    train_clients[client_name] = train_data

    x_test = test_images[np.array(test_client_samples[i])]
    y_test = test_labels[np.array(
        test_client_samples[i])].astype('int64').squeeze()
    test_data = collections.OrderedDict((('image', x_test), ('label', y_test)))
    test_clients[client_name] = test_data

  train_dataset = tff.simulation.datasets.TestClientData(train_clients)
  test_dataset = tff.simulation.datasets.TestClientData(test_clients)

  return train_dataset, test_dataset


def build_image_map(
    crop_shape: Union[tf.Tensor, Sequence[int]],
    distort: bool = False
) -> Callable[[tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
  """Builds a function that crops and normalizes CIFAR-10 elements.

  The image is first converted to a `tf.float32`, then cropped (according to
  the `distort` argument). Finally, its values are normalized via
  `tf.image.per_image_standardization`.

  Args:
    crop_shape: A tuple (crop_height, crop_width, channels) specifying the
      desired crop shape for pre-processing batches. This cannot exceed (32, 32,
      3) element-wise. The element in the last index should be set to 3 to
      maintain the RGB image structure of the elements.
    distort: A boolean indicating whether to distort the image via random crops
      and flips. If set to False, the image is resized to the `crop_shape` via
      `tf.image.resize_with_crop_or_pad`.

  Returns:
    A callable accepting a tensor and performing the crops and normalization
    discussed above.
  """

  if distort:

    def crop_fn(image):
      image = tf.image.random_crop(image, size=crop_shape)
      image = tf.image.random_flip_left_right(image)
      return image

  else:

    def crop_fn(image):
      return tf.image.resize_with_crop_or_pad(
          image, target_height=crop_shape[0], target_width=crop_shape[1])

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
    num_parallel_calls: int = tf.data.experimental.AUTOTUNE
) -> Callable[[tf.data.Dataset], tf.data.Dataset]:
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
    A callable performing the preprocessing described above.
  """
  if num_epochs < 1:
    raise ValueError('num_epochs must be a positive integer.')
  if shuffle_buffer_size <= 1:
    shuffle_buffer_size = 1

  image_map_fn = build_image_map(crop_shape, distort_image)

  def preprocess_fn(dataset):
    return (
        dataset.shuffle(shuffle_buffer_size).repeat(num_epochs)
        # We map before batching to ensure that the cropping occurs
        # at an image level (eg. we do not perform the same crop on
        # every image within a batch)
        .map(image_map_fn,
             num_parallel_calls=num_parallel_calls).batch(batch_size))

  return preprocess_fn


def get_federated_datasets(train_client_batch_size: int = 20,
                           test_client_batch_size: int = 100,
                           train_client_epochs_per_round: int = 1,
                           test_client_epochs_per_round: int = 1,
                           train_shuffle_buffer_size: int = 1000,
                           test_shuffle_buffer_size: int = 1,
                           crop_shape: Tuple[int, int, int] = CIFAR_SHAPE,
                           serializable: bool = False):
  """Loads and preprocesses federated CIFAR10 training and testing sets.

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
    crop_shape: An iterable of integers specifying the desired crop shape for
      pre-processing. Must be convertable to a tuple of integers (CROP_HEIGHT,
      CROP_WIDTH, NUM_CHANNELS) which cannot have elements that exceed (32, 32,
      3), element-wise. The element in the last index should be set to 3 to
      maintain the RGB image structure of the elements.
    serializable: Boolean indicating whether the returned datasets are intended
      to be serialized and shipped across RPC channels. If `True`, stateful
      transformations will be disallowed.

  Returns:
    A tuple (cifar_train, cifar_test) of `tff.simulation.datasets.ClientData`
    instances representing the federated training and test datasets.
  """
  if not isinstance(crop_shape, collections.abc.Iterable):
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

  cifar_train, cifar_test = load_cifar10_federated()

  train_preprocess_fn = create_preprocess_fn(
      num_epochs=train_client_epochs_per_round,
      batch_size=train_client_batch_size,
      shuffle_buffer_size=train_shuffle_buffer_size,
      crop_shape=crop_shape,
      distort_image=not serializable)

  test_preprocess_fn = create_preprocess_fn(
      num_epochs=test_client_epochs_per_round,
      batch_size=test_client_batch_size,
      shuffle_buffer_size=test_shuffle_buffer_size,
      crop_shape=crop_shape,
      distort_image=False)

  cifar_train = cifar_train.preprocess(train_preprocess_fn)
  cifar_test = cifar_test.preprocess(test_preprocess_fn)
  return cifar_train, cifar_test


def get_centralized_datasets(
    train_batch_size: int = 20,
    test_batch_size: int = 100,
    train_shuffle_buffer_size: int = 10000,
    test_shuffle_buffer_size: int = 1,
    crop_shape: Tuple[int, int, int] = CIFAR_SHAPE
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
  """Loads and preprocesses centralized CIFAR10 training and testing sets.

  Args:
    train_batch_size: The batch size for the training dataset.
    test_batch_size: The batch size for the test dataset.
    train_shuffle_buffer_size: An integer specifying the buffer size used to
      shuffle the train dataset via `tf.data.Dataset.shuffle`. If set to an
      integer less than or equal to 1, no shuffling occurs.
    test_shuffle_buffer_size: An integer specifying the buffer size used to
      shuffle the test dataset via `tf.data.Dataset.shuffle`. If set to an
      integer less than or equal to 1, no shuffling occurs.
    crop_shape: An iterable of integers specifying the desired crop shape for
      pre-processing. Must be convertable to a tuple of integers (CROP_HEIGHT,
      CROP_WIDTH, NUM_CHANNELS) which cannot have elements that exceed (32, 32,
      3), element-wise. The element in the last index should be set to 3 to
      maintain the RGB image structure of the elements.

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

  train_preprocess_fn = create_preprocess_fn(
      num_epochs=1,
      batch_size=train_batch_size,
      shuffle_buffer_size=train_shuffle_buffer_size,
      crop_shape=crop_shape,
      distort_image=True)
  cifar_train = train_preprocess_fn(cifar_train)

  test_preprocess_fn = create_preprocess_fn(
      num_epochs=1,
      batch_size=test_batch_size,
      shuffle_buffer_size=test_shuffle_buffer_size,
      crop_shape=crop_shape,
      distort_image=False)
  cifar_test = test_preprocess_fn(cifar_test)

  return cifar_train, cifar_test
