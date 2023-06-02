# Copyright 2022, Google LLC.
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
"""Data preprocessing functions for Landmarks."""

import collections
import functools
from typing import OrderedDict, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from personalization_benchmark.cross_device import constants
from personalization_benchmark.cross_device.datasets import emnist
from personalization_benchmark.cross_device.datasets import mobilenet_v2

# We use 128 as the image size (not 224 as in https://arxiv.org/abs/2003.08082)
# for faster experiments.
_IMAGE_SIZE = 128
_NUM_GROUPS = 8
_NUM_CLASSES = 2028
_SHUFFLE_BUFFER_SIZE = 3500  # The largest client has at most 3500 examples.
# We follow the setup in https://arxiv.org/abs/2003.08082, and use 64 to limit
# the maximum number of examples a client can contribute in a *training* round
# (note that we do not apply this limitation for validation and test clients).
_MAX_ELEMENTS_PER_CLIENT = 64
# GLD has 1262 clients. We split into 1112/50/100 for train/validation/test.
_NUM_VALID_CLIENTS = 50
_NUM_TEST_CLIENTS = 100
_ACCURACY_NAME = 'sparse_categorical_accuracy'


def _lazy_import_slim_inception_preprocess_lib():
  r"""Lazily imports `inception_preprocessing`.

  This is an external module and must be downloaded and placed under the path
  `personalization_benchmark/cross_device/datasets/`:

  wget https://raw.githubusercontent.com/tensorflow/models/master/research/slim\
      /preprocessing/inception_preprocessing.py

  Returns:
    The slim `inception_preprocessing` module.
  """

  try:
    from personalization_benchmark.cross_device.datasets import inception_preprocessing as preprocess_lib
    return preprocess_lib
  except ImportError as import_error:
    raise ImportError(
        'Cannot import `inception_preprocessing`. Please download '
        'https://raw.githubusercontent.com/tensorflow/models/master/research/slim/preprocessing/inception_preprocessing.py'
        'and place under `personalization_benchmark/cross_device/datasets/`.'
    ) from import_error


def _get_synthetic_landmark_data() -> tf.data.Dataset:
  """Returns a dataset of 3 elements matching the structure of GLD.

  Returns:
    A dataset of 3 elements that matches the structure of the data produced by
    `tff.simulation.datasets.gldv2.load_data`. Each element has keys
    `image/decoded` and `class`. Note that the created synethetic images have
    the same shape, but the actual images can have different shapes.
  """
  batch = collections.OrderedDict([
      ('image/decoded', tf.ones(shape=(3, 600, 800, 3), dtype=tf.uint8)),
      ('class', tf.ones(shape=(3, 1), dtype=tf.int64))
  ])
  return tf.data.Dataset.from_tensor_slices(batch)


# We have to map a single image at a time (instead of a batch) because the
# original GLD images have various shapes.
def _map_fn(element: OrderedDict[str, tf.Tensor],
            is_training: bool) -> Tuple[tf.Tensor, tf.Tensor]:
  """Preprocesses an image for training/eval using Inception-style networks."""
  preprocess_lib = _lazy_import_slim_inception_preprocess_lib()
  image = preprocess_lib.preprocess_image(
      element['image/decoded'],
      _IMAGE_SIZE,
      _IMAGE_SIZE,
      is_training=is_training)
  # When `is_training=False`, the image shape is [image_size, image_size, None],
  # which is incompatible with shape expected by finetuning eval computation.
  image = tf.reshape(image, [_IMAGE_SIZE, _IMAGE_SIZE, 3])
  label = element['class']
  return image, label


def create_model_and_data(
    num_local_epochs: int, train_batch_size: int, use_synthetic_data: bool
) -> Tuple[constants.ModelFnType, constants.FederatedDatasetsType,
           constants.ProcessFnType, constants.SplitDataFnType, str]:
  """Creates model, datasets, and processing functions for GLD.

  Args:
    num_local_epochs: Number of local epochs performed by a client in a round of
      FedAvg training. It is used in the returned `train_preprocess_fn`, which
      contains an operation `train_dataset.repeat(num_local_epochs)`.
    train_batch_size: Batch size used in FedAvg training. It is also used in the
      returned `train_preprocess_fn`, which contains an operation
      `train_dataset.batch(train_batch_size)`.
    use_synthetic_data: Whether to use synthetic data. This should only be set
      to True for debugging and testing purposes.

  Returns:
    1. A no-argument function that returns a `tff.learning.Model`.
    2. An `OrderedDict` of three federated datasets with keys being
      `constants.TRAIN_CLIENTS_KEY`, `constants.VALID_CLIENTS_KEY` and
      `constants.TEST_CLIENT_KEY`.
    3. A dataset preprocess function for per-client dataset in FedAvg training.
      Batching and repeat operations are contained.
    4. A dataset split function for per-client dataset in evaluation (including
      both validation clients and test clients). Specifically, each client's
      examples are split into two equal-sized unbatched datasets (a
      personalization dataset and a test dataset).The personalization dataset is
      used for finetuning the model in `finetuning_trainer` or choosing the best
      model in `hypcluster_trainer`.
    5. The accuracy name key stored in the evaluation metrics. It will be used
      in postprocessing the validation clients and test clients metrics in the
      `finetuning_trainer`.
  """
  if use_synthetic_data:
    raw_client_data = tff.simulation.datasets.ClientData.from_clients_and_tf_fn(
        client_ids=['synthetic'],
        serializable_dataset_fn=lambda _: _get_synthetic_landmark_data())
  else:
    raw_client_data, _ = tff.simulation.datasets.gldv2.load_data(gld23k=False)
  train_map_fn = lambda element: _map_fn(element, is_training=True)

  def train_preprocess_fn(data: tf.data.Dataset) -> tf.data.Dataset:
    """Processes a client's local dataset for training."""
    return data.map(
        train_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(
            _SHUFFLE_BUFFER_SIZE).take(_MAX_ELEMENTS_PER_CLIENT).repeat(
                num_local_epochs).batch(train_batch_size)

  input_spec = train_preprocess_fn(
      raw_client_data.create_tf_dataset_for_client(
          raw_client_data.client_ids[0])).element_spec

  def model_fn() -> tff.learning.models.VariableModel:
    keras_model = mobilenet_v2.create_mobilenet_v2(
        input_shape=(_IMAGE_SIZE, _IMAGE_SIZE, 3),
        num_groups=_NUM_GROUPS,
        num_classes=_NUM_CLASSES)
    return tff.learning.models.from_keras_model(
        keras_model=keras_model,
        input_spec=input_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(_ACCURACY_NAME)],
    )

  if use_synthetic_data:
    train_client_ids = raw_client_data.client_ids
    valid_client_ids = raw_client_data.client_ids
    test_client_ids = raw_client_data.client_ids
  else:
    random_state = np.random.RandomState(seed=constants.SPLIT_CLIENTS_SEED)
    shuffled_client_ids = random_state.permutation(
        raw_client_data.client_ids).tolist()
    valid_client_ids = shuffled_client_ids[:_NUM_VALID_CLIENTS]
    test_client_ids = shuffled_client_ids[_NUM_VALID_CLIENTS:(
        _NUM_VALID_CLIENTS + _NUM_TEST_CLIENTS)]
    train_client_ids = shuffled_client_ids[(_NUM_VALID_CLIENTS +
                                            _NUM_TEST_CLIENTS):]
  new_client_data_fn = functools.partial(
      tff.simulation.datasets.ClientData.from_clients_and_tf_fn,
      serializable_dataset_fn=raw_client_data.serializable_dataset_fn)
  datasets = collections.OrderedDict()
  datasets[constants.TRAIN_CLIENTS_KEY] = new_client_data_fn(train_client_ids)
  datasets[constants.VALID_CLIENTS_KEY] = new_client_data_fn(valid_client_ids)
  datasets[constants.TEST_CLIENTS_KEY] = new_client_data_fn(test_client_ids)

  test_map_fn = lambda element: _map_fn(element, is_training=False)

  @tf.function
  def split_and_add_data_fn(
      raw_data: tf.data.Dataset) -> OrderedDict[str, tf.data.Dataset]:
    """Splits into two unbatched datasets, adds extra examples to the 2nd set.

    Args:
      raw_data: A client's local dataset before any processing is applied.

    Returns:
      An `OrderedDict` of two datasets with `constants.PERSONALIZATION_DATA_KEY`
      and `constants.TEST_DATA_KEY` as keys. The two datasets are formed by
      first splitting the `raw_data` into two unbatched datasets.
    """
    # Splits the `raw_data` into two datasets, and processes them. We do not
    # apply batching here, because
    # `tff.learning.build_personalization_eval_computation` expects *unbatched*
    # client-side datasets. Batching is part of user-supplied personalization
    # function.
    personalization_data, test_data = emnist.split_half(
        raw_data.shuffle(
            _SHUFFLE_BUFFER_SIZE,
            seed=constants.SPLIT_CLIENTS_SEED,
            reshuffle_each_iteration=False,
        )
    )
    personalization_data = personalization_data.map(
        train_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    test_data = test_data.map(
        test_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    # Creates the final data structure.
    final_data = collections.OrderedDict()
    final_data[constants.PERSONALIZATION_DATA_KEY] = personalization_data
    final_data[constants.TEST_DATA_KEY] = test_data
    return final_data

  return (
      model_fn,
      datasets,
      train_preprocess_fn,
      split_and_add_data_fn,
      _ACCURACY_NAME,
  )
