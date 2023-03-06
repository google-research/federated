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
"""Preprocessing library for EMNIST baseline tasks."""
from collections.abc import Callable
from typing import Optional

import tensorflow as tf
import tensorflow_federated as tff

MAX_CLIENT_DATASET_SIZE = 418


def _reshape_images(element):
  return tf.expand_dims(element['pixels'], axis=-1), element['label']


def _reshape_images_and_shift_label(element):
  features = tf.expand_dims(element['pixels'], axis=-1)
  labels = tf.where(
      element['label'] < 36, element['label'], element['label'] - 26
  )
  return features, labels


def create_preprocess_fn(
    preprocess_spec: tff.simulation.baselines.ClientSpec,
    num_parallel_calls: tf.Tensor = tf.data.experimental.AUTOTUNE,
    debug_seed: Optional[int] = None,
    label_distribution_shift: bool = False,
) -> Callable[[tf.data.Dataset], tf.data.Dataset]:
  """Creates a preprocessing function for EMNIST client datasets.

  The preprocessing shuffles, repeats, batches, and then reshapes, using
  the `shuffle`, `repeat`, `take`, `batch`, and `map` attributes of a
  `tf.data.Dataset`, in that order.

  Args:
    preprocess_spec: A `tff.simulation.baselines.ClientSpec` containing
      information on how to preprocess clients.
    num_parallel_calls: An integer representing the number of parallel calls
      used when performing `tf.data.Dataset.map`.
    debug_seed: An optional integer seed for deterministic shuffling and
      mapping. Intended for unittesting.
    label_distribution_shift: Label distribution shift by shifting the original
      integer label 36-61 (i.e., characters a-z) to 10-35 (i.e., characters A-Z
      ).

  Returns:
    A callable taking as input a `tf.data.Dataset`, and returning a
    `tf.data.Dataset` formed by preprocessing according to the input arguments.
  """
  shuffle_buffer_size = preprocess_spec.shuffle_buffer_size
  if shuffle_buffer_size is None:
    shuffle_buffer_size = MAX_CLIENT_DATASET_SIZE

  if label_distribution_shift:
    batch_map_fn = _reshape_images_and_shift_label
  else:
    batch_map_fn = _reshape_images

  def preprocess_fn(dataset: tf.data.Dataset) -> tf.data.Dataset:
    if shuffle_buffer_size > 1:
      dataset = dataset.shuffle(shuffle_buffer_size, seed=debug_seed)
    if preprocess_spec.num_epochs > 1:
      dataset = dataset.repeat(preprocess_spec.num_epochs)
    if preprocess_spec.max_elements is not None:
      dataset = dataset.take(preprocess_spec.max_elements)
    dataset = dataset.batch(preprocess_spec.batch_size, drop_remainder=False)
    return dataset.map(
        batch_map_fn,
        num_parallel_calls=num_parallel_calls,
        deterministic=debug_seed is not None,
    )

  return preprocess_fn
