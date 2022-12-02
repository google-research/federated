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
"""Transforms for data to and from the input format for Tensorflow Federated."""

import collections
from collections.abc import Callable
import functools

import tensorflow as tf
import tensorflow_federated as tff


# TFF assumes these are the keys for feature and labels (respectively) in the
# dictionary that comprises the elements of the tf.data.Dataset.
FEATURE_KEY = 'x'
LABELS_KEY = 'y'


def _scale_image_pixels(image: tf.Tensor) -> tf.Tensor:
  """Scales image pixels from [0, 255] to [-1, 255/128-1].

  Args:
    image: A tensor with the input image. Pixel value range [0, 255].

  Returns:
    The float image tensor [-1, 255/128-1]. Use `255/128-1` instead of 1 to
    match the conversion.
  """
  image = tf.cast(image, dtype=tf.float32)
  return image / 128. - 1


def _integer_label_batch_map_fn(batched_element_dict, *, image_shape, image_key,
                                identity_key, squeeze_dim):
  """Converts batched dict of client data to format expected by embedding model.
  """
  images = batched_element_dict[image_key]
  images = _scale_image_pixels(images)
  # Ensure & set the image shape for shape inference for the Dataset element
  # spec.
  batched_shape = (None,) + image_shape
  tf.ensure_shape(images, batched_shape)
  images.set_shape(batched_shape)
  features = collections.OrderedDict(images=images)

  identities = batched_element_dict[identity_key]
  if squeeze_dim > -1:
    identities = tf.squeeze(identities, axis=squeeze_dim)
  identity_names = tf.strings.as_string(identities)
  labels = collections.OrderedDict(
      identity_names=identity_names,
      identity_indices=identities,
  )

  return collections.OrderedDict([
      (FEATURE_KEY, features),
      (LABELS_KEY, labels),
  ])


def _get_gray2color_resize_image_fn(image_key, image_shape):
  """Returns a function to resize image."""

  def resize_image_fn(element_dict):
    image = element_dict[image_key]
    tf.debugging.assert_rank(image, 3)
    resize_image = tf.image.resize(image, image_shape[:2])
    color_image = tf.cond(
        tf.math.logical_and(
            tf.equal(tf.shape(resize_image)[2], 1),
            tf.equal(image_shape[2], 3)),
        lambda: tf.image.grayscale_to_rgb(resize_image), lambda: resize_image)
    element_dict[image_key] = color_image
    return element_dict

  return resize_image_fn


def get_integer_label_preprocess_fn(
    preprocess_spec: tff.simulation.baselines.ClientSpec,
    *,
    image_shape: tuple[int, int, int],
    resize_images: bool = False,
    image_key: str = 'image',
    identity_key: str = 'identity',
) -> Callable[[tf.data.Dataset], tf.data.Dataset]:
  """Returns the function to map the parsed examples to TFF input.

  There are two main functionality of the returned preprocessing function:
  (1) dataset preprocessing including shuffling, batching, repeating;
  (2) map image and identities to TFF simulation format.

  Args:
    preprocess_spec: A `tff.simulation.baselines.ClientSpec` defines dataset
      processing parameters.
    image_shape: Image shape for format checking or resizing.
    resize_images: If True, preprocessing will resize all images to the expected
      height and width, using `tf.image.resize`.
    image_key: The key to get image from a dataset element.
    identity_key: The key to get identity from a dataset element.
  """

  def preprocess_fn(dataset: tf.data.Dataset) -> tf.data.Dataset:
    if dataset.element_spec[identity_key].shape.rank == 1:
      squeeze_dim = 1
    else:
      squeeze_dim = -1  # no squeeze
    batch_map_fn = functools.partial(
        _integer_label_batch_map_fn,
        image_shape=image_shape,
        image_key=image_key,
        identity_key=identity_key,
        squeeze_dim=squeeze_dim)
    if preprocess_spec.shuffle_buffer_size > 1:
      dataset = dataset.shuffle(preprocess_spec.shuffle_buffer_size)
    if preprocess_spec.num_epochs > 1:
      dataset = dataset.repeat(preprocess_spec.num_epochs)
    if preprocess_spec.max_elements is not None:
      dataset = dataset.take(preprocess_spec.max_elements)
    if resize_images:
      resize_image_fn = _get_gray2color_resize_image_fn(image_key, image_shape)
      dataset = dataset.map(
          resize_image_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(preprocess_spec.batch_size)
    return dataset.map(
        batch_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  return preprocess_fn
