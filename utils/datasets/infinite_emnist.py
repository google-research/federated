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
"""Library for augmenting EMNIST with pseudo-clients."""

import numpy as np
import tensorflow as tf
import tensorflow_addons.image as tfa_image
import tensorflow_federated as tff

_CLIENT_ID_SEPARATOR = ':'


def _compile_transform(angle: float = 0.0,
                       shear: float = 0.0,
                       scale_x: float = 1.0,
                       scale_y: float = 1.0,
                       translation_x: float = 0,
                       translation_y: float = 0):
  """Compiles affine transform parameters into single projective transform.

  The transformations are performed in the following order: rotation, shearing,
  scaling, and translation.

  Args:
    angle: The angle of counter-clockwise rotation, in degrees.
    shear: The amount of shear. Precisely, shear*x is added to the y coordinate
      after centering.
    scale_x: The amount to scale in the x-axis.
    scale_y: The amount to scale in the y-axis.
    translation_x: The number of pixels to translate in the x-axis.
    translation_y: The number of pixels to translate in the y-axis.

  Returns:
    A length 8 tensor representing the composed transform.
  """
  angle = angle * np.pi / 180
  size = 28

  # angles_to_projective_transforms performs rotations around center of image.
  rotation = tfa_image.transform_ops.angles_to_projective_transforms(
      angle, size, size)

  # shearing and scaling require centering and decentering.
  half = (size - 1) / 2.0
  center = tfa_image.translate_ops.translations_to_projective_transforms(
      [-half, -half])
  shear = [1., 0., 0., -shear, 1., 0., 0., 0.]
  scaling = [1. / scale_x, 0., 0., 0., 1. / scale_y, 0., 0., 0.]
  decenter = tfa_image.translate_ops.translations_to_projective_transforms(
      [half, half])

  translation = tfa_image.translate_ops.translations_to_projective_transforms(
      [translation_x, translation_y])
  return tfa_image.transform_ops.compose_transforms(
      transforms=[rotation, center, shear, scaling, decenter, translation])


def _make_transform_fn(client_id: str):
  """Generates a pseudorandom affine transform from the raw_client_id and index.

  If the index is 0 no transform is applied.

  Args:
    client_id: The client_id.

  Returns:
    The data transform fn.
  """
  split_client_id = tf.strings.split(client_id, _CLIENT_ID_SEPARATOR)
  index = tf.cast(tf.strings.to_number(split_client_id[1]), tf.int32)
  raw_client_id = split_client_id[0]

  client_hash = tf.strings.to_hash_bucket_fast(raw_client_id, 2**32)
  index_seed = tf.cast(index, tf.int64)

  def random_scale(min_val, seed):
    b = tf.math.log(min_val)
    log_val = tf.random.stateless_uniform((), (seed, index_seed), b, -b)
    return tf.math.exp(log_val)

  transform = _compile_transform(
      angle=tf.random.stateless_uniform((), (client_hash, index_seed), -20, 20),
      shear=tf.random.stateless_uniform((), (client_hash + 1, index_seed), -0.2,
                                        0.2),
      scale_x=random_scale(0.8, client_hash + 2),
      scale_y=random_scale(0.8, client_hash + 3),
      translation_x=tf.random.stateless_uniform(
          (), (client_hash + 4, index_seed), -5, 5),
      translation_y=tf.random.stateless_uniform(
          (), (client_hash + 5, index_seed), -5, 5))

  @tf.function
  def _transform_fn(data):
    """Applies a random transform to the pixels."""
    # EMNIST background is 1.0 but tfa_image.transform assumes 0.0, so invert.
    pixels = 1.0 - data['pixels']

    pixels = tfa_image.transform(pixels, transform, 'BILINEAR')

    # num_bits=9 actually yields 256 unique values.
    pixels = tf.quantization.quantize_and_dequantize(
        pixels, 0.0, 1.0, num_bits=9, range_given=True)

    pixels = 1.0 - pixels

    result = data.copy()

    # The first pseudoclient applies the identity transformation.
    result['pixels'] = tf.cond(
        tf.equal(index, 0), lambda: data['pixels'], lambda: pixels)

    return result

  return _transform_fn


def get_infinite(emnist_client_data: tff.simulation.datasets.ClientData,
                 client_expansion_factor: int):
  """Converts a Federated EMNIST dataset into an Infinite Federated EMNIST set.

  Infinite Federated EMNIST expands each writer from the EMNIST dataset into
  some number of pseudo-clients each of whose characters are the same but apply
  a fixed random affine transformation to the original user's characters. The
  distribution over affine transformation is approximately equivalent to the one
  described at https://www.cs.toronto.edu/~tijmen/affNIST/. It applies the
  following transformations in this order:

    1. A random rotation chosen uniformly between -20 and 20 degrees.
    2. A random shearing adding between -0.2 to 0.2 of the x coordinate to the
       y coordinate (after centering).
    3. A random scaling between 0.8 and 1.25 (sampled log uniformly).
    4. A random translation between -5 and 5 pixels in both the x and y axes.

  The first pseudo-client (with index 0) applies the identity transformation, so
  the original user is in the dataset in addition to (client_expansion_factor -
  1) transformed users.

  Args:
    emnist_client_data: The `tff.simulation.datasets.ClientData` to convert.
    client_expansion_factor: How many pseudo-clients to generate for each real
      client. Each pseudo-client is formed by applying a given random affine
      transformation to the characters written by a given real user. The first
      pseudo-client for a given user applies the identity transformation, so the
      original users are always included.

  Returns:
    A `tff.simulation.datasets.TransformingClientData` that expands each client
      into `client_expansion_factor` pseudo-clients.
  """
  if client_expansion_factor <= 1:
    raise ValueError('`client_expansion_factor` must be greater than 1.')

  for client_id in emnist_client_data.client_ids:
    assert _CLIENT_ID_SEPARATOR not in client_id

  def expand_client_id(client_id):
    return [
        client_id + _CLIENT_ID_SEPARATOR + str(i)
        for i in range(client_expansion_factor)
    ]

  def reduce_client_id(client_id):
    return tf.strings.split(client_id, sep=_CLIENT_ID_SEPARATOR)[0]

  return tff.simulation.datasets.TransformingClientData(emnist_client_data,
                                                        _make_transform_fn,
                                                        expand_client_id,
                                                        reduce_client_id)
