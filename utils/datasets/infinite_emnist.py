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

import hashlib
import math

import numpy as np
import tensorflow as tf
import tensorflow_addons.image as tfa_image
import tensorflow_federated as tff


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
  angle = math.radians(angle)
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


def _make_transform_fn(raw_client_id: str, index: int):
  """Generates a pseudorandom affine transform based on the client_id and index.

  If the index is 0, `None` is returned so no transform is applied by the
  transforming_client_data.

  Args:
    raw_client_id: The raw client_id.
    index: The index of the pseudo-client.

  Returns:
    The transformed data.
  """
  if index == 0:
    return None

  stable_hash_of_client_id = int.from_bytes(
      hashlib.md5(raw_client_id.encode()).digest(), byteorder='big')
  np.random.seed((stable_hash_of_client_id + index) % (2**32))

  def random_scale(min_val):
    b = math.log(min_val)
    return math.exp(np.random.uniform(b, -b))

  transform = _compile_transform(
      angle=np.random.uniform(-20, 20),
      shear=np.random.uniform(-0.2, 0.2),
      scale_x=random_scale(0.8),
      scale_y=random_scale(0.8),
      translation_x=np.random.uniform(-5, 5),
      translation_y=np.random.uniform(-5, 5))

  def _transform_fn(data):
    """Applies a random transform to the pixels."""
    # EMNIST background is 1.0 but tfa_image.transform assumes 0.0, so invert.
    pixels = 1.0 - data['pixels']

    pixels = tfa_image.transform(pixels, transform, 'BILINEAR')

    # num_bits=9 actually yields 256 unique values.
    pixels = tf.quantization.quantize_and_dequantize(
        pixels, 0.0, 1.0, num_bits=9, range_given=True)

    data['pixels'] = 1.0 - pixels
    return data

  return _transform_fn


def get_infinite(emnist_client_data: tff.simulation.datasets.ClientData,
                 num_pseudo_clients: int):
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

  Args:
    emnist_client_data: The `tff.simulation.datasets.ClientData` to convert.
    num_pseudo_clients: How many pseudo-clients to generate for each real
      client. Each pseudo-client is formed by applying a given random affine
      transformation to the characters written by a given real user. The first
      pseudo-client for a given user applies the identity transformation, so the
      original users are always included.

  Returns:
    An expanded `tff.simulation.datasets.ClientData`.
  """
  num_client_ids = len(emnist_client_data.client_ids)

  return tff.simulation.datasets.TransformingClientData(
      raw_client_data=emnist_client_data,
      make_transform_fn=_make_transform_fn,
      num_transformed_clients=(num_client_ids * num_pseudo_clients))
