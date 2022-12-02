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
"""Simple networks with two convolutional layers.

A similar structure is used for EMNIST in https://arxiv.org/abs/1602.05629;
the structure is similar to LeNet in
  "Gradient-based learning applied to document recognition".
"""

import functools

import tensorflow as tf

from dp_visual_embeddings.models import keras_utils

_DEFAULT_IMAGE_SHAPE = (28, 28, 1)


def image2embedding(image_shape: tuple[int, int, int] = _DEFAULT_IMAGE_SHAPE,
                    embedding_dim_size: int = 128,
                    always_normalize: bool = False) -> tf.keras.Model:
  """Instantiates the LeNet CNN architecture.

  Args:
    image_shape: Shape of input image.
    embedding_dim_size: The number of dimensions in the output embedding.
    always_normalize: If `True`, will also normalize embedding to unit L2 norm
      during training.

  Returns:
    A keras model with a LeNet architecture.
  """
  if tf.keras.backend.image_data_format() == 'channels_first':
    raise NotImplementedError('Assuming channels last')

  data_format = 'channels_last'
  pool = functools.partial(
      tf.keras.layers.MaxPooling2D,
      pool_size=(2, 2),
      padding='same',
      data_format=data_format)
  conv2d = functools.partial(
      tf.keras.layers.Conv2D,
      kernel_size=5,
      padding='same',
      data_format=data_format,
      activation=tf.nn.relu)
  model = tf.keras.models.Sequential([
      tf.keras.layers.Input(shape=image_shape, name='images'),
      conv2d(filters=32, name='conv1'),
      pool(),
      conv2d(filters=64, name='conv2'),
      pool(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(
          embedding_dim_size, activation=None, use_bias=True,
          name='embeddings'),
      keras_utils.EmbedNormLayer(always_normalize=always_normalize)
  ])

  return model


def lenet_with_head(num_classes: int,
                    image_shape: tuple[int, int, int] = _DEFAULT_IMAGE_SHAPE,
                    embedding_dim_size: int = 128,
                    use_normalize: bool = False) -> keras_utils.EmbeddingModel:
  """LeNet with head for classification and embedding training.

  Args:
    num_classes: Defines the size of the head output.
    image_shape: Shape of input image.
    embedding_dim_size: Defines the size of the embedding vector for an input
      image.
    use_normalize: Whether to normalize the embedding and head weights during
      training.

  Returns:
    A `keras_utils.EmbeddingModel` with attributes (model, global_variables,
      client_variables)
  """
  base_model = image2embedding(
      image_shape=image_shape,
      embedding_dim_size=embedding_dim_size,
      always_normalize=use_normalize)

  return keras_utils.add_embedding_head(
      base_model=base_model,
      num_identities=num_classes,
      use_normalize=use_normalize)
