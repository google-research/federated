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
"""Build a model for EMNIST classification."""

import functools
from typing import Optional

import tensorflow as tf


def create_conv_dropout_model(only_digits: bool = True,
                              seed: Optional[int] = None):
  """Convolutional model with droupout for EMNIST experiments.

  Args:
    only_digits: If True, uses a final layer with 10 outputs, for use with the
      digits only EMNIST dataset. If False, uses 62 outputs for the larger
      dataset.
    seed: A random seed governing the model initialization and layer randomness.
      If not `None`, then the global random seed will be set before constructing
      the tensor initializer, in order to guarantee the same model is produced.

  Returns:
    A `tf.keras.Model`.
  """
  data_format = 'channels_last'
  if seed is not None:
    tf.random.set_seed(seed)
  initializer = tf.keras.initializers.GlorotNormal(seed=seed)

  model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(
          32,
          kernel_size=(3, 3),
          activation='relu',
          data_format=data_format,
          input_shape=(28, 28, 1),
          kernel_initializer=initializer),
      tf.keras.layers.Conv2D(
          64,
          kernel_size=(3, 3),
          activation='relu',
          data_format=data_format,
          kernel_initializer=initializer),
      tf.keras.layers.MaxPool2D(pool_size=(2, 2), data_format=data_format),
      tf.keras.layers.Dropout(0.25, seed=seed),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(
          128, activation='relu', kernel_initializer=initializer),
      tf.keras.layers.Dropout(0.5, seed=seed),
      tf.keras.layers.Dense(
          10 if only_digits else 62,
          activation=tf.nn.softmax,
          kernel_initializer=initializer),
  ])

  return model


def create_original_fedavg_cnn_model(only_digits: bool = True,
                                     seed: Optional[int] = None):
  """The CNN model used in https://arxiv.org/abs/1602.05629.

  The number of parameters when `only_digits=True` is (1,663,370), which matches
  what is reported in the paper.

  Args:
    only_digits: If True, uses a final layer with 10 outputs, for use with the
      digits only EMNIST dataset. If False, uses 62 outputs for the larger
      dataset.
    seed: A random seed governing the model initialization and layer randomness.
      If not `None`, then the global random seed will be set before constructing
      the tensor initializer, in order to guarantee the same model is produced.

  Returns:
    A `tf.keras.Model`.
  """
  data_format = 'channels_last'
  if seed is not None:
    tf.random.set_seed(seed)
  initializer = tf.keras.initializers.GlorotNormal(seed=seed)

  max_pool = functools.partial(
      tf.keras.layers.MaxPooling2D,
      pool_size=(2, 2),
      padding='same',
      data_format=data_format)
  conv2d = functools.partial(
      tf.keras.layers.Conv2D,
      kernel_size=5,
      padding='same',
      data_format=data_format,
      activation=tf.nn.relu,
      kernel_initializer=initializer)
  model = tf.keras.models.Sequential([
      conv2d(filters=32, input_shape=(28, 28, 1)),
      max_pool(),
      conv2d(filters=64),
      max_pool(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(
          512, activation=tf.nn.relu, kernel_initializer=initializer),
      tf.keras.layers.Dense(
          10 if only_digits else 62,
          activation=tf.nn.softmax,
          kernel_initializer=initializer),
  ])

  return model


def create_two_hidden_layer_model(only_digits: bool = True,
                                  hidden_units: int = 200,
                                  seed: Optional[int] = None):
  """Create a two hidden-layer fully connected neural network.

  Args:
    only_digits: A boolean that determines whether to only use the digits in
      EMNIST, or the full EMNIST-62 dataset. If True, uses a final layer with 10
      outputs, for use with the digit-only EMNIST dataset. If False, uses 62
      outputs for the larger dataset.
    hidden_units: An integer specifying the number of units in the hidden layer.
      We default to 200 units, which matches that in
      https://arxiv.org/abs/1602.05629.
    seed: A random seed governing the model initialization and layer randomness.
      If not `None`, then the global random seed will be set before constructing
      the tensor initializer, in order to guarantee the same model is produced.

  Returns:
    A `tf.keras.Model`.
  """
  if seed is not None:
    tf.random.set_seed(seed)
  initializer = tf.keras.initializers.GlorotNormal(seed=seed)

  model = tf.keras.models.Sequential([
      tf.keras.layers.Reshape(input_shape=(28, 28, 1), target_shape=(28 * 28,)),
      tf.keras.layers.Dense(
          hidden_units, activation=tf.nn.relu, kernel_initializer=initializer),
      tf.keras.layers.Dense(
          hidden_units, activation=tf.nn.relu, kernel_initializer=initializer),
      tf.keras.layers.Dense(
          10 if only_digits else 62,
          activation=tf.nn.softmax,
          kernel_initializer=initializer),
  ])

  return model


def create_1m_cnn_model(only_digits: bool = True, seed: Optional[int] = None):
  """A CNN model with slightly under 2^20 (roughly 1 million) params.

  A simple CNN model for the EMNIST character recognition task that is very
  similar to the default recommended model from `create_conv_dropout_model`
  but has slightly under 2^20 parameters. This is useful if the downstream task
  involves randomized Hadamard transform, which requires the model weights /
  gradients / deltas concatednated as a single vector to be padded to the
  nearest power-of-2 dimensions.

  This model is used in https://arxiv.org/abs/2102.06387.

  When `only_digits=False`, the returned model has 1,018,174 trainable
  parameters. For `only_digits=True`, the last dense layer is slightly smaller.

  Args:
    only_digits: If True, uses a final layer with 10 outputs, for use with the
      digits only EMNIST dataset. If False, uses 62 outputs for the larger
      dataset.
    seed: A random seed governing the model initialization and layer randomness.
      If not `None`, then the global random seed will be set before constructing
      the tensor initializer, in order to guarantee the same model is produced.

  Returns:
    A `tf.keras.Model`.
  """
  data_format = 'channels_last'
  if seed is not None:
    tf.random.set_seed(seed)
  initializer = tf.keras.initializers.GlorotUniform(seed=seed)

  model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(
          32,
          kernel_size=(3, 3),
          activation='relu',
          data_format=data_format,
          input_shape=(28, 28, 1),
          kernel_initializer=initializer),
      tf.keras.layers.MaxPool2D(pool_size=(2, 2), data_format=data_format),
      tf.keras.layers.Conv2D(
          64,
          kernel_size=(3, 3),
          activation='relu',
          data_format=data_format,
          kernel_initializer=initializer),
      tf.keras.layers.Dropout(0.25),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(
          128, activation='relu', kernel_initializer=initializer),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(
          10 if only_digits else 62,
          activation=tf.nn.softmax,
          kernel_initializer=initializer),
  ])

  return model
