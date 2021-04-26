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
                              seed: Optional[int] = 0):
  """Convolutional model with droupout for EMNIST experiments.

  Args:
    only_digits: If True, uses a final layer with 10 outputs, for use with the
      digits only EMNIST dataset. If False, uses 62 outputs for the larger
      dataset.
    seed: A random seed governing the model initialization and layer randomness.
      If set to `None`, No random seed is used.

  Returns:
    A `tf.keras.Model`.
  """
  data_format = 'channels_last'
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
                                     seed: Optional[int] = 0):
  """The CNN model used in https://arxiv.org/abs/1602.05629.

  The number of parameters when `only_digits=True` is (1,663,370), which matches
  what is reported in the paper.

  Args:
    only_digits: If True, uses a final layer with 10 outputs, for use with the
      digits only EMNIST dataset. If False, uses 62 outputs for the larger
      dataset.
    seed: A random seed governing the model initialization and layer randomness.

  Returns:
    A `tf.keras.Model`.
  """
  data_format = 'channels_last'
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
                                  seed: Optional[int] = 0):
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

  Returns:
    A `tf.keras.Model`.
  """
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
