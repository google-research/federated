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
"""Uncompiled keras models for different datasets.

For FashionMNIST, the options are:
  * MLP_MODEL: A multi-layer perceptron with 2 hidden layers:
    784 -> H -> H -> 10 (predictions), where H is the number of hidden units
  * CONV_MODEL: A LeNet-style convolutional network with 2 convolutional
    layers and two dense layers.

NOTES:
  * We only assume classification models *without* a last softmax layer
    (see auditing_trainer.py).
"""

import tensorflow as tf

from lidp_auditing import constants


def get_model_for_dataset(
    dataset_name: str, model_type: str, hidden_units: int = 256
) -> tf.keras.Model:
  """Create an uncompiled keras model for the dataset.

  Args:
    dataset_name: name of the dataset.
    model_type: type of the model ('mlp', 'conv', etc.).
    hidden_units: the number of hidden units in the model (if applicable).

  Returns:
    An uncompiled `tf.keras.Model`.

  Raises:
    RuntimeError: if the dataset is not known.
  """
  if dataset_name == constants.FASHION_MNIST_DATASET:
    return _get_fashion_mnist_models(model_type, hidden_units=hidden_units)
  elif dataset_name == constants.PURCHASE_DATASET:
    return _get_purchase_models(model_type, hidden_units=hidden_units)
  else:
    raise RuntimeError('Unknown dataset %s' % (dataset_name,))


def _get_purchase_models(model_type, hidden_units=256):
  """Create models for Purchase."""
  del model_type  # only one model type is supported
  return tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(600, 1)),
      tf.keras.layers.Dense(hidden_units, activation=tf.nn.relu),
      tf.keras.layers.Dense(
          hidden_units,
          activation=tf.nn.relu,
      ),
      tf.keras.layers.Dense(100),
  ])


def _get_fashion_mnist_models(model_type, hidden_units=256):
  """Create models for FashionMNIST."""
  assert model_type in [
      constants.LINEAR_MODEL,
      constants.MLP_MODEL,
  ]
  if model_type == constants.LINEAR_MODEL:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(10),
    ])
  elif model_type == constants.MLP_MODEL:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
        tf.keras.layers.Dense(hidden_units, activation=tf.nn.relu),
        tf.keras.layers.Dense(
            hidden_units,
            activation=tf.nn.relu,
        ),
        tf.keras.layers.Dense(10),
    ])
  else:
    raise RuntimeError('Unknown model type %s' % (model_type,))

  return model
