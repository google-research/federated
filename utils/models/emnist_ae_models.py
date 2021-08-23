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
"""Build a model for EMNIST autoencoder classification."""

import functools
from typing import Optional

import tensorflow as tf


def create_autoencoder_model(seed: Optional[int] = None):
  """Bottleneck autoencoder for EMNIST autoencoder experiments.

  Args:
    seed: A random seed governing the model initialization and layer randomness.
      If not `None`, then the global random seed will be set before constructing
      the tensor initializer, in order to guarantee the same model is produced.

  Returns:
    A `tf.keras.Model`.
  """
  if seed is not None:
    tf.random.set_seed(seed)
  initializer = tf.keras.initializers.GlorotNormal(seed=seed)
  dense_layer = functools.partial(
      tf.keras.layers.Dense, kernel_initializer=initializer)

  model = tf.keras.models.Sequential([
      dense_layer(1000, activation='sigmoid', input_shape=(784,)),
      dense_layer(500, activation='sigmoid'),
      dense_layer(250, activation='sigmoid'),
      dense_layer(30, activation='linear'),
      dense_layer(250, activation='sigmoid'),
      dense_layer(500, activation='sigmoid'),
      dense_layer(1000, activation='sigmoid'),
      dense_layer(784, activation='sigmoid'),
  ])

  return model
