# Copyright 2021, Google LLC.
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
"""Shared Keras layers for Federated Dual Encoder projects."""

from typing import Callable, Optional

import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
class MaskedAverage(tf.keras.layers.Layer):
  """Keras layer for computing the average of a masked tensor along an axis.

  Computes the mean along an axis, ignoring values masked by a Keras layer
  that produced masked output, e.g.
  tf.keras.layers.Embedding(..., mask_zero=True). Output (and its mask) have
  their rank reduced by one.

  Swallows mask if it currently has rank 2 to avoid producing trivial masks.
  """

  def __init__(self, axis, **kwargs):
    super().__init__(**kwargs)
    self.supports_masking = True
    self.axis = axis

  def call(self, inputs, mask=None):
    if mask is None:
      raise ValueError('Inputs to `MaskedAverage` need to be masked.')

    # Get inputs with masked values replaced by 0's.
    mask = tf.expand_dims(tf.cast(mask, 'float32'), axis=-1)
    masked_input = inputs * mask

    # Perform a mask-weighted sum across the axis, divide by the weights to
    # get the weighted average.
    weighted_summed_input = tf.reduce_sum(masked_input, axis=self.axis)
    total_weights = tf.reduce_sum(mask, axis=self.axis)
    return tf.math.divide_no_nan(weighted_summed_input, total_weights)

  def compute_mask(self, inputs, mask=None):
    if mask is None:
      raise ValueError('Inputs to `MaskedAverage` need to be masked.')

    # Prevent passing on trivial [batch_size] masks.
    if len(tf.keras.backend.int_shape(mask)) == 2:
      return None

    return tf.reduce_any(mask, axis=self.axis)

  def get_config(self):
    config = super().get_config()
    config.update({'axis': self.axis})
    return config


@tf.keras.utils.register_keras_serializable()
class MaskedReshape(tf.keras.layers.Layer):
  """Keras layer for reshaping a tensor along with its mask.

  Similar to `tf.keras.layers.Reshape`, but handles masks produced by a previous
  layer, e.g. tf.keras.layers.Embedding(..., mask_zero=True).
  """

  def __init__(self, new_inputs_shape, new_mask_shape, **kwargs):
    super().__init__(**kwargs)
    self.supports_masking = True
    self.new_inputs_shape = new_inputs_shape
    self.new_mask_shape = new_mask_shape

  def call(self, inputs, mask=None):
    if mask is None:
      raise ValueError('Inputs to `MaskedReshape` need to be masked.')

    return tf.reshape(inputs, self.new_inputs_shape)

  def compute_mask(self, inputs, mask=None):
    if mask is None:
      raise ValueError('Inputs to `MaskedReshape` need to be masked.')

    return tf.reshape(mask, self.new_mask_shape)

  def get_config(self):
    config = super().get_config()
    config.update({
        'new_inputs_shape': self.new_inputs_shape,
        'new_mask_shape': self.new_mask_shape,
    })
    return config


@tf.keras.utils.register_keras_serializable()
class EmbeddingSpreadoutRegularizer(tf.keras.regularizers.Regularizer):
  """Regularizer for ensuring embeddings are spreadout within embedding space.

  This is a variation on approaches for ensuring that embeddings/encodings of
  different items in either an input space or an output space are "spread out".
  This corresponds to randomly selected pairs of embeddings having a low
  dot product or cosine similarity, and can be used to effectively add a form
  of negative sampling to models. The original paper that proposed
  regularization for this purpose is at https://arxiv.org/abs/1708.06320,
  although we do a few things different here: (1) we apply spreadout on input
  embeddings, not the learned encodings of a two-tower model, (2) we apply the
  regularization per batch, instead of per epoch, (3) we leave L2 normalization
  before applying spreadout as optional–since using the dot product here.

  `spreadout_lambda` scales the regularization magnitude. `normalization_fn` is
  the normalization function to be applied to embeddings before applying
  spreadout. If `l2_regularization` is nonzero, standard L2 regularization with
  `l2_regularization` as a scaling constant is applied. Note that
  `l2_regularization` is applied before any normalization, so this has the same
  effect as adding L2 regularization with `tf.keras.regularizers.l2`.

  Usage example:
    embedding_layer = tf.keras.layers.Embedding(
        ...
        embeddings_regularizer=EmbeddingSpreadoutRegularizer(spreadout_lambda))
  """

  def __init__(self,
               spreadout_lambda: float = 0.0,
               normalization_fn: Optional[
                   Callable[[tf.Tensor], tf.Tensor]] = None,
               l2_regularization: float = 0.0):
    self.spreadout_lambda = spreadout_lambda
    self.normalization_fn = normalization_fn
    self.l2_regularization = l2_regularization

  def __call__(self, weights):
    total_regularization = 0.0

    # Apply optional L2 regularization before normalization.
    if self.l2_regularization:
      total_regularization += self.l2_regularization * tf.reduce_sum(
          tf.square(weights))

    if self.normalization_fn is not None:
      weights = self.normalization_fn(weights)

    similarities = tf.matmul(weights, weights, transpose_b=True)
    similarities = tf.linalg.set_diag(
        similarities,
        tf.zeros([tf.shape(weights)[0]], dtype=tf.float32))
    similarities_norm = tf.sqrt(tf.reduce_sum(tf.square(similarities)))

    total_regularization += self.spreadout_lambda * similarities_norm

    return total_regularization

  def get_config(self):
    return {
        'spreadout_lambda': self.spreadout_lambda,
        'normalization_fn': self.normalization_fn,
        'l2_regularization': self.l2_regularization,
    }
