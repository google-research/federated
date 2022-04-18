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
"""Utilities for dual encoder model."""

from typing import Callable, Optional

import tensorflow as tf

NormalizationFnType = Optional[Callable[[tf.Tensor], tf.Tensor]]
l2_normalize_fn = lambda x: tf.math.l2_normalize(x, axis=-1)


@tf.function
def get_predicted_embeddings(y_pred, y_true, normalization_fn=l2_normalize_fn):
  """Helper for retrieving optionally normalized embeddings from y_pred.

  Args:
    y_pred: dual encoder model output. If the model outputs embeddings, `y_pred`
      is concatenate(context_embedding, full vocab label embeddings) with shape
      [batch_size + label_embedding_vocab_size, final_embedding_dim]. If the
      model outputs similarities, `y_pred` is the similarity matrix with shape
      [batch_size, label_embedding_vocab_size] between context and full vocab
      label embeddings.
    y_true: the true labels with shape [batch_size, 1].
    normalization_fn: The normalization function to be applied to both context
      and label embeddings.

  Returns:
    Optionally normalized context and label embeddings.
  """
  batch_size = tf.shape(y_true)[0]
  context_embedding, label_embedding = y_pred[:batch_size], y_pred[batch_size:]

  # Optionally apply nomalization_fn to both context and label embeddings,
  # computing the cosine similarity rather than the dot product.
  if normalization_fn is not None:
    context_embedding = normalization_fn(context_embedding)
    label_embedding = normalization_fn(label_embedding)

  return context_embedding, label_embedding


@tf.function
def get_embeddings_and_similarities(y_pred,
                                    y_true,
                                    expect_embeddings=True,
                                    normalization_fn=l2_normalize_fn):
  """Retrieving the context and label embeddings and the similarities between them.

  Args:
    y_pred: Dual encoder model output. When expect_embeddings is true, `y_pred`
      is concatenate(context_embedding, full vocab label embeddings) with shape
      [batch_size + label_embedding_vocab_size, final_embedding_dim]. When
      `expect_embeddings` is False, `y_pred` is the similarity matrix with shape
      [batch_size, label_embedding_vocab_size] between context and full vocab
      label embeddings.
    y_true: The true labels with shape [batch_size, 1].
    expect_embeddings: If `expect_embeddings` is True, `y_pred` is the context
      and label embeddings. Otherwise, the y_pred is the batch or global
      similarities.
    normalization_fn: The normalization function to be applied to both context
      and label embeddings.

  Returns:
    The optionally normalized context and label embeddings as well as the
    similarities between them. The context and label embeddings are `None` if
    `expect_embeddings` is False.
  """

  if expect_embeddings:
    context_embedding, label_embedding = (
        get_predicted_embeddings(y_pred, y_true, normalization_fn))

    # similarities[i][j] is the dot product of the ith context embedding and
    # the jth label embedding in a batch.
    similarities = tf.matmul(
        context_embedding, label_embedding, transpose_b=True)
  else:
    context_embedding = label_embedding = None
    similarities = y_pred

  return context_embedding, label_embedding, similarities


class Similarities(tf.keras.layers.Layer):
  """Keras layer for computing similarities over context/label embeddings.

  Takes in context embeddings within a batch and label embeddings to computes a
  similarities matrix where similarities[i][j] is the dot product similarity
  between context embedding i and label embedding j.

  If label embeddings are those within the same batch, this function computes
  the batch similarity.
  If label embeddings are those for the full vocabulary, this function computes
  the global similarity.

  Optionally apply normalization to the embeddings, computing cosine similarity
  instead of dot product.
  """

  def __init__(self,
               normalization_fn: NormalizationFnType = l2_normalize_fn,
               **kwargs):
    super().__init__(**kwargs)
    self.normalization_fn = normalization_fn

  def call(self, inputs):
    if len(inputs) != 2:
      raise ValueError(
          'Exactly two inputs must be provided, context embeddings and label '
          'embeddings, but %d inputs were provided.' % len(inputs))

    context_embedding, label_embedding = inputs

    # Optionally apply normalization to both context and label embeddings,
    # computing the cosine similarity rather than the dot product.
    if self.normalization_fn is not None:
      context_embedding = self.normalization_fn(context_embedding)
      label_embedding = self.normalization_fn(label_embedding)

    # similarities[i][j] is the dot product of the ith context embedding and
    # the jth label embedding in a batch.
    similarities = tf.matmul(
        context_embedding, label_embedding, transpose_b=True)
    return similarities

  def get_config(self):
    config = super().get_config()
    config.update({
        'normalization_fn': self.normalization_fn,
    })
    return config


NORMALIZATION_FN_MAP = {
    'none': None,
    'l2_normalize': l2_normalize_fn,
}
