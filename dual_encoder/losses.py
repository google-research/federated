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
"""Loss functions for dual encoder models.

Compatible with both server-side Keras and TFF.
"""

from typing import Callable

import tensorflow as tf

from dual_encoder import model_utils as utils


class BatchSoftmax(tf.keras.losses.Loss):
  """Compute batch softmax over batch similarities between output embeddings.

  Takes output context and label embeddings of a dual encoder model (`y_pred`)
  and weights for each example in the batch (`y_true`), computes similarities
  via (optionally normalized) dot product between each pair of context and label
  embeddings (optionally normalized with `normalization_fn`) within the batch.
  Then computes the batch softmax for each example, where the "label" for each
  context is the diagonal element in the similarities matrix.

  If `expect_embeddings` is False, `y_pred` is expected to be a pre-calculated
  similarities matrix, and normalization is ignored. This is useful for reducing
  repeated computation of the similarities matrix in loss functions and metrics.
  Note that if `expect_embeddings` if False, `spreadout_context_lambda` and
  `spreadout_label_lambda` must be 0, since these depend on the context and
  label embeddings.

  Optionally allows the user to apply spreadout regularization with a given
  scaling lambda. Context spreadout can be applied to encourage context
  embeddings within a batch to be perpendicular. Label spreadout can be applied
  to do the same for label embeddings. Cross spreadout encourages off-diagonal
  context and label embeddings to be perpendicular, effectively providing a
  knob to modulate the "negative part" of batch softmax. See
  https://arxiv.org/abs/1708.06320 for more information on spreadout
  regularization, although in this work and follow-ups spreadout is only applied
  to one tower or both towers individually (unlike cross spreadout), and is not
  applied per batch with the core loss function during training. Note that the
  values of `spreadout_context_lambda`, `spreadout_label_lambda` and
  `spreadout_cross_lambda` are expect to be non-negative.

  Using `y_true` as weights allows us to conform to the typical Keras interface
  of providing labels along with input data, enabling easier TFF support.

  If `use_global_similarity` is True, `y_true` is expected to be a
  pre-calculated global similarities matrix or a (context embeddings, full label
  embeddings) tuple which is used to calculate global similarities. It is
  required to be False when calling `BatchSoftmax`.

  This class is called when using the batch softmax as the loss function and the
  model outputs batch similarities or embeddings to compute the batch
  similarities. It should produce exactly the same results as
  `BatchSoftmaxWithGlobalSimilarity` with global similarity.
  """

  def __init__(
      self,
      normalization_fn: utils.NormalizationFnType = utils.l2_normalize_fn,
      expect_embeddings: bool = True,
      spreadout_context_lambda: float = 0.0,
      spreadout_label_lambda: float = 0.0,
      spreadout_cross_lambda: float = 0.0,
      label_indices_fn: Callable[[tf.Tensor], tf.Tensor] = tf.range,
      use_global_similarity: bool = False,
      **kwargs):
    if not expect_embeddings and spreadout_context_lambda:
      raise ValueError(
          '`spreadout_context_lambda` must be 0 if `expect_embeddings` is'
          'False, but got %f.' % spreadout_context_lambda)

    if not expect_embeddings and spreadout_label_lambda:
      raise ValueError(
          '`spreadout_label_lambda` must be 0 if `expect_embeddings` is False, '
          'but got %f.' % spreadout_label_lambda)

    if use_global_similarity:
      raise ValueError(
          '`BatchSoftmax` does not support global similarity, as indicated by'
          f'use_global_similarity = {use_global_similarity}. Consider using'
          '`BatchSoftmaxWithGlobalSimilarity`.')

    self.normalization_fn = normalization_fn
    self.expect_embeddings = expect_embeddings
    self.spreadout_context_lambda = spreadout_context_lambda
    self.spreadout_label_lambda = spreadout_label_lambda
    self.spreadout_cross_lambda = spreadout_cross_lambda
    self.label_indices_fn = label_indices_fn
    super().__init__(**kwargs)

  @tf.function
  def call(
      self,
      y_true: tf.Tensor,
      y_pred: tf.Tensor):
    """Compute softmax loss with batch labels as negatives.

    Args:
      y_true: The weights for each example in the batch with shape [batch_size]
        or [batch_size, 1].
      y_pred: When expect_embedding is False, `y_pred` is the pre-calculated
        batch similarities matrix with shape [batch_size, batch_size]; when
        `expected_embedding` is True, `y_pred` is the output batch context and
        label embedddings of dual encoder model [context_embedding,
        label_embedding].

    Returns:
      The softmax loss with batch labels as negatives. Optionally add
      spreadout regularizations with given scaling lambdas.
    """

    context_embedding, label_embedding, similarities = (
        utils.get_embeddings_and_similarities(
            y_pred, y_true, self.expect_embeddings, self.normalization_fn))

    # Use the diagonal elements of similarities as labels for each row.
    curr_batch_size = tf.shape(similarities)[0]
    label_indices = self.label_indices_fn(curr_batch_size)
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=label_indices, logits=similarities)

    # Perform a weighted sum, then divide by the sum of weights to get the
    # weighted average of the per-example losses. Flatten weights first in case
    # they have shape (batch_size, 1).
    y_true = tf.cast(tf.reshape(y_true, [-1]), losses.dtype)
    loss = tf.math.divide_no_nan(
        tf.reduce_sum(losses * y_true), tf.reduce_sum(y_true))

    # Apply spreadout if needed.
    loss = _update_loss_with_spreadout_loss(
        loss=loss,
        context_embedding=context_embedding,
        label_embedding=label_embedding,
        similarities=similarities,
        spreadout_context_lambda=self.spreadout_context_lambda,
        spreadout_label_lambda=self.spreadout_label_lambda,
        spreadout_cross_lambda=self.spreadout_cross_lambda,
        label_indices=label_indices)

    return loss

  def get_config(self):
    config = {
        'normalization_fn': self.normalization_fn,
        'expect_embeddings': self.expect_embeddings,
        'spreadout_context_lambda': self.spreadout_context_lambda,
        'spreadout_label_lambda': self.spreadout_label_lambda,
        'spreadout_cross_lambda': self.spreadout_cross_lambda,
        'label_indices_fn': self.label_indices_fn,
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


# TODO(b/179517823): Merge the "BatchSoftmax*" implementations by supporting
# use_global_similarity as an arg,
class BatchSoftmaxWithGlobalSimilarity(tf.keras.losses.Loss):
  """Compute batch softmax over global similarities between output embeddings.

  Takes output context and full vocab label embedddings of a dual encoder model
  (`y_pred`) and the label of each example in the batch (`y_true`), computes
  global similarities via (optionally normalized) dot product between each pair
  of context and label embeddings (optionally normalized with
  `normalization_fn`) . Then computes batch softmax for each example.

  If `expect_embeddings` is False, y_pred is expected to be a pre-calculated
  global similarities matrix, and normalization is ignored. Note that if
  `expect_embeddings` is False, `spreadout_context_lambda` and
  `spreadout_label_lambda` must be 0, since these depend on the context and
  label embeddings.

  Optionally allows the user to apply spreadout regularization with a given
  scaling lambda. When applying the spreadout regularization, getting the batch
  label embeddings first so that all of the spreadout calculations happen over
  the batch. Refer to the description of `class BatchSoftmax` for more
  details about the spreadout regularization. Note that the
  values of `spreadout_context_lambda`, `spreadout_label_lambda` and
  `spreadout_cross_lambda` are expect to be non-negative.

  If 'use_global_similarity' is True, y_true is expected to be a pre-calculated
  global similarities matrix or a (context embeddings, full label embeddings)
  tuple which is used to calculate global similarities. It is required to be
  True when calling `BatchSoftmaxWithGlobalSimilarity`.

  This class is called when using the batch softmax as the loss function but the
  model outputs global similarities or embeddings to compute the global
  similarities. It produces exactly the same results as `BatchSoftmax` without
  global similarities.
  """

  def __init__(
      self,
      normalization_fn: utils.NormalizationFnType = utils.l2_normalize_fn,
      expect_embeddings: bool = True,
      spreadout_context_lambda: float = 0.0,
      spreadout_label_lambda: float = 0.0,
      spreadout_cross_lambda: float = 0.0,
      use_global_similarity: bool = True,
      **kwargs):
    if not expect_embeddings and spreadout_context_lambda:
      raise ValueError(
          '`spreadout_context_lambda` must be 0 if `expect_embeddings` is'
          'False, but got %f.' % spreadout_context_lambda)

    if not expect_embeddings and spreadout_label_lambda:
      raise ValueError(
          '`spreadout_label_lambda` must be 0 if `expect_embeddings` is False, '
          'but got %f.' % spreadout_label_lambda)

    if not use_global_similarity:
      raise ValueError(
          '`BatchSoftmaxWithGlobalSimilarity` does not support batch'
          'similarity, as indicated by use_global_similarity ='
          f'{use_global_similarity}. Consider using BatchSoftmax instead.')

    self.normalization_fn = normalization_fn
    self.expect_embeddings = expect_embeddings
    self.spreadout_context_lambda = spreadout_context_lambda
    self.spreadout_label_lambda = spreadout_label_lambda
    self.spreadout_cross_lambda = spreadout_cross_lambda
    super().__init__(**kwargs)

  @tf.function
  def call(
      self,
      y_true: tf.Tensor,
      y_pred: tf.Tensor):
    """Compute softmax loss with in batch labels as negatives.

    Args:
      y_true: The true labels with shape [batch_size, 1] or [batch_size].
      y_pred: When expect_embedding is False, `y_pred` is a pre-calculated
        global similarities matrix with shape [batch_size,
        label_embedding_vocab_size]; when `expected_embedding` is True, `y_pred`
        is the concatenated output context and full vocab label embedddings of a
        dual encoder model, with shape
        [batch_size + label_embedding_vocab_size, embedding_dim].

    Returns:
      The softmax loss with in batch labels as negatives. Optionally add
      spreadout regularizations with given scaling lambdas.
    """

    context_embedding, label_embedding, similarities = (
        utils.get_embeddings_and_similarities(
            y_pred, y_true, self.expect_embeddings, self.normalization_fn))

    # Extract the batch similarities given the batch label (y_true).
    batch_similarities = tf.gather(
        similarities, tf.reshape(y_true, [-1]), axis=1)
    # The batch similarities should be in shape [batch_size, batch_size].
    # Use the diagonal elements of batch similarities as labels for each row.
    curr_batch_size = tf.shape(batch_similarities)[0]
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.range(curr_batch_size), logits=batch_similarities)
    loss = tf.reduce_mean(losses)

    # Apply spreadout if needed.
    if self.expect_embeddings:
      # Getting the batch label embeddings so all of the spreadout calculations
      # happen over the batch.
      label_embedding = tf.gather(
          label_embedding, tf.transpose(y_true)[0], axis=0)
    loss = _update_loss_with_spreadout_loss(
        loss=loss,
        context_embedding=context_embedding,
        label_embedding=label_embedding,
        similarities=batch_similarities,
        spreadout_context_lambda=self.spreadout_context_lambda,
        spreadout_label_lambda=self.spreadout_label_lambda,
        spreadout_cross_lambda=self.spreadout_cross_lambda)

    return loss

  def get_config(self):
    config = {
        'normalization_fn': self.normalization_fn,
        'expect_embeddings': self.expect_embeddings,
        'spreadout_context_lambda': self.spreadout_context_lambda,
        'spreadout_label_lambda': self.spreadout_label_lambda,
        'spreadout_cross_lambda': self.spreadout_cross_lambda,
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


class GlobalSoftmax(tf.keras.losses.Loss):
  """Compute global softmax over global similarities between output embeddings.

  Takes output context and full vocab label embedddings of a dual encoder model
  (`y_pred`) and the label of each example in the batch (`y_true`), computes
  global similarities via (optionally normalized) dot product between each pair
  of context and label embeddings (optionally normalized with
  `normalization_fn`) . Then computes global softmax for each example. When
  using global softmax, all the non-label items are used as negatives when
  computing the softmax loss with the input batch.

  If `expect_embeddings` is False, `y_pred` is expected to be a pre-calculated
  global similarities matrix, and normalization is ignored. Note that if
  `expect_embeddings` is False, `spreadout_context_lambda` and
  `spreadout_label_lambda` must be 0, since these depend on the context and
  label embeddings.

  Optionally allows the user to apply spreadout regularization with a given
  scaling lambda. Note that `spreadout_label_lambda` includes full vocab labels,
  but `spreadout_cross_lambda` includes only batch labels. Refer to description
  of `class BatchSoftmax` for more details about the spreadout regularization.
  Note that the values of `spreadout_context_lambda`, `spreadout_label_lambda`
  and `spreadout_cross_lambda` are expect to be non-negative.

  If 'use_global_similarity' is True, `y_true` is expected to be a
  pre-calculated global similarities matrix or a (context embeddings, full label
  embeddings) tuple which is used to calculate global similarities. It is
  required to be True when calling `GlobalSoftmax`.

  This class is called when using the global softmax as the loss function. The
  model should output global similarities or embeddings to compute the global
  similarities. The global softmax loss uses the full vocab labels as negatives.
  """

  def __init__(
      self,
      normalization_fn: utils.NormalizationFnType = utils.l2_normalize_fn,
      expect_embeddings: bool = True,
      spreadout_context_lambda: float = 0.0,
      spreadout_label_lambda: float = 0.0,
      spreadout_cross_lambda: float = 0.0,
      use_global_similarity: bool = True,
      **kwargs):
    if not expect_embeddings and spreadout_context_lambda:
      raise ValueError(
          '`spreadout_context_lambda` must be 0 if `expect_embeddings` is'
          'False, but got %f.' % spreadout_context_lambda)

    if not expect_embeddings and spreadout_label_lambda:
      raise ValueError(
          '`spreadout_label_lambda` must be 0 if `expect_embeddings` is False, '
          'but got %f.' % spreadout_label_lambda)

    if not use_global_similarity:
      raise ValueError(
          '`GlobalSoftmax` does not support batch similarity, as indicated by'
          f'`use_global_similarity` = {use_global_similarity}. Use global'
          'similarity for calculating global softmax.')

    self.normalization_fn = normalization_fn
    self.expect_embeddings = expect_embeddings
    self.spreadout_context_lambda = spreadout_context_lambda
    self.spreadout_label_lambda = spreadout_label_lambda
    self.spreadout_cross_lambda = spreadout_cross_lambda
    super().__init__(**kwargs)

  @tf.function
  def call(
      self,
      y_true: tf.Tensor,
      y_pred: tf.Tensor):
    """Compute softmax loss with full vocab labels as negatives.

    Args:
      y_true: The true labels with shape [batch_size, 1].
      y_pred: When `expect_embedding` is False, `y_pred` is a pre-calculated
        global similarities matrix with shape [batch_size,
        label_embedding_vocab_size]; when `expect_embedding` is True, `y_pred`
        is the output context and full vocab label embedddings of a dual encoder
        model [context_embedding, full_vocab_label_embedding].

    Returns:
      The softmax loss with full vocab labels as negatives. Optionally add
      spreadout regularizations with given scaling lambdas.
    """

    context_embedding, label_embedding, similarities = (
        utils.get_embeddings_and_similarities(
            y_pred, y_true, self.expect_embeddings, self.normalization_fn))

    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=tf.transpose(y_true)[0], logits=similarities)
    loss = tf.reduce_mean(losses)

    # Apply spreadout if needed.
    loss = _update_loss_with_spreadout_loss(
        loss=loss,
        context_embedding=context_embedding,
        label_embedding=label_embedding,
        similarities=similarities,
        spreadout_context_lambda=self.spreadout_context_lambda,
        spreadout_label_lambda=self.spreadout_label_lambda,
        spreadout_cross_lambda=self.spreadout_cross_lambda,
        label_indices=tf.transpose(y_true)[0])

    return loss

  def get_config(self):
    config = {
        'normalization_fn': self.normalization_fn,
        'expect_embeddings': self.expect_embeddings,
        'spreadout_context_lambda': self.spreadout_context_lambda,
        'spreadout_label_lambda': self.spreadout_label_lambda,
        'spreadout_cross_lambda': self.spreadout_cross_lambda,
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


class Hinge(tf.keras.losses.Loss):
  """Compute hinge loss over positive samples.

  Optionally applies `normalization_fn` to context and label embeddings.

  If `expect_embeddings` is False, `y_pred` is expected to be a pre-calculated
  global similarities matrix, and normalization is ignored. Note that if
  `expect_embeddings` is False, `spreadout_context_lambda` and
  `spreadout_label_lambda` must be 0, since these depend on the context and
  label embeddings.

  Optionally allows the user to apply spreadout regularization with a given
  scaling lambda. Note that `spreadout_label_lambda` includes full vocab labels,
  but `spreadout_cross_lambda` includes only batch labels. Refer to description
  of `class BatchSoftmax` for more details about the spreadout regularization.
  Note that the values of `spreadout_context_lambda`, `spreadout_label_lambda`
  and `spreadout_cross_lambda` are expect to be non-negative.

  If `use_global_similarity` is True, `y_true` is expected to be a
  pre-calculated global similarities matrix or a (context embeddings, full label
  embeddings) tuple which is used to calculate global similarities. Otherwise,
  `y_true` is either a pre-calculated batch similarities matrix or a (context
  embeddings, batch label embeddings) tuple which is used to calculate batch
  similarities.
  """

  def __init__(
      self,
      normalization_fn: utils.NormalizationFnType = utils.l2_normalize_fn,
      expect_embeddings: bool = True,
      spreadout_context_lambda: float = 0.0,
      spreadout_label_lambda: float = 0.0,
      spreadout_cross_lambda: float = 0.0,
      use_global_similarity: bool = False,
      **kwargs):
    if not expect_embeddings and spreadout_context_lambda:
      raise ValueError(
          '`spreadout_context_lambda` must be 0 if `expect_embeddings` is'
          ' False, but got %f.' % spreadout_context_lambda)

    if not expect_embeddings and spreadout_label_lambda:
      raise ValueError(
          '`spreadout_label_lambda` must be 0 if `expect_embeddings` is False, '
          'but got %f.' % spreadout_label_lambda)

    self.normalization_fn = normalization_fn
    self.expect_embeddings = expect_embeddings
    self.spreadout_context_lambda = spreadout_context_lambda
    self.spreadout_label_lambda = spreadout_label_lambda
    self.spreadout_cross_lambda = spreadout_cross_lambda
    self.use_global_similarity = use_global_similarity
    super().__init__(**kwargs)

  @tf.function
  def call(
      self,
      y_true: tf.Tensor,
      y_pred: tf.Tensor):

    context_embedding, label_embedding, similarities = (
        utils.get_embeddings_and_similarities(
            y_pred, y_true, self.expect_embeddings, self.normalization_fn))

    if self.use_global_similarity:
      # Extract the batch similarities given the batch label (y_true).
      similarities = tf.gather(
          similarities, tf.transpose(y_true)[0], axis=1)

    # Use the diagonal elements of similarities to compute the Hinge Loss for
    # positive examples.
    positive_similarities = tf.linalg.diag_part(similarities)
    loss = tf.math.reduce_mean(
        tf.math.square(
            tf.math.maximum(0.9 - positive_similarities, 0.0)),
        axis=-1)

    # Apply spreadout if needed.
    loss = _update_loss_with_spreadout_loss(
        loss=loss,
        context_embedding=context_embedding,
        label_embedding=label_embedding,
        similarities=similarities,
        spreadout_context_lambda=self.spreadout_context_lambda,
        spreadout_label_lambda=self.spreadout_label_lambda,
        spreadout_cross_lambda=self.spreadout_cross_lambda)

    return loss

  def get_config(self):
    config = {
        'normalization_fn': self.normalization_fn,
        'expect_embeddings': self.expect_embeddings,
        'spreadout_context_lambda': self.spreadout_context_lambda,
        'spreadout_label_lambda': self.spreadout_label_lambda,
        'spreadout_cross_lambda': self.spreadout_cross_lambda,
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


class Spreadout(tf.keras.losses.Loss):
  """Compute spreadout loss.

  Optionally applies `normalization_fn` to context and label embeddings.
  `expect_embeddings` must be true.

  Allows the user to apply spreadout regularization with a given
  scaling lambda. Note that `spreadout_label_lambda` includes full vocab labels,
  but `spreadout_cross_lambda` includes only batch labels. Refer to description
  of `class BatchSoftmax` for more details about spreadout regularization.
  Note that the values of `spreadout_context_lambda`, `spreadout_label_lambda`
  and `spreadout_cross_lambda` are expected to be non-negative.

  If `use_global_similarity` is True, `y_true` is expected to be a
  pre-calculated global similarities matrix or a (context embeddings, full label
  embeddings) tuple which is used to calculate global similarities. Otherwise,
  `y_true` is either a pre-calculated batch similarities matrix or a (context
  embeddings, batch label embeddings) tuple which is used to calculate batch
  similarities.
  """

  def __init__(
      self,
      normalization_fn: utils.NormalizationFnType = utils.l2_normalize_fn,
      expect_embeddings: bool = True,
      spreadout_context_lambda: float = 0.0,
      spreadout_label_lambda: float = 0.0,
      spreadout_cross_lambda: float = 0.0,
      use_global_similarity: bool = False,
      **kwargs):

    if not expect_embeddings:
      raise ValueError('`expect_embeddings` must be true for Spreadout loss.')

    self.normalization_fn = normalization_fn
    self.expect_embeddings = expect_embeddings
    self.spreadout_context_lambda = spreadout_context_lambda
    self.spreadout_label_lambda = spreadout_label_lambda
    self.spreadout_cross_lambda = spreadout_cross_lambda
    self.use_global_similarity = use_global_similarity
    super().__init__(**kwargs)

  @tf.function
  def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:

    context_embedding, label_embedding, similarities = (
        utils.get_embeddings_and_similarities(y_pred, y_true,
                                              self.expect_embeddings,
                                              self.normalization_fn))

    if self.use_global_similarity:
      # Extract the batch similarities given the batch label (y_true).
      similarities = tf.gather(similarities, tf.transpose(y_true)[0], axis=1)

    # Compute spreadout loss.
    loss = _update_loss_with_spreadout_loss(
        loss=0.0,
        context_embedding=context_embedding,
        label_embedding=label_embedding,
        similarities=similarities,
        spreadout_context_lambda=self.spreadout_context_lambda,
        spreadout_label_lambda=self.spreadout_label_lambda,
        spreadout_cross_lambda=self.spreadout_cross_lambda)

    return loss

  def get_config(self):
    config = {
        'normalization_fn': self.normalization_fn,
        'expect_embeddings': self.expect_embeddings,
        'spreadout_context_lambda': self.spreadout_context_lambda,
        'spreadout_label_lambda': self.spreadout_label_lambda,
        'spreadout_cross_lambda': self.spreadout_cross_lambda,
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.function
def _compute_spreadout_loss(similarities, label_indices=None):
  if label_indices is None:
    label_indices = tf.range(tf.shape(similarities)[0])

  self_similarities = tf.gather(
      similarities, label_indices, axis=1, batch_dims=1)
  spreadout_loss = tf.sqrt(tf.reduce_sum(tf.square(similarities)) -
                           tf.reduce_sum(tf.square(self_similarities)))
  return spreadout_loss


@tf.function
def _update_loss_with_spreadout_loss(loss,
                                     context_embedding,
                                     label_embedding,
                                     similarities,
                                     spreadout_context_lambda,
                                     spreadout_label_lambda,
                                     spreadout_cross_lambda,
                                     label_indices=None):
  """Apply spreadout and update the loss value if needed."""

  # Apply context spreadout if needed.
  if spreadout_context_lambda:
    context_similarities = tf.matmul(
        context_embedding, context_embedding, transpose_b=True)
    loss += spreadout_context_lambda * _compute_spreadout_loss(
        context_similarities)

  # Apply label spreadout if needed.
  if spreadout_label_lambda:
    label_similarities = tf.matmul(
        label_embedding, label_embedding, transpose_b=True)
    loss += spreadout_label_lambda * _compute_spreadout_loss(
        label_similarities)

  # Apply cross spreadout if needed.
  if spreadout_cross_lambda:
    loss += spreadout_cross_lambda * _compute_spreadout_loss(
        similarities, label_indices)

  return loss
