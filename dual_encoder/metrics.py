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
"""Metrics for dual encoder models.

Compatible with both server-side Keras and TFF.
"""

from typing import Optional

import tensorflow as tf

from dual_encoder import model_utils as utils


@tf.function
def _compute_recall(similarities, label_indices, recall_k):
  """Compute the label recall of each example."""

  # Get the indices of similar labels in sorted order for each context.
  sorted_similarities = tf.argsort(
      similarities, axis=-1, direction='DESCENDING')

  # Get the ranks of the correct label for each context.
  label_indices = tf.cast(label_indices, sorted_similarities.dtype)
  ranks = tf.where(tf.equal(sorted_similarities, label_indices))[:, -1]

  # Compare the ranks with recall_k to produce recall indicators for each
  # example.
  example_recalls = tf.math.less(ranks, recall_k)

  return example_recalls


class BatchRecall(tf.keras.metrics.Mean):
  """Keras metric computing label recall within a batch.

  Computes similarities between each context and label embedding (optionally
  normalized with `normalization_fn`) in a batch and for each context determines
  whether the correct label is within the first `recall_k` labels sorted by
  similarity. Returns the fraction of examples where the label is within the
  top `recall_k`.

  If `expect_embeddings` is False, then assumes `y_pred` is a pre-computed
  similarities matrix between context and label embeddings, and
  `normalization_fn` is ignored. This is useful to avoid recomputation of the
  similarities matrix across losses and metrics.

  Context and label embeddings or similarities are provided as `y_pred` and
  weights for each example in the batch is provided as `y_true`, to conform to
  the typical Keras pattern and enable TFF support.

  This class is called when using the batch recall as the eval metric. The model
  outputs (`y_pred`) should be the batch similarities or embeddings to compute
  the batch similarities. It produces exactly the same results as
  `BatchRecallWithGlobalSimilarity` with global similarities.
  """

  def __init__(self,
               recall_k: int = 10,
               normalization_fn:
               utils.NormalizationFnType = utils.l2_normalize_fn,
               expect_embeddings: bool = True,
               name: str = 'batch_recall',
               **kwargs):
    super().__init__(name=name, **kwargs)
    self.recall_k = recall_k
    self.normalization_fn = normalization_fn
    self.expect_embeddings = expect_embeddings

  def update_state(self,
                   y_true: tf.Tensor,
                   y_pred: tf.Tensor,
                   sample_weight: Optional[tf.Tensor] = None):
    """Compute the batch recall and update state of the metric.

    Args:
      y_true: The true labels with shape [batch_size, 1].
      y_pred: Model output. When `self.expect_embeddings` is True, `y_pred` is
        concatenate(context_embedding, label_embeddings) with shape [batch_size
        + batch_size, final_embedding_dim]. When `self.expect_embeddings` is
        False, `y_pred` is the similarity matrix with shape [batch_size,
        batch_size] between context and label embeddings.
      sample_weight: Optional weighting of each example. Defaults to 1.
    """
    _, _, similarities = (
        utils.get_embeddings_and_similarities(
            y_pred, y_true, self.expect_embeddings, self.normalization_fn))

    # Get the batch label indices for each example.
    curr_batch_size = tf.shape(similarities)[0]
    label_indices = tf.expand_dims(tf.range(curr_batch_size), -1)
    # Compute the example recall in batch.
    example_recalls = _compute_recall(
        similarities, label_indices, self.recall_k)

    super().update_state(example_recalls, sample_weight=sample_weight)

  def get_config(self):
    config = {
        'recall_k': self.recall_k,
        'normalization_fn': self.normalization_fn,
        'expect_embeddings': self.expect_embeddings,
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


class BatchRecallWithGlobalSimilarity(tf.keras.metrics.Mean):
  """Keras metric computing batch label recall when giving global similarities.

  Computes similarities between each context in a batch and full vocab label
  embedding. Optionally normalizes embeddings with `normalization_fn`. For each
  context determines whether the correct label is within the first `recall_k`
  labels sorted by similarity within a batch. Returns the fraction of examples
  where the label is within the top `recall_k` in batch.

  If `expect_embeddings` is False, then assumes y_pred is a pre-computed global
  similarities matrix between batch context and full vocab label embeddings, and
  `normalization_fn` is ignored. This is useful to avoid recomputation of the
  similarities matrix across losses and metrics.

  Context and full vocab label embeddings are provided as `y_pred` and the true
  label for each sample in the batch is provided as `y_true`, to conform to the
  typical Keras pattern and enable TFF support.

  This class is called when using the batch recall as the eval metric but the
  model outputs (`y_pred`) are the global similarities or embeddings to compute
  the batch similarities. It produces exactly the same results as `BatchRecall`
  without global similarities.
  """

  def __init__(self,
               recall_k: int = 10,
               normalization_fn:
               utils.NormalizationFnType = utils.l2_normalize_fn,
               expect_embeddings: bool = True,
               name: str = 'batch_recall_with_global_similarity',
               **kwargs):
    super().__init__(name=name, **kwargs)
    self.recall_k = recall_k
    self.normalization_fn = normalization_fn
    self.expect_embeddings = expect_embeddings

  def update_state(self,
                   y_true: tf.Tensor,
                   y_pred: tf.Tensor,
                   sample_weight: Optional[tf.Tensor] = None):
    """Compute the batch recall and update state of the metric.

    Args:
      y_true: The true labels with shape [batch_size, 1].
      y_pred: Model output. When `self.expect_embeddings` is True, `y_pred` is
        concatenate(context_embedding, full vocab label embeddings) with shape
        [batch_size + label_embedding_vocab_size, final_embedding_dim].
        When `self.expect_embeddings` is False, `y_pred` is the similarity
        matrix with shape [batch_size, label_embedding_vocab_size] between
        context and full vocab label embeddings.
      sample_weight: Optional weighting of each example. Defaults to 1.
    """
    _, _, similarities = (
        utils.get_embeddings_and_similarities(
            y_pred, y_true, self.expect_embeddings, self.normalization_fn))

    # Retrieve the batch similarities from the global similarities matrix with
    # the batch label (y_true).
    batch_similarities = tf.gather(
        similarities, tf.transpose(y_true)[0], axis=1)

    # Get the batch label indices for each example.
    curr_batch_size = tf.shape(similarities)[0]
    label_indices = tf.expand_dims(tf.range(curr_batch_size), -1)

    # Compute the example recall in batch.
    example_recalls = _compute_recall(
        batch_similarities, label_indices, self.recall_k)

    super().update_state(example_recalls, sample_weight)

  def get_config(self):
    config = {
        'recall_k': self.recall_k,
        'normalization_fn': self.normalization_fn,
        'expect_embeddings': self.expect_embeddings,
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


class GlobalRecall(tf.keras.metrics.Mean):
  """Keras metric computing global label recall for top_k.

  Computes global similarities between each context in a batch and the full
  vocab label embedding. Optionally normalizes embeddings with
  `normalization_fn`. For each context determines whether the correct label is
  within the first `recall_k` labels sorted by similarity. Returns the fraction
  of examples where the label is within the top `recall_k`.

  If `expect_embeddings` is False, then assumes `y_pred` is a pre-computed
  global similarities matrix between context and full vocab label embeddings,
  and `normalization_fn` is ignored. This is useful to avoid recomputation of
  the similarities matrix across losses and metrics.

  Context and full vocab label embeddings or similarities are provided as
  `y_pred` and the true label for each sample in the batch is provided as
  `y_true`.

  This class is called when using the global recall as the eval metric. The
  model should output the global similarities or embeddings to compute the
  global similarities.
  """

  def __init__(self,
               recall_k: int = 10,
               normalization_fn:
               utils.NormalizationFnType = utils.l2_normalize_fn,
               expect_embeddings: bool = True,
               name: str = 'global_recall',
               **kwargs):
    super().__init__(name=name, **kwargs)
    self.recall_k = recall_k
    self.normalization_fn = normalization_fn
    self.expect_embeddings = expect_embeddings

  def update_state(self,
                   y_true: tf.Tensor,
                   y_pred: tf.Tensor,
                   sample_weight: Optional[tf.Tensor] = None):
    """Compute the global recall and update state of the metric.

    Args:
      y_true: The true labels with shape [batch_size, 1].
      y_pred: Model output. When `self.expect_embeddings` is True, `y_pred` is
        concatenate(context_embedding, full vocab label embeddings) with shape
        [batch_size + label_embedding_vocab_size, final_embedding_dim].
        When `self.expect_embeddings` is False, `y_pred` is the similarity
        matrix with shape [batch_size, label_embedding_vocab_size] between
        context and full vocab label embeddings.
      sample_weight: Optional weighting of each example. Defaults to 1.
    """
    _, _, similarities = (
        utils.get_embeddings_and_similarities(
            y_pred, y_true, self.expect_embeddings, self.normalization_fn))

    # Compute the example recall among the full vocab.
    example_recalls = _compute_recall(similarities, y_true, self.recall_k)

    super().update_state(example_recalls, sample_weight)

  def get_config(self):
    config = {
        'recall_k': self.recall_k,
        'normalization_fn': self.normalization_fn,
        'expect_embeddings': self.expect_embeddings,
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


class BatchMeanRank(tf.keras.metrics.Mean):
  """Keras metric computing mean rank of correct label within a batch.

  Computes similarities between each context and label embedding in a batch and
  for each context gets the rank of the correct label among all of the labels in
  a batch, sorted by similarity. Returns the mean rank across all examples.
  Optionally normalizes embeddings with `normalization_fn`.

  If `expect_embeddings` is False, then assumes `y_pred` is a pre-computed
  similarities matrix between context and label embeddings, and
  `normalization_fn` is ignored. This is useful to avoid recomputation of the
  similarities matrix across losses and metrics.

  Context and label embeddings or similarities are provided as `y_pred` and
  weights for each example in the batch is provided as `y_true`, to conform to
  the typical Keras pattern and enable TFF support.
  """

  def __init__(self,
               normalization_fn:
               utils.NormalizationFnType = utils.l2_normalize_fn,
               expect_embeddings: bool = True,
               name: str = 'batch_mean_rank',
               **kwargs):
    super().__init__(name=name, **kwargs)
    self.normalization_fn = normalization_fn
    self.expect_embeddings = expect_embeddings

  def update_state(self,
                   y_true: tf.Tensor,
                   y_pred: tf.Tensor,
                   sample_weight: Optional[tf.Tensor] = None):

    _, _, similarities = (
        utils.get_embeddings_and_similarities(
            y_pred, y_true, self.expect_embeddings, self.normalization_fn))

    # Get the indices of similar labels in sorted order for each context.
    sorted_similarities = tf.argsort(
        similarities, axis=-1, direction='DESCENDING')

    # Get the ranks of the correct label for each context.
    curr_batch_size = tf.shape(similarities)[0]
    label_indices = tf.expand_dims(tf.range(curr_batch_size), -1)
    ranks = tf.where(tf.equal(sorted_similarities, label_indices))[:, -1]
    ranks = tf.cast(ranks, 'float32')

    super().update_state(ranks, sample_weight=sample_weight)

  def get_config(self):
    config = {
        'normalization_fn': self.normalization_fn,
        'expect_embeddings': self.expect_embeddings,
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


class GlobalMeanRank(tf.keras.metrics.Mean):
  """Keras metric computing mean rank of correct label globally.

  Computes similarities between each context in a batch and full vocab label
  embedding. For each context gets the rank of the correct label among all of
  the labels in the full vocab, sorted by similarity. Returns the mean rank
  across all examples. Optionally normalizes embeddings with `normalization_fn`.

  If `expect_embeddings` is False, then assumes `y_pred` is a pre-computed
  global similarities matrix between context and full vocab label embeddings,
  and `normalization_fn` is ignored. This is useful to avoid recomputation of
  the similarities matrix across losses and metrics.

  Context and full vocab label embeddings or similarities are provided as
  `y_pred` and the true label for each sample in the batch is provided as
  `y_true`, to conform to the typical Keras pattern and enable TFF support.
  """

  def __init__(self,
               normalization_fn:
               utils.NormalizationFnType = utils.l2_normalize_fn,
               expect_embeddings: bool = True,
               name: str = 'global_mean_rank',
               **kwargs):
    super().__init__(name=name, **kwargs)
    self.normalization_fn = normalization_fn
    self.expect_embeddings = expect_embeddings

  def update_state(self,
                   y_true: tf.Tensor,
                   y_pred: tf.Tensor,
                   sample_weight: Optional[tf.Tensor] = None):

    _, _, similarities = (
        utils.get_embeddings_and_similarities(
            y_pred, y_true, self.expect_embeddings, self.normalization_fn))

    # Get the indices of similar labels in sorted order for each context.
    sorted_similarities = tf.argsort(
        similarities, axis=-1, direction='DESCENDING')

    # Get the ranks of the correct label for each context.
    label_indices = tf.cast(y_true, dtype=tf.int32)
    ranks = tf.where(tf.equal(sorted_similarities, label_indices))[:, -1]
    ranks = tf.cast(ranks, 'float32')

    super().update_state(ranks, sample_weight=sample_weight)

  def get_config(self):
    config = {
        'normalization_fn': self.normalization_fn,
        'expect_embeddings': self.expect_embeddings,
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


class BatchSimilaritiesNorm(tf.keras.metrics.Mean):
  """Keras metric computing the Frobenius norm of context-label similarities.

  Computes similarities between each context and label embedding in a batch and
  then returns the Frobenius norm of the similarities matrix, neglecting the
  diagonal elements. This effectively measures how "spread-out" the context and
  label embeddings within a batch are. Optionally normalizes embeddings with
  `normalization_fn`.

  If `expect_embeddings` is False, then assumes `y_pred` is a pre-computed
  similarities matrix between context and label embeddings, and
  `normalization_fn` is ignored. This is useful to avoid recomputation of the
  similarities matrix across losses and metrics.

  Context and label embeddings or similarities are provided as `y_pred` to
  conform to the typical Keras pattern and enable TFF support. Note that
  `y_true` (which may be used to encode weights) is ignored for this metric
  since the norm is calculated per-batch, unlike `BatchRecall` above, which is
  calculated for each example in a batch and averaged. Thus, there isn't a
  natural notion of weighting per-example values for this metric.
  """

  def __init__(self,
               normalization_fn:
               utils.NormalizationFnType = utils.l2_normalize_fn,
               expect_embeddings: bool = True,
               name: str = 'batch_similarities_norm',
               **kwargs):
    super().__init__(name=name, **kwargs)
    self.normalization_fn = normalization_fn
    self.expect_embeddings = expect_embeddings

  def update_state(self,
                   y_true: tf.Tensor,
                   y_pred: tf.Tensor,
                   sample_weight: Optional[tf.Tensor] = None):

    _, _, similarities = (
        utils.get_embeddings_and_similarities(
            y_pred, y_true, self.expect_embeddings, self.normalization_fn))

    off_diagonal_similarities = tf.linalg.set_diag(
        similarities, tf.zeros(tf.shape(similarities)[0]))
    norm = tf.sqrt(
        tf.reduce_sum(tf.square(off_diagonal_similarities)))

    super().update_state(norm, sample_weight=sample_weight)

  def get_config(self):
    config = {
        'normalization_fn': self.normalization_fn,
        'expect_embeddings': self.expect_embeddings,
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


class NumExamplesCounter(tf.keras.metrics.Sum):
  """A custom sum that counts the number of examples seen.

  `sample_weight` is unused since this is just a counter.
  """

  def __init__(self,
               name: str = 'num_examples',
               **kwargs):
    super().__init__(name=name, **kwargs)

  def update_state(self, y_true, y_pred, sample_weight=None):
    return super().update_state(tf.shape(y_true)[0])


class NumBatchesCounter(tf.keras.metrics.Sum):
  """A custom sum that counts the number of batches seen.

  `sample_weight` is unused since this is just a counter.
  """

  def __init__(self,
               name: str = 'num_batches',
               **kwargs):
    super().__init__(name=name, **kwargs)

  def update_state(self, y_true, y_pred, sample_weight=None):
    return super().update_state(1)
