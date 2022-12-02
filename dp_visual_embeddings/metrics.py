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
"""Metrics for embedding models."""

from typing import Any, Optional

import tensorflow as tf

from dp_visual_embeddings.models import keras_utils


class EmbeddingCategoricalAccuracy(tf.keras.metrics.Metric):
  """Custom metric wrapping Keras CategoricalAccuracy metrics."""

  def __init__(self,
               name: str = 'embedding_categorical_accuracy_metric',
               **kwargs):
    super().__init__(name=name, **kwargs)
    self._keras_metric = tf.keras.metrics.SparseCategoricalAccuracy()

  def reset_state(self):
    self._keras_metric.reset_state()

  def update_state(self,
                   y_true: dict[str, Any],
                   y_pred: list[tf.Tensor],
                   sample_weight: Optional[tf.Tensor] = None):
    """Updates the metric states.

    Args:
      y_true: the groudtruth label for computing metrics. It is a dictionary
        structure with `identity_indices` as one of the keys, mapping to the
        sparse categorical label.
      y_pred: a length-2 list of [logits, embeddings] predicted by an embedding
        model.
      sample_weight: optional sample weight for the metric.
    """
    label = y_true['identity_indices']
    pred_logits, _ = y_pred
    self._keras_metric.update_state(label, pred_logits, sample_weight)

  def result(self) -> tf.Tensor:
    return self._keras_metric.result()


# Parameter for the `PairwiseRecallAtFAR` metric
_DEFAULT_NUM_THRESHOLDS = 200


class PairwiseRecallAtFAR(tf.keras.metrics.SensitivityAtSpecificity):
  """Estimates the recall, at a given FAR, for matching pairs of embeddings.

  Given pairs of `((embedding_1, true_id_1), (embedding_2, true_id_2))`, we can
  map this to a binary classification problem. For a similarity function `S` and
  threshold `t`, the prediction should be `S(embedding_1, embedding_2) > t`.
  This prediction is meant to produce the same result as the comparison
  `true_id_1 == true_id_2`, and it's consider a "true positive" when these
  match. The metric this class computes on the pairs is the recall at a given
  fixed false accept rate (FAR, a.k.a. false positive rate). This estimates the
  threshold at which that FAR is achieved, and returns the recall for that same
  threshold.

  Warning: The recall computed here is based on a lot of sampling and
  approximation. It only compares pairs of embeddings within each eval batch,
  and not across eval batches. Depending on the batch size and how the batches
  are sampled, this can make the metrics imprecise or biased. This metrics is
  mainly meant for initial estimates and tuning.
  """

  def __init__(self,
               far: float,
               num_thresholds: int = _DEFAULT_NUM_THRESHOLDS,
               name: Optional[str] = None,
               dtype: Optional[tf.DType] = None):
    """Initializes the metric object.

    Args:
      far: The False Accept Rate (FAR) at which to compute the recall.
      num_thresholds: The number of thresholds at which to compute the metrics.
        The threshold with the closest FAR to the right one will be used for the
        result. The thresholds are "fixed" constants evenly distributed in
        [0, 1] in the `tf.keras.metrics.SensitivityAtSpecificity`
        implementation.
      name: Name of the metric instance.
      dtype: Data type of the metric result.
    """
    # Specificity = 1 - FPR.
    super().__init__(
        1.0 - far, num_thresholds=num_thresholds, name=name, dtype=dtype)

  def update_state(self, embeddings: tf.Tensor, identities: tf.Tensor):
    """Accumulates statistics for a batch of embeddings and ground-truth IDs.

    Args:
      embeddings: Tensor of shape `[batch_size, embedding_dim]` with the
        recognition embeddings, or other feature vectors. The `batch_size` must
        be inferrable at graph construction time.
      identities: Tensor of identity labels, of shape `[batch_size]`.

    Returns:
      Update op from parent class.
    """
    tf.debugging.assert_rank(embeddings, 2)
    tf.debugging.assert_rank(identities, 1)
    batch_size = tf.shape(embeddings)[0]
    tf.debugging.assert_equal(tf.size(identities), batch_size)

    is_match = tf.math.equal(
        tf.expand_dims(identities, 1), tf.expand_dims(identities, 0))

    # A [batch_size, batch_size] matrix of the L2 distance between pairs of
    # embeddings.
    distance = tf.math.reduce_euclidean_norm(
        tf.expand_dims(embeddings, 1) - tf.expand_dims(embeddings, 0), axis=2)
    sq_distance = tf.square(distance)

    # For embeddings x,y it is (1 - ||x-y||_2^2/4).
    similarity = (
        tf.ones([batch_size, batch_size], dtype=sq_distance.dtype) -
        sq_distance / 4.)

    # Use 0/1 sample weights to ignore redundant pairs (or "pairs" that are the
    # same example repeated).
    index = tf.range(batch_size)
    right_index, left_index = tf.meshgrid(index, index)
    is_lower_triangle = tf.math.less(right_index, left_index)

    # Reshape to 1D, the super metric will treat each pair as an individual
    # classification example.
    num_pairs = batch_size * batch_size
    pair_labels = tf.reshape(is_match, [num_pairs])
    pair_similarities = tf.reshape(similarity, [num_pairs])
    pair_weights = tf.reshape(is_lower_triangle, [num_pairs])
    super().update_state(pair_labels, pair_similarities, pair_weights)


class EmbeddingRecallAtFAR(tf.keras.metrics.Metric):
  """Custom metric wrapping RecallAtFAR metrics."""

  def __init__(self,
               far: float,
               num_thresholds: int = 200,
               name: str = 'embedding_recall_at_far',
               **kwargs):
    super().__init__(name=name, **kwargs)
    self._keras_metric = PairwiseRecallAtFAR(
        far=far, num_thresholds=num_thresholds, name=name, dtype=tf.float32)
    self._far = far
    self._num_thresholds = num_thresholds
    self.norm_layer = keras_utils.EmbedNormLayer()

  def reset_state(self):
    self._keras_metric.reset_state()

  def update_state(self, y_true: dict[str, Any], y_pred: list[tf.Tensor]):
    """Updates the metric states.

    Args:
      y_true: the groudtruth label for computing metrics. It is a dictionary
        structure with `identity_indices` as one of the keys, mapping to the
        sparse categorical label.
      y_pred: a length-2 list of [logits, embeddings] predicted by an embedding
        model.
    """
    label = y_true['identity_indices']
    _, embeddings = y_pred
    # TODO(b/217480046): we normalize the embeddings here as this metric is used
    # for both training and validation. We should consider only use it for
    # validation.
    normalized_embeddings = self.norm_layer(embeddings, training=False)
    self._keras_metric.update_state(
        embeddings=normalized_embeddings, identities=label)

  def result(self) -> tf.Tensor:
    return self._keras_metric.result()

  def get_config(self) -> dict[str, Any]:
    # Metrics has to be constructed from `get_configs` in TFF.
    config = super().get_config()
    config['far'] = self._far
    config['num_thresholds'] = self._num_thresholds
    return config
