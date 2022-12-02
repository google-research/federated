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
"""Losses and objectives used in federated training."""

from typing import Any

import tensorflow as tf


class EmbeddingLoss(tf.keras.losses.Loss):
  """Custom loss wrapping sparse cat x-e for embedding models."""

  def __init__(
      self,
      reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.AUTO,
      name: str = 'embedding_loss',
  ):
    super().__init__(reduction=reduction, name=name)
    self._reduction = reduction

  def call(self, y_true: dict[str, Any], y_pred: list[tf.Tensor]) -> tf.Tensor:
    """Returns the loss.

    Args:
      y_true: the groudtruth label for computing metrics. It is a dictionary
        structure with `identity_indices` as one of the keys, mapping to the
        sparse categorical label.
      y_pred: a length-2 list of [logits, embeddings] predicted by an embedding
        model.
    """
    label = y_true['identity_indices']
    pred_logits, _ = y_pred
    scce = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=self._reduction)
    loss = scce(label, pred_logits)
    return loss
