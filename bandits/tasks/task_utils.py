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
"""Tasks utils for bandits process in TFF simulation."""
from typing import Any, Optional

import tensorflow as tf
import tensorflow_federated as tff

from bandits import bandits_utils


class VecNormLayer(tf.keras.layers.Layer):
  """A keras layer to normalize the vector."""

  def call(self, vectors: tf.Tensor) -> tf.Tensor:
    norms = tf.norm(vectors, ord='euclidean', axis=1, keepdims=True)
    return tf.math.divide_no_nan(vectors, norms)


class WrapCategoricalAccuracy(tf.keras.metrics.SparseCategoricalAccuracy):
  """Wrap Keras `SparseCategoricalAccuracy` to support bandits data format."""

  def update_state(
      self,
      y_true: dict[str, Any],
      y_pred: tf.Tensor,
      sample_weight: Optional[tf.Tensor] = None,
  ):
    """Accumulates the metric statistics.

    Inherited from `tf.keras.metrics.SparseCategoricalAccuracy`, the shape of
    `y_true[bandits_utils.BanditsKeys.label]` and `y_pred` are different.

    Args:
      y_true: The groudtruth label for computing metrics. It is a dictionary
        structure with `label` as one of the keys.
      y_pred: The predicted probability/logits values.
      sample_weight: Optional per-example coefficients.
    """
    label = y_true[bandits_utils.BanditsKeys.label]
    super().update_state(label, y_pred, sample_weight)


class WeightCategoricalAccuracy(tf.keras.metrics.SparseCategoricalAccuracy):
  """`SparseCategoricalAccuracy` weighted by bandits importance sampling."""

  def update_state(self, y_true: dict[str, Any], y_pred: tf.Tensor):  # pytype: disable=signature-mismatch
    """Accumulates the metric statistics.

    Inherited from `tf.keras.metrics.SparseCategoricalAccuracy`, the shape of
    `y_true[bandits_utils.BanditsKeys.label]` and `y_pred` are different.
    Uses the bandits importance sampling weights for the metrics, which is
    `match(y_true, y_pred) * sample_weight / sum(sample_weight)`.

    Args:
      y_true: The groudtruth label for computing metrics. It is a dictionary
        structure with `label` as one of the keys.
      y_pred: The predicted probability/logits values.
    """
    label = y_true[bandits_utils.BanditsKeys.label]
    sample_weight = (
        y_true[bandits_utils.BanditsKeys.weight_scale]
        / y_true[bandits_utils.BanditsKeys.prob]
    )
    super().update_state(label, y_pred, sample_weight)


class WrapRecall(tf.keras.metrics.Recall):
  """Wrap Keras `Recall` to support bandits data format."""

  def update_state(
      self,
      y_true: dict[str, Any],
      y_pred: tf.Tensor,
      sample_weight: Optional[tf.Tensor] = None,
  ):
    """Accumulates the metric statistics.

    Inherited from `tf.keras.metrics.Recall`, the shape of
    `y_true[bandits_utils.BanditsKeys.label]` and `y_pred` are different.

    Args:
      y_true: The groudtruth label for computing metrics. It is a dictionary
        structure with `label` as one of the keys.
      y_pred: The predicted probability/logits values.
      sample_weight: Optional per-example coefficients.
    """
    label = y_true[bandits_utils.BanditsKeys.label]
    super().update_state(label, y_pred, sample_weight)


class BanditsMSELoss(tf.keras.losses.Loss):
  """Bandits mean square error loss.

  loss = mean(1/prob * || prediction(action) - reward ||^2)

  The `reward` is task and algorithm specific. A common practice to define
  rewards from a supervised simulation problem is to use '1' for correct
  prediction, and '0' for incorrect prediction. Section 2.4 of "A Contextual
  Bandit Bake-off" (https://arxiv.org/abs/1802.04064) suggests to define such
  values in {0, 1} or {-1, 0} with the majority of {0} for variance reduction.
  The "Bandit Bake-off" paper defines "loss" instead of "reward", which uses
  '0' for correct prediction, and '1' for incorrect prediction, and `argmin`
  instead of `argmax` for model inference.

  The `prediction` are raw values like neural network predictions before the
  softmax, which are epected to be in the same space as `reward`.

  The `prob` is between 0 and 1. If `importance_weighting`=False, no weighting
  based on `prob` will be applied, i.e.,
  loss = mean(|| prediction(action) - reward ||^2).
  """

  def __init__(self, importance_weighting: bool = True):
    super().__init__()
    self._importance_weighting = importance_weighting

  def call(self, y_true: dict[str, Any], y_pred: tf.Tensor) -> tf.Tensor:
    """Returns the loss.

    Args:
      y_true: The bandits data of `action`, `reward`, `prob` for each sample. A
        common practice to define rewards from a supervised simulation problem
        is to use '1' for correct prediction, and '0' for incorrect prediction.
        See "A Contextual Bandit Bake-off" (https://arxiv.org/abs/1802.04064)
        for examples.
      y_pred: The predicted logits values (e.g., neural network predictions
        before the softmax).
    """
    mse = tf.keras.losses.MeanSquaredError()
    action = y_true[bandits_utils.BanditsKeys.action]
    reward = y_true[bandits_utils.BanditsKeys.reward]
    action_logits = tf.gather(y_pred, action, axis=1, batch_dims=1)
    if self._importance_weighting:
      sample_weight = (
          y_true[bandits_utils.BanditsKeys.weight_scale]
          / y_true[bandits_utils.BanditsKeys.prob]
      )
    else:
      sample_weight = None
    loss = mse(
        tf.expand_dims(reward, axis=1),
        tf.expand_dims(action_logits, axis=1),
        sample_weight=sample_weight,
    )
    return loss


class BanditsCELoss(tf.keras.losses.Loss):
  """Bandits cross entropy error loss.

  loss = mean(1/prob * CE(prediction(action), reward))

  The `reward` has to use '1' for correct prediction, and '0' for incorrect
  prediction.

  The `prediction` are raw values like neural network predictions before the
  softmax, which are epected to be in the same space as `reward`.

  The `prob` is between 0 and 1. If `importance_weighting`=False, no weighting
  based on `prob` will be applied, i.e.,
  loss = mean(CE(prediction(action), reward)).
  """

  def __init__(self, importance_weighting: bool = True):
    super().__init__()
    self._importance_weighting = importance_weighting

  def call(self, y_true: dict[str, Any], y_pred: tf.Tensor) -> tf.Tensor:
    """Returns the loss.

    Args:
      y_true: The bandits data of `action`, `reward`, `prob` for each sample. A
        common practice to define rewards from a supervised simulation problem
        is to use '1' for correct prediction, and '0' for incorrect prediction.
        See "A Contextual Bandit Bake-off" (https://arxiv.org/abs/1802.04064)
        for examples.
      y_pred: The predicted logits values (e.g., neural network predictions
        before the softmax).
    """
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    action = y_true[bandits_utils.BanditsKeys.action]
    reward = y_true[bandits_utils.BanditsKeys.reward]
    if self._importance_weighting:
      sample_weight = (
          y_true[bandits_utils.BanditsKeys.weight_scale]
          / y_true[bandits_utils.BanditsKeys.prob]
      )
    else:
      sample_weight = None
    action_logits = tf.gather(y_pred, action, axis=1, batch_dims=1)
    loss = bce(
        tf.expand_dims(reward, axis=1),
        tf.expand_dims(action_logits, axis=1),
        sample_weight=sample_weight,
    )
    return loss


class SupervisedMSELoss(tf.keras.losses.Loss):
  """Mean square error loss for supervised training.

  loss = mean(||prediction - one_hot(label) ||^2)

  Computes the mean squared error of predicted rewards versus actual rewards for
  all actions, ignoring possible selected action, probability, and reward in
  bandits data.
  """

  def __init__(
      self,
      num_arms: int,
      reward_right: float = bandits_utils.REWARD_RIGHT,
      reward_wrong: float = bandits_utils.REWARD_WRONG,
  ):
    super().__init__()
    self._num_arms = num_arms
    self._reward_right = reward_right
    self._reward_wrong = reward_wrong

  def call(self, y_true: dict[str, Any], y_pred: tf.Tensor) -> tf.Tensor:
    """Returns the loss.

    Args:
      y_true: The data contains supervised `bandits_utils.BanditsKeys.label`.
      y_pred: The predicted logits values (e.g., neural network predictions
        before the softmax).
    """
    mse = tf.keras.losses.MeanSquaredError()
    label = y_true[bandits_utils.BanditsKeys.label]
    one_hot = tf.one_hot(
        label,
        depth=self._num_arms,
        on_value=self._reward_right,
        off_value=self._reward_wrong,
    )
    loss = mse(y_pred, one_hot)
    return loss


class SupervisedCELoss(tf.keras.losses.Loss):
  """Cross entropy loss for supervised training."""

  def call(self, y_true: dict[str, Any], y_pred: tf.Tensor) -> tf.Tensor:
    """Returns the loss.

    Args:
      y_true: The data contains supervised `bandits_utils.BanditsKeys.label`.
      y_pred: The predicted logits values (e.g., neural network predictions
        before the softmax).
    """
    sce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    label = y_true[bandits_utils.BanditsKeys.label]
    loss = sce(label, y_pred)
    return loss


class MultiLabelMSELoss(tf.keras.losses.Loss):
  """Mean square error loss for multi-label supervised training.

  loss = mean(||prediction - label||^2)

  Computes the mean squared error of predicted rewards versus actual rewards for
  all actions, ignoring possible selected action, probability, and reward in
  bandits data. The multi-label tensor has 0/1 values of same shape as
  prediction.
  """

  def __init__(
      self,
      reward_right: float = bandits_utils.REWARD_RIGHT,
      reward_wrong: float = bandits_utils.REWARD_WRONG,
  ):
    super().__init__()
    self._reward_right = reward_right
    self._reward_wrong = reward_wrong

  def call(self, y_true: dict[str, Any], y_pred: tf.Tensor) -> tf.Tensor:
    """Returns the loss.

    Args:
      y_true: The data contains supervised `bandits_utils.BanditsKeys.label`.
      y_pred: The predicted logits values (e.g., neural network predictions
        before the softmax).
    """
    mse = tf.keras.losses.MeanSquaredError()
    label = tf.cast(y_true[bandits_utils.BanditsKeys.label], tf.float32)
    label_reward = self._reward_right * label + self._reward_wrong * (
        1.0 - label
    )
    loss = mse(y_pred, label_reward)
    return loss


class MultiLabelCELoss(tf.keras.losses.Loss):
  """Cross entropy loss for multi-label supervised training.

  Use a binary cross entropy loss for each action/label.
  """

  def call(self, y_true: dict[str, Any], y_pred: tf.Tensor) -> tf.Tensor:
    """Returns the loss.

    Args:
      y_true: The data contains supervised `bandits_utils.BanditsKeys.label`.
      y_pred: The predicted logits values (e.g., neural network predictions
        before the softmax).
    """
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    label = y_true[bandits_utils.BanditsKeys.label]
    loss = bce(label, y_pred)
    return loss


def clientdata_for_select_clients(
    client_data: tff.simulation.datasets.ClientData,
    population_client_selection: str,
) -> tff.simulation.datasets.ClientData:
  """Returns ClientData by selecting clients from given `client_data`.

  `population_client_selection` is in the format of "start_index-end_index" for
  [start_index, end_index) of all the sorted clients in `client_data`. For
  example "0-1000" will select the first 1000 clients.

  Args:
    client_data: A given `tff.simulation.datasets.ClientData` to be selected
      from.
    population_client_selection: A string to provide the index for client
      selection.
  """
  client_ids = sorted(client_data.client_ids)
  id_str = population_client_selection.split('-')
  assert len(id_str) == 2
  start_idx, end_idx = int(id_str[0]), int(id_str[1])
  assert start_idx >= 0
  assert end_idx <= len(client_ids)
  assert start_idx <= end_idx
  return tff.simulation.datasets.ClientData.from_clients_and_tf_fn(
      client_ids[start_idx:end_idx],
      serializable_dataset_fn=client_data.serializable_dataset_fn,
  )
