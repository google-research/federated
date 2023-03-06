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
"""Epsilon-greedy algorithms for federated bandits."""

import collections
from collections.abc import Callable
from typing import Any, Optional

import tensorflow as tf
import tensorflow_federated as tff

from bandits import bandits_utils


# TODO(b/215566681): the bandits inference function is based on batches after
# preprocessing, we should add a second preprocessing for training.
def build_epsilon_greedy_bandit_data_fn(
    data_element_spec: Any,
    *,
    reward_fn: Optional[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]],
    epsilon: float = 0.2,
) -> tuple[bandits_utils.BanditFnType, Any]:
  """Creates a function for epsilon-greedy bandits inference.

  The returned function will perform epsilon-greedy algorithms for bandits,
  inference, i.e., exploration with epsilon probability. The returned function
  can be used as `bandit_data_fn` in
  `bandits_process.build_bandits_iterative_process` for bandits simulation in
  TFF.

  The bandits data returned by `bandit_data_fn` will match `bandit_data_spec`,
  which is an `OrderedDict` with key 'x' for data features, 'y' for bandits
  information generated during inference time, and 'y' itself is an
  `OrderedDict` with keys given by `bandits_utils.BanditsKeys`.

  Args:
    data_element_spec: The data element spec of the input dataset, which has the
      format of (data, label).
    reward_fn: A function returns `reward` for the given `action` based on input
      `labels`. If None, use `bandits_utils.get_default_reward_fn`.
    epsilon: The probabiliy of bandits exploration.

  Returns:
    A tuple of (bandits_data_fn, bandit_data_spec).
  """
  bandits_utils.check_zero_one(epsilon, 'epsilon')

  if reward_fn is None:
    label_rank = data_element_spec[1].shape.rank
    assert label_rank >= 1 and label_rank <= 2
    sparse_label = True if label_rank == 1 else False
    reward_fn = bandits_utils.get_default_reward_fn(sparse_label=sparse_label)

  @tf.function
  def epsilon_greedy_bandit_data(
      model: tff.learning.models.VariableModel,
      inference_model_weights: Any,
      dataset: tf.data.Dataset,
  ) -> tf.data.Dataset:
    model_vars = tff.learning.models.ModelWeights.from_model(model)
    tf.nest.map_structure(
        lambda v, t: v.assign(t), model_vars, inference_model_weights
    )

    def _batch_map_fn(features, labels):
      pred_logits = model.predict_on_batch(features, training=False)
      batch_size, arms_num = tf.shape(pred_logits)[0], tf.shape(pred_logits)[1]
      pred_action = tf.argmax(pred_logits, axis=1, output_type=tf.int32)
      # We can use tf.random.uniform for independent noise on clients, see
      # https://www.tensorflow.org/federated/tutorials/random_noise_generation
      random_action = tf.random.uniform(
          shape=[batch_size], minval=0, maxval=arms_num, dtype=tf.int32
      )
      epsilon_indicator = (
          tf.random.uniform(
              shape=[batch_size], minval=0, maxval=1, dtype=tf.float32
          )
          < epsilon
      )
      action = tf.where(epsilon_indicator, random_action, pred_action)
      action_match_pred = tf.cast(
          tf.math.equal(action, pred_action), dtype=tf.float32
      )
      per_action_epsilon = epsilon / tf.cast(arms_num, dtype=tf.float32)
      prob = (1 - epsilon + per_action_epsilon) * action_match_pred + (
          1.0 - action_match_pred
      ) * per_action_epsilon

      # weight_scale is used to make sure the importance sampling weight has
      # maximum ~1. As the importance sampling weights are 1/prob, i.e.,
      # 1/(1 - epsilon + per_action_epsilon) or 1/per_action_epsilon for
      # epsilon greedy, the weight_scale is the minimum of the two.
      weight_scale = tf.math.minimum(
          1 - epsilon + per_action_epsilon, per_action_epsilon
      )
      # Handles the edge case of greedy action when epsilon is 0.
      if tf.math.equal(weight_scale, tf.constant(0.0, dtype=tf.float32)):
        weight_scale = tf.constant(1.0, dtype=tf.float32)

      reward = reward_fn(labels, action)
      new_y = collections.OrderedDict([
          (bandits_utils.BanditsKeys.label, labels),
          (bandits_utils.BanditsKeys.action, action),
          (bandits_utils.BanditsKeys.reward, reward),
          (bandits_utils.BanditsKeys.prob, prob),
          (bandits_utils.BanditsKeys.weight_scale, weight_scale),
      ])
      return collections.OrderedDict(x=features, y=new_y)

    return dataset.map(
        _batch_map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

  bandit_data_spec = bandits_utils.supervised_to_bandits_data_spec(
      data_element_spec
  )
  return epsilon_greedy_bandit_data, bandit_data_spec
