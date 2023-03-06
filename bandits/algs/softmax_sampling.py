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
"""Softmax sampling for bandits inference."""

import collections
from collections.abc import Callable
from typing import Any, Optional

import tensorflow as tf
import tensorflow_federated as tff

from bandits import bandits_utils


@tf.function
def _softmax_actions(
    pred_logits: tf.Tensor,
    temperature: float,
) -> tuple[tf.Tensor, tf.Tensor]:
  """Returns the selected actions and all action probabilities based on softmax sampling.

  Args:
    pred_logits: Output logits of a minibatch; a tensor of size (batch_size,
      num_arms). The logits are expected in a roughly value range of [-1, 1].
    temperature: Parameter of softmax.
  """
  batch_size = tf.shape(pred_logits)[0]
  num_arms = tf.shape(pred_logits)[1]
  prob = tf.keras.activations.softmax(pred_logits / temperature, axis=-1)
  cumsum_prob = tf.math.cumsum(prob, axis=1)
  # We can use tf.random.uniform for independent noise on clients, see
  # https://www.tensorflow.org/federated/tutorials/random_noise_generation
  random_val = tf.random.uniform(
      shape=[batch_size], minval=0, maxval=1, dtype=tf.float32
  )
  less_idx = tf.math.less(tf.expand_dims(random_val, axis=1), cumsum_prob)
  # The `action_helper` is constructed so that if the `cumsum_prob` is larger
  # than or equal to `random_val`, the values are the action index; if
  # `cumsum_prob` is smaller, the values are a constant number of the largest
  # possible index of action. When taking the min of `action_helper`,
  # the action where random_val falls in the `cumsum_prob` backet is returned;
  # i.e., the action is sampled based on the softmax `prob`.
  action_helper = tf.where(
      less_idx,
      tf.broadcast_to(
          tf.range(num_arms, dtype=tf.int32), shape=[batch_size, num_arms]
      ),
      (num_arms - 1) * tf.ones(shape=(batch_size, num_arms), dtype=tf.int32),
  )
  action = tf.math.reduce_min(action_helper, axis=1)
  return action, prob


# TODO(b/215566681): the bandits inference function is based on batches after
# preprocessing, we should add a second preprocessing for training.
def build_softmax_bandit_data_fn(
    data_element_spec: Any,
    *,
    reward_fn: Optional[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]],
    temperature: float = 1,
) -> tuple[bandits_utils.BanditFnType, Any]:
  """Creates a function for FALCON bandits inference.

  The returned function will perform softmax sampling algorithms for bandits,
  inference, i.e., exploration with probability,
  prob(action) = exp(pred(action)/temprature) / sum(exp(pred/temprature))

  The returned function can be used as `bandit_data_fn` in
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
    temperature: The parameter to control exploration in softmax sampling.

  Returns:
    A tuple of (bandits_data_fn, bandit_data_spec).
  """
  if temperature <= 0:
    raise ValueError(f'temperature has to be positive, got {temperature}')

  if reward_fn is None:
    label_rank = data_element_spec[1].shape.rank
    assert label_rank >= 1 and label_rank <= 2
    sparse_label = True if label_rank == 1 else False
    reward_fn = bandits_utils.get_default_reward_fn(sparse_label=sparse_label)

  @tf.function
  def softmax_bandit_data(
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
      action, all_prob = _softmax_actions(
          pred_logits=pred_logits, temperature=temperature
      )
      prob = tf.gather(all_prob, action, axis=1, batch_dims=1)
      # Importance sampling weights are not used.
      weight_scale = 1.0
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
  return softmax_bandit_data, bandit_data_spec
