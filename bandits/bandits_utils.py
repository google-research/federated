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
"""Utility functions for federated bandits."""
import collections
from collections.abc import Callable, Sequence
import dataclasses
from typing import Any

import tensorflow as tf
import tensorflow_federated as tff


REWARD_RIGHT = 1.0
REWARD_WRONG = 0.0
REWARD_INACCURATE = (REWARD_RIGHT + REWARD_WRONG) * 0.5

BanditFnType = Callable[
    [tff.learning.models.VariableModel, Any, tf.data.Dataset], tf.data.Dataset
]


@dataclasses.dataclass
class BanditsKeys:
  """Structure defining the feedback for bandits data.

  The per-client training Dataset is represented as an (x, y) tuple
  in the usual Keras fashion, with `x` representing the context of
  the example (e.g., the features for a supervised learning problem).
  However, bandit algorithms need additional information about what happened
  at inference time in order to train, and we store this by making
  `y` and OrderedDict with keys as defined by this class.

  This class defines the string keys instead of the bandits
  data structure itself because TFF dataset serialization only supports
  `OrderedDict`.

  Attributes:
    label: Supervised label only used for evaluation.
    action: Action selected by the bandits algorithm during inference.
    reward: Corresponding reward of the action.
    prob: Corresponding probability of the action.
    weight_scale: An scalar weight for importance sampling in addition to
      1/prob.
  """

  label: str = 'label'
  action: str = 'action'
  reward: str = 'reward'
  prob: str = 'prob'
  weight_scale: str = 'weight_scale'


def check_data_element_spec(
    data_element_spec: Sequence[Any], expect_len: int = 2
):
  if len(data_element_spec) != expect_len:
    raise ValueError(
        'The elements of the preprocessed datasets for bandits inference have'
        'to be a tuple of (data, label).'
    )


def check_zero_one(value: float, name: str):
  if value < 0 or value > 1:
    raise ValueError(f'{name} must be between 0 and 1, get {value}')


def supervised_to_bandits_data_spec(data_element_spec: Any):
  check_data_element_spec(data_element_spec)
  return collections.OrderedDict(
      x=data_element_spec[0],
      y=collections.OrderedDict([
          (BanditsKeys.label, data_element_spec[1]),
          (
              BanditsKeys.action,
              tf.TensorSpec(shape=(None,), dtype=tf.int32),
          ),
          (
              BanditsKeys.reward,
              tf.TensorSpec(shape=(None,), dtype=tf.float32),
          ),
          (
              BanditsKeys.prob,
              tf.TensorSpec(shape=(None,), dtype=tf.float32),
          ),
          (
              BanditsKeys.weight_scale,
              tf.TensorSpec(shape=(), dtype=tf.float32),
          ),
      ]),
  )


def dataset_format_map(dataset: tf.data.Dataset) -> tf.data.Dataset:
  """Returns a supervised dataset that matches the format of bandits dataset.

  Add placeholder action/reward/prob so the returned dataset matches the
  `BanditsKeys` format for `y`. This function can transform supervised dataset
  to pass the TFF type check for bandits algorithms. The dataset then can be
  used for validation, or supervised algorithm baselines.

  Args:
    dataset: A supervised dataset; each sample is a (feature, label) tuple.
  """

  def map_fn(x, y):
    batch_size = tf.shape(y)[0]
    return collections.OrderedDict(
        x=x,
        y=collections.OrderedDict([
            (BanditsKeys.label, y),
            (BanditsKeys.action, tf.zeros([batch_size], dtype=tf.int32)),
            (BanditsKeys.reward, tf.zeros([batch_size], dtype=tf.float32)),
            (BanditsKeys.prob, tf.ones([batch_size], dtype=tf.float32)),
            (BanditsKeys.weight_scale, tf.constant(1, dtype=tf.float32)),
        ]),
    )

  return dataset.map(map_fn)


def get_default_reward_fn(
    reward_right: float = REWARD_RIGHT,
    reward_wrong: float = REWARD_WRONG,
    sparse_label: bool = True,
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
  """Returns a function transforming labels to rewards for given actions.

  The action will get `reward_right` if it is one of the correct labels, and
  will get `reward_wrong` if it is an incorrect label.

  Args:
    reward_right: The rewards for correct inference.
    reward_wrong: The rewards for incorrect inference.
    sparse_label: If True, assumes the input `labels` of the returned
      `reward_fn` is a vector of sparse labels. If False, assumes the input
      `labels` of the returned `reward_fn` is a matrix of multi-label 0/1
      values.
  """
  if sparse_label:  # Sparse categorical labels.

    def action_right_fn(labels, action):
      return tf.cast(tf.math.equal(action, labels), dtype=tf.float32)

  else:  # Multi-label 0/1 vectors.

    def action_right_fn(labels, action):
      return tf.gather(labels, action, axis=1, batch_dims=1)

  def reward_fn(labels, action):
    action_right = action_right_fn(labels, action)
    return reward_right * action_right + reward_wrong * (1.0 - action_right)

  return reward_fn


def get_emnist_dist_shift_reward_fn(
    reward_right: float = REWARD_RIGHT,
    reward_wrong: float = REWARD_WRONG,
    reward_inaccurate: float = REWARD_INACCURATE,
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
  """Returns a function transforming labels to rewards for given actions.

  The reward function is specialized for the distribution shift setting for
  EMNIST-62. Assuming the initial model will observe integer labels 10-35 for
  labels 36-61, recognizing 36-61 as 10-35 will be considered `inaccurate`
  instead of `wrong`.

  Args:
    reward_right: The rewards for correct inference.
    reward_wrong: The rewards for incorrect inference.
    reward_inaccurate: The rewards for inaccurate inference. Specialized for
      EMNIST-62 distribution shift setting.
  """

  def reward_fn(labels, action):
    action_right = tf.cast(tf.math.equal(action, labels), dtype=tf.float32)
    indicator = tf.cast(tf.math.equal(action, labels - 26), dtype=tf.float32)
    action_inacc = tf.where(labels > 35, indicator, 0)
    return (
        reward_right * action_right
        + reward_inaccurate * action_inacc
        + reward_wrong * (1.0 - action_right - action_inacc)
    )

  return reward_fn


def get_stackoverflow_dist_shift_reward_fn(
    tag_size: int,
    reward_right: float = REWARD_RIGHT,
    reward_wrong: float = REWARD_WRONG,
    use_synthetic_tag: bool = False,
) -> Callable[[tf.Tensor, tf.Tensor], tf.Tensor]:
  """Returns a function transforming multi-labels to rewards for given actions.

  The reward function is specialized for the distribution shift setting for
  StackOverflow. The rewards are inverse propotional to the frequency of the
  top tags, i.e.,
  reward(top_i_tag) = reward_right * top_1_frequency/top_i_frequency. For the
  top 50 tags of StackOverflow, the tag counts are in [210151, 43035], so the
  reward scales are in [1, 5).

  Args:
    tag_size: The total number of tags.
    reward_right: The rewards for correct inference.
    reward_wrong: The rewards for incorrect inference.
    use_synthetic_tag: If True, use synthetic for testing.
  """
  if use_synthetic_tag:
    tag_cnts = tff.simulation.datasets.stackoverflow.get_synthetic_tag_counts()
  else:
    tag_cnts = tff.simulation.datasets.stackoverflow.load_tag_counts()
  cnts = list(tag_cnts.values())[:tag_size]
  reward_scales = [float(cnts[0]) / x for x in cnts]
  print('scales', len(reward_scales), ':', reward_scales)

  def reward_fn(labels, action):
    action_right = tf.gather(labels, action, axis=1, batch_dims=1)
    scales = tf.gather(reward_scales, action)
    return reward_right * action_right * scales + reward_wrong * (
        1.0 - action_right
    )

  return reward_fn
