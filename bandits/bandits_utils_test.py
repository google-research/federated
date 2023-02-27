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
"""Tests for bandits_utils."""

from absl.testing import parameterized
import tensorflow as tf

from bandits import bandits_utils

_DATA_DIM = 3


def _get_synthetic_dataset(
    positive_samples: int = 8,
    negative_samples: int = 16,
    batch_size: int = 4,
    non_separable_ratio: float = 0,
) -> tf.data.Dataset:
  # Create separable data where positive features corresponding to class 1, and
  # negative features corresponding to class 0. Flip the class label (to be
  # incorrect) for `non_separable_ratio` of total samples.
  features = tf.concat(
      [
          tf.ones([positive_samples, _DATA_DIM], dtype=tf.float32),
          -tf.ones([negative_samples, _DATA_DIM], dtype=tf.float32),
      ],
      axis=0,
  )
  non_separable_positive = int(positive_samples * non_separable_ratio)
  non_separable_negative = int(negative_samples * non_separable_ratio)
  if non_separable_positive > 0 and non_separable_negative > 0:
    labels = tf.concat(
        [
            tf.ones(
                [positive_samples - non_separable_positive], dtype=tf.int32
            ),
            tf.zeros([non_separable_positive], dtype=tf.int32),
            tf.zeros(
                [negative_samples - non_separable_negative], dtype=tf.int32
            ),
            tf.ones([non_separable_negative], dtype=tf.int32),
        ],
        axis=0,
    )
  else:
    labels = tf.concat(
        [
            tf.ones([positive_samples], dtype=tf.int32),
            tf.zeros([negative_samples], dtype=tf.int32),
        ],
        axis=0,
    )
  return (
      tf.data.Dataset.from_tensor_slices((features, labels))
      .shuffle(positive_samples + negative_samples)
      .batch(batch_size)
  )


class BanditsUtilsTest(tf.test.TestCase, parameterized.TestCase):

  def test_dataset_format_map_without_model(self):
    positive_samples, negative_samples, non_separable_ratio = 8, 16, 0.4
    supervised_dataset = _get_synthetic_dataset(
        positive_samples=positive_samples,
        negative_samples=negative_samples,
        non_separable_ratio=non_separable_ratio,
    )
    bandits_dataset = bandits_utils.dataset_format_map(
        dataset=supervised_dataset
    )
    for batch in bandits_dataset:
      label_shape = tf.shape(batch['y'][bandits_utils.BanditsKeys.label])
      self.assertAllEqual(
          batch['y'][bandits_utils.BanditsKeys.action], tf.zeros(label_shape)
      )
      self.assertAllEqual(
          batch['y'][bandits_utils.BanditsKeys.reward], tf.zeros(label_shape)
      )
      self.assertAllEqual(
          batch['y'][bandits_utils.BanditsKeys.prob], tf.ones(label_shape)
      )

  @parameterized.named_parameters(('sparse', True), ('multilabel', False))
  def test_default_reward_fn(self, sparse_label):
    label_size = 62
    labels = tf.constant(list(range(label_size)) * 2, dtype=tf.int32)
    correct_action = labels
    incorrect_action = (correct_action - 26) % label_size
    if not sparse_label:
      labels = tf.one_hot(labels, depth=label_size, dtype=tf.float32)
    reward_fn = bandits_utils.get_default_reward_fn(sparse_label=sparse_label)
    reward = reward_fn(labels, correct_action)
    self.assertAllEqual(
        reward, tf.constant(bandits_utils.REWARD_RIGHT, shape=[label_size * 2])
    )
    reward = reward_fn(labels, incorrect_action)
    self.assertAllEqual(
        reward, tf.constant(bandits_utils.REWARD_WRONG, shape=[label_size * 2])
    )

  def test_emnist_dist_shift_reward_fn(self):
    label_size = 62
    labels = tf.constant(list(range(label_size)) * 2, dtype=tf.int32)
    correct_action = labels
    incorrect_action = (correct_action - 1) % label_size
    inaccurate_action = (correct_action - 26) % label_size
    reward_fn = bandits_utils.get_emnist_dist_shift_reward_fn()
    reward = reward_fn(labels, correct_action)
    self.assertAllEqual(
        reward, tf.constant(bandits_utils.REWARD_RIGHT, shape=[label_size * 2])
    )
    reward = reward_fn(labels, incorrect_action)
    self.assertAllEqual(
        reward, tf.constant(bandits_utils.REWARD_WRONG, shape=[label_size * 2])
    )
    reward = reward_fn(labels, inaccurate_action)
    self.assertEqual(
        bandits_utils.REWARD_INACCURATE * 26.0 / 62.0, tf.reduce_mean(reward)
    )

  def test_stackoverflow_dist_shift_reward_fn(self):
    tag_size = 20
    labels = tf.constant(list(range(tag_size)), dtype=tf.int32)
    correct_action = labels
    incorrect_action = (correct_action - 1) % tag_size
    labels = tf.one_hot(labels, depth=tag_size, dtype=tf.float32)
    reward_fn = bandits_utils.get_stackoverflow_dist_shift_reward_fn(
        tag_size, use_synthetic_tag=True
    )
    reward = reward_fn(labels, correct_action)
    self.assertAllGreaterEqual(reward, bandits_utils.REWARD_RIGHT)
    for a, b in zip(reward[1:].numpy(), reward[:-1].numpy()):
      self.assertGreaterEqual(a, b)
    reward = reward_fn(labels, incorrect_action)
    self.assertAllEqual(
        reward, tf.constant(bandits_utils.REWARD_WRONG, shape=[tag_size])
    )


if __name__ == '__main__':
  tf.test.main()
