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
"""Tests for epsilon_greedy."""

import functools

import tensorflow as tf
import tensorflow_federated as tff

from bandits import bandits_utils
from bandits.algs import epsilon_greedy
from bandits.tasks import task_utils

_DATA_DIM = 3
_DATA_ELEMENT_SPEC = [
    tf.TensorSpec([None, _DATA_DIM], dtype=tf.float32),
    tf.TensorSpec([None], dtype=tf.int32),
]


build_epsilon_greedy_bandit_data_fn = functools.partial(
    epsilon_greedy.build_epsilon_greedy_bandit_data_fn, reward_fn=None
)


def _get_synthetic_model():
  # Create a linear model with positive values for class 1 parameter
  # and negative values for class 0 parameter
  inputs = tf.keras.Input(shape=(_DATA_DIM,))
  outputs = tf.keras.layers.Dense(2, use_bias=False)(inputs)
  keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
  tff_model = tff.learning.from_keras_model(
      keras_model,
      # The loss is not used for inference.
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      input_spec=_DATA_ELEMENT_SPEC,
  )
  synthetic_weights = tff.learning.models.ModelWeights(
      trainable=[
          tf.concat(
              [
                  -tf.ones([_DATA_DIM, 1], dtype=tf.float32),
                  tf.ones([_DATA_DIM, 1], dtype=tf.float32),
              ],
              axis=1,
          )
      ],
      non_trainable=[],
  )
  return tff_model, synthetic_weights


def _get_synthetic_dataset(
    positive_samples: int = 8,
    negative_samples: int = 16,
    batch_size: int = 4,
    non_separable_ratio: float = 0,
) -> tf.data.Dataset:
  # Create separable data where positive features corresponding to class 1, and
  # negative features corresponding to class 0.
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


class EpsilonGreedyTest(tf.test.TestCase):

  def test_bandits_data_spec(self):
    bandits_data_fn, bandits_data_spec = build_epsilon_greedy_bandit_data_fn(
        data_element_spec=_DATA_ELEMENT_SPEC
    )
    self.assertLen(bandits_data_spec, 2)
    self.assertEqual(bandits_data_spec['x'], _DATA_ELEMENT_SPEC[0])
    self.assertLen(bandits_data_spec['y'], 5)
    self.assertEqual(
        bandits_data_spec['y'][bandits_utils.BanditsKeys.label],
        _DATA_ELEMENT_SPEC[1],
    )
    self.assertEqual(
        bandits_data_spec['y'][bandits_utils.BanditsKeys.action],
        tf.TensorSpec(shape=(None,), dtype=tf.int32),
    )
    self.assertEqual(
        bandits_data_spec['y'][bandits_utils.BanditsKeys.reward],
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    )
    self.assertEqual(
        bandits_data_spec['y'][bandits_utils.BanditsKeys.prob],
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    )
    self.assertEqual(
        bandits_data_spec['y'][bandits_utils.BanditsKeys.weight_scale],
        tf.TensorSpec(shape=(), dtype=tf.float32),
    )

    tff_model, model_weights = _get_synthetic_model()
    supervised_dataset = _get_synthetic_dataset(
        positive_samples=1, negative_samples=2
    )
    self.assertAllEqual(supervised_dataset.element_spec, _DATA_ELEMENT_SPEC)
    bandits_dataset = bandits_data_fn(
        model=tff_model,
        inference_model_weights=model_weights,
        dataset=supervised_dataset,
    )
    self.assertAllEqual(bandits_dataset.element_spec, bandits_data_spec)

  def test_no_exploration(self):
    positive_samples, negative_samples = 4, 8
    bandits_data_fn, _ = build_epsilon_greedy_bandit_data_fn(
        data_element_spec=_DATA_ELEMENT_SPEC,
        epsilon=0,  # no exploration
    )
    supervised_dataset = _get_synthetic_dataset(
        positive_samples=positive_samples, negative_samples=negative_samples
    )
    tff_model, model_weights = _get_synthetic_model()
    bandits_dataset = bandits_data_fn(
        model=tff_model,
        inference_model_weights=model_weights,
        dataset=supervised_dataset,
    )
    for batch in bandits_dataset:
      label_shape = tf.shape(batch['y'][bandits_utils.BanditsKeys.label])
      self.assertAllEqual(
          batch['y'][bandits_utils.BanditsKeys.label],
          batch['y'][bandits_utils.BanditsKeys.action],
      )
      self.assertAllEqual(
          batch['y'][bandits_utils.BanditsKeys.reward],
          tf.constant(bandits_utils.REWARD_RIGHT, shape=label_shape),
      )
      self.assertAllEqual(
          batch['y'][bandits_utils.BanditsKeys.prob], tf.ones(label_shape)
      )
      self.assertEqual(batch['y'][bandits_utils.BanditsKeys.weight_scale], 1.0)

  def test_pure_exploration(self):
    tf.random.set_seed(42)
    positive_samples, negative_samples = 400, 800
    bandits_data_fn, _ = build_epsilon_greedy_bandit_data_fn(
        data_element_spec=_DATA_ELEMENT_SPEC,
        epsilon=1,  # pure exploration, no exploitation
    )
    supervised_dataset = _get_synthetic_dataset(
        positive_samples=positive_samples, negative_samples=negative_samples
    )
    tff_model, model_weights = _get_synthetic_model()
    bandits_dataset = bandits_data_fn(
        model=tff_model,
        inference_model_weights=model_weights,
        dataset=supervised_dataset,
    )
    actions, labels, rewards = [], [], []
    for batch in bandits_dataset:
      label_shape = tf.shape(batch['y'][bandits_utils.BanditsKeys.label])
      actions.append(batch['y'][bandits_utils.BanditsKeys.action])
      labels.append(batch['y'][bandits_utils.BanditsKeys.label])
      rewards.append(batch['y'][bandits_utils.BanditsKeys.reward])
      self.assertAllEqual(
          batch['y'][bandits_utils.BanditsKeys.prob],
          tf.constant(0.5, shape=label_shape),
      )
      self.assertEqual(batch['y'][bandits_utils.BanditsKeys.weight_scale], 0.5)
    self.assertNear(
        tf.reduce_mean(tf.cast(tf.concat(labels, axis=0), tf.float32)),
        float(positive_samples) / (positive_samples + negative_samples),
        err=0.05,
    )
    # Two actions, random guess exploration will get 0.5.
    self.assertNear(
        tf.reduce_mean(tf.cast(tf.concat(actions, axis=0), tf.float32)),
        0.5,
        err=0.05,
    )
    self.assertNear(
        tf.reduce_mean(tf.concat(rewards, axis=0)),
        0.5 * bandits_utils.REWARD_RIGHT,
        err=0.05,
    )

  def test_exploration_exploitation(self):
    tf.random.set_seed(42)
    positive_samples, negative_samples = 800, 1600
    epsilon = 0.2
    bandits_data_fn, _ = build_epsilon_greedy_bandit_data_fn(
        data_element_spec=_DATA_ELEMENT_SPEC,
        epsilon=epsilon,
    )
    supervised_dataset = _get_synthetic_dataset(
        positive_samples=positive_samples, negative_samples=negative_samples
    )
    tff_model, model_weights = _get_synthetic_model()
    bandits_dataset = bandits_data_fn(
        model=tff_model,
        inference_model_weights=model_weights,
        dataset=supervised_dataset,
    )
    actions, labels, rewards, probs = [], [], [], []
    weight_scale = min([1 - epsilon + epsilon / 2, epsilon / 2.0])
    for batch in bandits_dataset:
      actions.append(batch['y'][bandits_utils.BanditsKeys.action])
      labels.append(batch['y'][bandits_utils.BanditsKeys.label])
      rewards.append(batch['y'][bandits_utils.BanditsKeys.reward])
      probs.append(batch['y'][bandits_utils.BanditsKeys.prob])
      self.assertEqual(
          weight_scale, batch['y'][bandits_utils.BanditsKeys.weight_scale]
      )
    positive_ratio = float(positive_samples) / (
        positive_samples + negative_samples
    )
    self.assertNear(
        tf.reduce_mean(tf.cast(tf.concat(labels, axis=0), tf.float32)),
        positive_ratio,
        err=0.05,
    )
    self.assertNear(
        tf.reduce_mean(tf.cast(tf.concat(actions, axis=0), tf.float32)),
        positive_ratio * (1 - epsilon) + 0.5 * epsilon,
        err=0.05,
    )
    self.assertNear(
        tf.reduce_mean(tf.concat(rewards, axis=0)),
        bandits_utils.REWARD_RIGHT * (1 - 0.5 * epsilon)
        + bandits_utils.REWARD_WRONG * 0.5 * epsilon,
        err=0.05,
    )
    self.assertNear(
        tf.reduce_mean(1.0 / tf.concat(probs, axis=0)), 2.0, err=0.15
    )
    self.assertLessEqual(
        tf.reduce_max(weight_scale / tf.concat(probs, axis=0)), 1.0
    )

  def test_exploration_exploitation_non_separable(self):
    tf.random.set_seed(42)
    positive_samples, negative_samples = 800, 1600
    epsilon, non_separable_ratio = 0.2, 0.4
    bandits_data_fn, _ = build_epsilon_greedy_bandit_data_fn(
        data_element_spec=_DATA_ELEMENT_SPEC, epsilon=epsilon
    )
    supervised_dataset = _get_synthetic_dataset(
        positive_samples=positive_samples,
        negative_samples=negative_samples,
        non_separable_ratio=non_separable_ratio,
    )
    tff_model, model_weights = _get_synthetic_model()
    bandits_dataset = bandits_data_fn(
        model=tff_model,
        inference_model_weights=model_weights,
        dataset=supervised_dataset,
    )
    actions, labels, rewards, probs = [], [], [], []
    weight_scale = min([1 - epsilon + epsilon / 2, epsilon / 2.0])
    for batch in bandits_dataset:
      actions.append(batch['y'][bandits_utils.BanditsKeys.action])
      labels.append(batch['y'][bandits_utils.BanditsKeys.label])
      rewards.append(batch['y'][bandits_utils.BanditsKeys.reward])
      probs.append(batch['y'][bandits_utils.BanditsKeys.prob])
      self.assertEqual(
          weight_scale, batch['y'][bandits_utils.BanditsKeys.weight_scale]
      )
    sample_positive_ratio = positive_samples / (
        positive_samples + negative_samples
    )
    label_positive_ratio = (
        positive_samples * (1 - non_separable_ratio)
        + negative_samples * non_separable_ratio
    ) / (positive_samples + negative_samples)
    self.assertNear(
        tf.reduce_mean(tf.cast(tf.concat(labels, axis=0), tf.float32)),
        label_positive_ratio,
        err=0.05,
    )
    self.assertNear(
        tf.reduce_mean(tf.cast(tf.concat(actions, axis=0), tf.float32)),
        sample_positive_ratio * (1 - epsilon) + 0.5 * epsilon,
        err=0.05,
    )
    mean_rewards = (
        bandits_utils.REWARD_RIGHT * (1 - 0.5 * epsilon)
        + bandits_utils.REWARD_WRONG * 0.5 * epsilon
    ) * (1 - non_separable_ratio) + (
        bandits_utils.REWARD_RIGHT * 0.5 * epsilon
        + bandits_utils.REWARD_WRONG * (1 - 0.5 * epsilon)
    ) * non_separable_ratio
    self.assertNear(
        tf.reduce_mean(tf.concat(rewards, axis=0)), mean_rewards, err=0.05
    )
    self.assertNear(
        tf.reduce_mean(1.0 / tf.concat(probs, axis=0)), 2.0, err=0.15
    )
    self.assertLessEqual(
        tf.reduce_max(weight_scale / tf.concat(probs, axis=0)), 1.0
    )

  def test_loss(self):
    tf.random.set_seed(42)
    positive_samples, negative_samples = 4, 8
    bandits_sample_times = 500
    epsilon, non_separable_ratio = 0.2, 0.4
    bandits_data_fn, _ = build_epsilon_greedy_bandit_data_fn(
        data_element_spec=_DATA_ELEMENT_SPEC, epsilon=epsilon
    )
    supervised_dataset = _get_synthetic_dataset(
        positive_samples=positive_samples,
        negative_samples=negative_samples,
        non_separable_ratio=non_separable_ratio,
    )
    tff_model, model_weights = _get_synthetic_model()
    bandits_dataset = bandits_data_fn(
        model=tff_model,
        inference_model_weights=model_weights,
        dataset=supervised_dataset.repeat(bandits_sample_times),
    )
    mse = tf.keras.losses.MeanSquaredError()
    true_loss = []
    for batch in supervised_dataset:
      pred_logits = tff_model.predict_on_batch(batch[0], training=False)
      target = tf.one_hot(
          batch[1],
          2,
          on_value=bandits_utils.REWARD_RIGHT,
          off_value=bandits_utils.REWARD_WRONG,
          axis=-1,
      )
      true_loss.append(mse(target, pred_logits))
    bmse = task_utils.BanditsMSELoss()
    bandit_loss = []
    for batch in bandits_dataset:
      pred_logits = tff_model.predict_on_batch(batch['x'], training=False)
      bandit_loss.append(bmse(batch['y'], pred_logits))
    # Scales bandit loss by the number of arms as the sum of sampling weights
    # (weights_scale/prob) is ~epsilon.
    self.assertNear(
        tf.reduce_mean(true_loss), tf.reduce_mean(bandit_loss) / epsilon, 0.5
    )


if __name__ == '__main__':
  tf.test.main()
