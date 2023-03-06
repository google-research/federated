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
"""Tests for softmax_sampling."""
import collections
import functools
import math
import tensorflow as tf
import tensorflow_federated as tff

from bandits import bandits_utils
from bandits.algs import softmax_sampling


_DATA_DIM = 3
_DATA_ELEMENT_SPEC = [
    tf.TensorSpec([None, _DATA_DIM], dtype=tf.float32),
    tf.TensorSpec([None], dtype=tf.int32),
]


build_softmax_bandit_data_fn = functools.partial(
    softmax_sampling.build_softmax_bandit_data_fn, reward_fn=None
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


class SoftmaxTest(tf.test.TestCase):

  def test_bandits_data_spec(self):
    bandits_data_fn, bandits_data_spec = build_softmax_bandit_data_fn(
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

  def test_little_exploration(self):
    positive_samples, negative_samples = 4, 8
    bandits_data_fn, _ = build_softmax_bandit_data_fn(
        data_element_spec=_DATA_ELEMENT_SPEC,
        temperature=1e-5,  # little exploration
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
      self.assertAllClose(
          batch['y'][bandits_utils.BanditsKeys.prob], tf.ones(label_shape)
      )
      self.assertEqual(batch['y'][bandits_utils.BanditsKeys.weight_scale], 1.0)

  def test_exploration_exploitation(self):
    tf.random.set_seed(42)
    positive_samples, negative_samples = 800, 1600
    temperature = 0.5
    bandits_data_fn, _ = build_softmax_bandit_data_fn(
        data_element_spec=_DATA_ELEMENT_SPEC, temperature=temperature
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
    for batch in bandits_dataset:
      actions.append(batch['y'][bandits_utils.BanditsKeys.action])
      labels.append(batch['y'][bandits_utils.BanditsKeys.label])
      rewards.append(batch['y'][bandits_utils.BanditsKeys.reward])
      probs.append(batch['y'][bandits_utils.BanditsKeys.prob])
      self.assertEqual(1, batch['y'][bandits_utils.BanditsKeys.weight_scale])
    positive_ratio = float(positive_samples) / (
        positive_samples + negative_samples
    )
    self.assertNear(
        tf.reduce_mean(tf.cast(tf.concat(labels, axis=0), tf.float32)),
        positive_ratio,
        err=0.05,
    )
    exploitation_ratio = math.exp(1.0 / temperature) / (
        math.exp(1.0 / temperature) + math.exp(-1.0 / temperature)
    )
    self.assertGreater(
        tf.reduce_mean(tf.cast(tf.concat(actions, axis=0), tf.float32)),
        positive_ratio * exploitation_ratio * 0.95,
    )
    self.assertGreater(
        tf.reduce_mean(tf.concat(rewards, axis=0)),
        (
            bandits_utils.REWARD_RIGHT * exploitation_ratio
            + bandits_utils.REWARD_WRONG * (1 - exploitation_ratio)
        )
        * 0.95,
    )
    self.assertAllGreaterEqual(probs, 0.0)
    self.assertAllLessEqual(probs, 1.0)

  def test_softmax_actions(self):
    tf.random.set_seed(42)
    batch_size, arms_num = 2, 3
    temperature = 0.5
    pred_logits = tf.random.uniform(shape=[batch_size, arms_num])
    _, all_prob = softmax_sampling._softmax_actions(
        pred_logits, temperature=temperature
    )
    self.assertShapeEqual(pred_logits, all_prob)
    self.assertAllGreater(all_prob, 0.0)
    self.assertAllLess(all_prob, 1.0)
    self.assertAllClose(
        tf.math.reduce_sum(all_prob, axis=1), tf.ones([batch_size])
    )
    all_action = []
    for _ in range(2000):
      action, prob = softmax_sampling._softmax_actions(
          pred_logits, temperature=temperature
      )
      self.assertAllClose(prob, all_prob)
      all_action.extend(action.numpy())
    action_cnts = collections.Counter(all_action)
    cnts = [action_cnts[i] for i in range(arms_num)]
    cnt_prob = [i / sum(cnts) for i in cnts]
    self.assertAllClose(
        cnt_prob, tf.math.reduce_mean(all_prob, axis=0), rtol=0.07
    )

  def test_synthetic_prob(self):
    tf.random.set_seed(42)
    positive_samples, negative_samples = 160, 320
    temperature = 10.0
    bandits_data_fn, _ = build_softmax_bandit_data_fn(
        data_element_spec=_DATA_ELEMENT_SPEC, temperature=temperature
    )
    supervised_dataset = _get_synthetic_dataset(
        positive_samples=positive_samples,
        negative_samples=negative_samples,
        non_separable_ratio=0.5,
    )
    tff_model, model_weights = _get_synthetic_model()
    bandits_dataset = bandits_data_fn(
        model=tff_model,
        inference_model_weights=model_weights,
        dataset=supervised_dataset,
    )
    # Predicted logits are _DATA_DIM for the negative/positive all one vectors.
    exploitation_ratio = math.exp(_DATA_DIM / temperature) / (
        math.exp(_DATA_DIM / temperature) + math.exp(-_DATA_DIM / temperature)
    )
    probs = []
    for batch in bandits_dataset:
      probs.append(batch['y'][bandits_utils.BanditsKeys.prob])
    self.assertNear(tf.reduce_max(probs), exploitation_ratio, err=1e-7)
    self.assertNear(tf.reduce_min(probs), 1.0 - exploitation_ratio, err=1e-7)


if __name__ == '__main__':
  tf.test.main()
