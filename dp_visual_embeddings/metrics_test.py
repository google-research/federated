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
"""Tests for metrics."""

import collections
import tensorflow as tf

from dp_visual_embeddings import metrics


class PairwiseRecallAtFARTest(tf.test.TestCase):

  def test_handles_empty_metric(self):
    metric = metrics.PairwiseRecallAtFAR(0.1)
    result = metric.result()
    self.assertEqual(result, 0)

  def test_handles_empty_batch(self):
    embeddings = tf.zeros([0, 17], dtype=tf.float32)
    identities = tf.zeros([0], dtype=tf.int32)

    metric = metrics.PairwiseRecallAtFAR(0.1)
    metric.update_state(embeddings, identities)
    result = metric.result()
    self.assertEqual(result, 0)

  def test_estimate_recall_at_far(self):
    embeddings = [
        [0.8, 0.6],
        [-0.6, 0.8],
        [0.6, 0.8],
        [-0.8, 0.6],
        [-0.8, -0.6],
    ]
    embeddings_tensor = tf.constant(embeddings, tf.float32)

    identities = ['Alice', 'Bob', 'Alice', 'Bob', 'Alice']
    identities_tensor = tf.constant(identities, tf.string)

    # The metric will select an embedding similarity threshold at which the
    # False Accept Rate (FAR) is 0.2. This threshold is expected to be
    # in the range approximately bound by 0.65 to 0.95.
    metric = metrics.PairwiseRecallAtFAR(0.2)
    metric.update_state(embeddings_tensor, identities_tensor)
    result = metric.result()
    # The recall at the selected threshold and FAR is expected to be 0.5, from
    # 2 true positive pairs and 2 false negative pairs.
    self.assertEqual(result, 0.5)

  def test_estimate_across_multiple_batches(self):
    good_embeddings = tf.concat([
        tf.tile([[1.0, 0.0]], [16, 1]),
        tf.tile([[0.0, 1.0]], [18, 1]),
    ],
                                axis=0)
    good_identities = tf.concat([
        tf.ones([16], tf.int32),
        tf.zeros([18], tf.int32),
    ],
                                axis=0)

    index = tf.random.shuffle(tf.range(16 + 18), seed=2)
    good_embeddings = tf.gather(good_embeddings, index, axis=0)
    good_identities = tf.gather(good_identities, index)

    metric = metrics.PairwiseRecallAtFAR(0.1)
    metric.update_state(good_embeddings, good_identities)
    self.assertEqual(metric.result(), 1.0)

    bad_embeddings = [
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ]
    bad_identities = [1, 0, 0, 1]
    metric.update_state(bad_embeddings, bad_identities)
    self.assertLess(metric.result(), 1.0 - 1e-3)

  def test_recall_increases_with_increasing_far(self):
    embeddings = tf.random.normal([64, 4], seed=17)
    embeddings /= tf.math.reduce_euclidean_norm(
        embeddings, axis=1, keepdims=True)
    identities = tf.random.uniform([64],
                                   minval=0,
                                   maxval=3,
                                   dtype=tf.dtypes.int32)

    target_fars = [0.05, 0.1, 0.3, 0.6, 0.9, 0.95]
    recalls = []
    for far in target_fars:
      metric = metrics.PairwiseRecallAtFAR(far)
      metric.update_state(embeddings, identities)
      recalls.append(metric.result())

    # Asserts that the "recalls" sequence is monotonically increasing.
    for recall_1, recall_2 in zip(recalls[:-1], recalls[1:]):
      self.assertLess(recall_1, recall_2)


def _build_fake_inputs(batch_size=2,
                       image_height_and_width=84,
                       num_identities=3):
  images = tf.zeros(
      [batch_size, image_height_and_width, image_height_and_width, 3])
  identity_indices = tf.random.uniform([batch_size],
                                       maxval=num_identities,
                                       dtype=tf.int32,
                                       seed=4)
  return collections.OrderedDict(
      x=collections.OrderedDict(images=images),
      y=collections.OrderedDict(identity_indices=identity_indices))


def _fake_pred(identity_indices, num_identities):
  tf.debugging.assert_rank(identity_indices, 1)
  batch_size = tf.shape(identity_indices)[0]
  row_indices = tf.range(batch_size)
  matrix_indices = tf.stack([row_indices, identity_indices], axis=1)
  return tf.scatter_nd(
      matrix_indices, tf.ones([batch_size]), shape=[batch_size, num_identities])


def _fake_embedding(identity_indices, embedding_size):
  tf.debugging.assert_rank(identity_indices, 1)
  batch_size = tf.shape(identity_indices)[0]
  row_indices = tf.range(batch_size)
  matrix_indices = tf.stack([row_indices, identity_indices], axis=1)
  return tf.scatter_nd(
      matrix_indices, tf.ones([batch_size]), shape=[batch_size, embedding_size])


class EmbeddingMetricsTest(tf.test.TestCase):

  def test_embedding_categorical_accuracy(self):
    acc_metric = metrics.EmbeddingCategoricalAccuracy()
    batch_size, num_identities = 8, 3
    batch_input = _build_fake_inputs(
        batch_size=batch_size, num_identities=num_identities)
    identity_indices = batch_input['y']['identity_indices']
    fake_correct_pred = [
        _fake_pred(identity_indices, num_identities),  # logits
        None,  # embeddings
    ]
    acc = acc_metric(batch_input['y'], fake_correct_pred)
    self.assertEqual(1.0, acc)
    fake_incorrect_pred = [1.0 - fake_correct_pred[0], None]
    acc_metric.reset_state()
    acc = acc_metric(batch_input['y'], fake_incorrect_pred)
    self.assertEqual(0.0, acc)
    acc_metric.reset_state()
    acc = acc_metric(batch_input['y'], fake_correct_pred)
    acc = acc_metric(batch_input['y'], fake_incorrect_pred)
    self.assertEqual(0.5, acc)

  def test_embedding_recall_at_far(self):
    my_metric = metrics.EmbeddingRecallAtFAR(far=0.1)
    batch_size, num_identities, embedding_size = 8, 3, 16
    batch_input = _build_fake_inputs(
        batch_size=batch_size, num_identities=num_identities)
    identity_indices = batch_input['y']['identity_indices']
    fake_correct_embedding = [
        None,  # logits
        _fake_embedding(identity_indices, embedding_size),  # embeddings
    ]
    self.assertEqual(1.0, my_metric(batch_input['y'], fake_correct_embedding))

if __name__ == '__main__':
  tf.test.main()
