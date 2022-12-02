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
"""Tests for losses."""

import collections

from absl import logging
from absl.testing import parameterized

import tensorflow as tf

from dp_visual_embeddings import losses


def _build_fake_labels(batch_size, num_labels):
  identity_indices = tf.random.uniform([batch_size],
                                       maxval=num_labels,
                                       dtype=tf.int32)
  return collections.OrderedDict(identity_indices=identity_indices)


class LossesTest(tf.test.TestCase, parameterized.TestCase):

  def test_loss(self):
    batch_size = 7
    num_labels = 3
    labels = _build_fake_labels(batch_size=batch_size, num_labels=num_labels)

    logits = tf.random.normal(shape=(batch_size, num_labels))
    embedding_dim = 128
    embeddings = tf.random.uniform(
        shape=(batch_size, embedding_dim), minval=-1, maxval=1)
    predictions = (logits, embeddings)

    loss_fn = losses.EmbeddingLoss()
    loss_tensor = loss_fn(labels, predictions)
    loss_value = float(loss_tensor.numpy())
    logging.info('Loss value: %g', loss_value)


if __name__ == '__main__':
  tf.test.main()
