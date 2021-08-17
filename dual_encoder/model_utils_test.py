# Copyright 2021, Google LLC.
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

from absl.testing import absltest
import tensorflow as tf

from dual_encoder import model_utils as utils


class UtilsTest(absltest.TestCase):

  def test_get_predicted_embeddings_with_l2_normalize(self):
    y_pred = tf.constant(
        [[1, 0],
         [1.0, 1.0],
         [1, 1],
         [0, 1],
         [1, 0],
         [1, 0]]
    )
    y_true = tf.constant([[2], [3]])

    context_embeddings, label_embeddings = utils.get_predicted_embeddings(
        y_pred, y_true, normalization_fn=utils.l2_normalize_fn)

    expected_context_embeddings = tf.constant(
        [[1.0, 0.0],
         [0.7071067, 0.7071067]]
    )
    expected_label_embeddings = tf.constant(
        [[0.7071067, 0.7071067],
         [0.0, 1.0],
         [1.0, 0.0],
         [1.0, 0.0]]
    )

    tf.debugging.assert_near(context_embeddings, expected_context_embeddings)
    tf.debugging.assert_near(label_embeddings, expected_label_embeddings)

  def test_get_predicted_embeddings_without_normalization(self):
    y_pred = tf.constant(
        [[1, 0],
         [1.0, 1.0],
         [1, 1],
         [0, 1],
         [1, 0],
         [1, 0]]
    )
    y_true = tf.constant([[2], [3]])

    context_embeddings, label_embeddings = utils.get_predicted_embeddings(
        y_pred, y_true, normalization_fn=None)

    expected_context_embeddings = tf.constant(
        [[1.0, 0.0],
         [1.0, 1.0]]
    )
    expected_label_embeddings = tf.constant(
        [[1.0, 1.0],
         [0.0, 1.0],
         [1.0, 0.0],
         [1.0, 0.0]]
    )

    tf.debugging.assert_near(context_embeddings, expected_context_embeddings)
    tf.debugging.assert_near(label_embeddings, expected_label_embeddings)

  def test_get_embeddings_and_similarities(self):
    y_pred = tf.constant(
        [[1, 0],
         [1.0, 1.0],
         [1, 1],
         [0, 1],
         [1, 0],
         [1, 0]]
    )
    y_true = tf.constant([[2], [3]])

    context_embeddings, label_embeddings, similarities = (
        utils.get_embeddings_and_similarities(y_pred, y_true))

    expected_context_embeddings = tf.constant(
        [[1.0, 0.0],
         [0.7071067, 0.7071067]]
    )
    expected_label_embeddings = tf.constant(
        [[0.7071067, 0.7071067],
         [0.0, 1.0],
         [1.0, 0.0],
         [1.0, 0.0]]
    )
    expected_similarities = tf.constant(
        [[0.7071067, 0.0, 1.0, 1.0],
         [1.0, 0.7071067, 0.7071067, 0.7071067]]
    )

    tf.debugging.assert_near(context_embeddings, expected_context_embeddings)
    tf.debugging.assert_near(label_embeddings, expected_label_embeddings)
    tf.debugging.assert_near(similarities, expected_similarities)

  def test_get_embeddings_and_similarities_dot_product(self):
    y_pred = tf.constant(
        [[1, 0],
         [1.0, 1.0],
         [1, 1],
         [0, 1],
         [1, 0],
         [1, 0]]
    )
    y_true = tf.constant([[2], [3]])

    context_embeddings, label_embeddings, similarities = (
        utils.get_embeddings_and_similarities(
            y_pred, y_true, normalization_fn=None))

    expected_context_embeddings = tf.constant(
        [[1.0, 0.0],
         [1.0, 1.0]]
    )
    expected_label_embeddings = tf.constant(
        [[1.0, 1.0],
         [0.0, 1.0],
         [1.0, 0.0],
         [1.0, 0.0]]
    )
    expected_similarities = tf.constant(
        [[1.0, 0.0, 1.0, 1.0],
         [2.0, 1.0, 1.0, 1.0]]
    )

    tf.debugging.assert_near(context_embeddings, expected_context_embeddings)
    tf.debugging.assert_near(label_embeddings, expected_label_embeddings)
    tf.debugging.assert_near(similarities, expected_similarities)

  def test_get_embeddings_and_similarities_similarity(self):
    y_pred = tf.constant(
        [[0.7071067, 0.0, 1.0, 1.0],
         [1.0, 0.7071067, 0.7071067, 0.7071067]]
    )
    y_true = tf.constant([[2], [3]])

    context_embeddings, label_embeddings, similarities = (
        utils.get_embeddings_and_similarities(
            y_pred, y_true, expect_embeddings=False))

    expected_similarities = tf.constant(
        [[0.7071067, 0.0, 1.0, 1.0],
         [1.0, 0.7071067, 0.7071067, 0.7071067]]
    )

    self.assertIsNone(context_embeddings)
    self.assertIsNone(label_embeddings)
    tf.debugging.assert_near(similarities, expected_similarities)

  def test_similarities(self):
    similarities_layer = utils.Similarities()

    context_embedding = tf.constant(
        [[1, 2, 3],
         [4.0, 5.0, 6.0],
         [1, 1, 1],
         [1, 1, 1],
         [1, 1, 2]])
    label_embedding = tf.constant(
        [[1, 2, 3],
         [4.0, 5.0, 6.0],
         [1, 1, 1],
         [1, 1, 1],
         [-1, -2, -3]])

    similarities = similarities_layer([context_embedding, label_embedding])
    expected_similarities = tf.constant(
        [[0.9999999, 0.97463185, 0.92582005, 0.92582005, -0.9999999],
         [0.97463185, 1.0000001, 0.98692757, 0.98692757, -0.97463185],
         [0.92582005, 0.98692757, 0.99999994, 0.99999994, -0.92582005],
         [0.92582005, 0.98692757, 0.99999994, 0.99999994, -0.92582005],
         [0.98198044, 0.9770084, 0.942809, 0.942809, -0.98198044]])

    tf.debugging.assert_near(expected_similarities, similarities)

    # Also make sure layer.call works.
    similarities = similarities_layer.call([context_embedding, label_embedding])
    tf.debugging.assert_near(expected_similarities, similarities)

  def test_similarities_cosine_similarity(self):
    similarities_layer = utils.Similarities(normalization_fn=None)

    context_embedding = tf.constant(
        [[1, 2, 3],
         [4.0, 5.0, 6.0],
         [1, 1, 1],
         [1, 1, 1],
         [1, 1, 2]])
    label_embedding = tf.constant(
        [[1, 2, 3],
         [4.0, 5.0, 6.0],
         [1, 1, 1],
         [1, 1, 1],
         [-1, -2, -3]])

    similarities = similarities_layer([context_embedding, label_embedding])
    expected_similarities = tf.constant(
        [[14.0, 32, 6, 6, -14],
         [32, 77, 15, 15, -32],
         [6, 15, 3, 3, -6],
         [6, 15, 3, 3, -6],
         [9, 21, 4, 4, -9]])

    tf.debugging.assert_near(expected_similarities, similarities)

    # Also make sure layer.call works.
    similarities = similarities_layer.call([context_embedding, label_embedding])
    tf.debugging.assert_near(expected_similarities, similarities)

if __name__ == '__main__':
  absltest.main()
