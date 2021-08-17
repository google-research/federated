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

from dual_encoder import losses
from dual_encoder import model_utils as utils


class LossesTest(absltest.TestCase):

  def test_batch_softmax(self):
    loss = losses.BatchSoftmax()

    y_pred = tf.constant([[1, 2, 3], [4.0, 5.0, 6.0], [1, 1, 1], [1, 1, 1],
                          [1, 2, 3], [4.0, 5.0, 6.0], [1, 1, 1], [1, 1, 1]])
    y_true = tf.constant([1.0, 1.0, 1.0, 1.0])

    # Test both Keras-internal call and external call since behavior may be
    # slightly different due to type/shape conversion.
    loss_value = loss(y_true, y_pred)
    expected_loss_value = 1.3616819

    tf.debugging.assert_near(expected_loss_value, loss_value)

    loss_value = loss.call(y_true, y_pred)
    expected_loss_value = 1.3616819

    tf.debugging.assert_near(expected_loss_value, loss_value)

  def test_batch_softmax_similarities(self):
    loss = losses.BatchSoftmax(expect_embeddings=False)

    y_pred = tf.constant(
        [[0.9999999, 0.97463185, 0.92582005, 0.92582005, -0.9999999],
         [0.97463185, 1.0000001, 0.98692757, 0.98692757, -0.97463185],
         [0.92582005, 0.98692757, 0.99999994, 0.99999994, -0.92582005],
         [0.92582005, 0.98692757, 0.99999994, 0.99999994, -0.92582005],
         [0.98198044, 0.9770084, 0.942809, 0.942809, -0.98198044]])
    y_true = tf.constant([1.0, 1.0, 1.0, 1.0, 1.0])

    # Test both Keras-internal call and external call since behavior may be
    # slightly different due to type/shape conversion.
    loss_value = loss(y_true, y_pred)
    expected_loss_value = 1.7907718

    tf.debugging.assert_near(expected_loss_value, loss_value)

    loss_value = loss.call(y_true, y_pred)
    expected_loss_value = 1.7907718

    tf.debugging.assert_near(expected_loss_value, loss_value)

  def test_batch_softmax_y_true_expand_dims(self):
    loss = losses.BatchSoftmax()

    y_pred = tf.constant([[1, 2, 3], [4.0, 5.0, 6.0], [1, 1, 1], [1, 1, 1],
                          [1, 2, 3], [4.0, 5.0, 6.0], [1, 1, 1], [1, 1, 1]])
    y_true = tf.expand_dims(tf.constant([1.0, 1.0, 1.0, 1.0]), -1)

    # Test both Keras-internal call and external call since behavior may be
    # slightly different due to type/shape conversion.
    loss_value = loss(y_true, y_pred)
    expected_loss_value = 1.3616819

    tf.debugging.assert_near(expected_loss_value, loss_value)

    loss_value = loss.call(y_true, y_pred)
    expected_loss_value = 1.3616819

    tf.debugging.assert_near(expected_loss_value, loss_value)

  def test_batch_softmax_context_label_spreadout_no_embeddings(self):
    with self.assertRaises(ValueError):
      losses.BatchSoftmax(expect_embeddings=False, spreadout_context_lambda=0.1)

    with self.assertRaises(ValueError):
      losses.BatchSoftmax(expect_embeddings=False, spreadout_label_lambda=0.1)

    with self.assertRaises(ValueError):
      losses.BatchSoftmax(
          expect_embeddings=False,
          spreadout_context_lambda=0.1,
          spreadout_label_lambda=0.1)

  def test_batch_softmax_y_true_2d(self):
    loss = losses.BatchSoftmax()

    y_pred = tf.constant([[1, 2, 3], [4.0, 5.0, 6.0], [1, 1, 1], [1, 1, 1],
                          [1, 2, 3], [4.0, 5.0, 6.0], [1, 1, 1], [1, 1, 1]])
    y_true = tf.constant([[1.0], [1.0], [1.0], [1.0]])

    # Test both Keras-internal call and external call since behavior may be
    # slightly different due to type/shape conversion.
    loss_value = loss(y_true, y_pred)
    expected_loss_value = 1.3616819

    tf.debugging.assert_near(expected_loss_value, loss_value)

    loss_value = loss.call(y_true, y_pred)
    expected_loss_value = 1.3616819

    tf.debugging.assert_near(expected_loss_value, loss_value)

  def test_batch_softmax_weighted(self):
    loss = losses.BatchSoftmax()

    y_pred = tf.constant([[1, 2, 3], [4.0, 5.0, 6.0], [1, 1, 1], [1, 1, 1],
                          [1, 1, 2], [1, 2, 3], [4.0, 5.0, 6.0], [1, 1, 1],
                          [1, 1, 1], [-1, -2, -3]])
    y_true = tf.constant([1.0, 1.0, 1.0, 1.0, 3.0])

    loss_value = loss(y_true, y_pred)
    expected_loss_value = 2.2404877798897878

    tf.debugging.assert_near(expected_loss_value, loss_value)

  def test_batch_softmax_weighted_y_true_2d(self):
    loss = losses.BatchSoftmax()

    y_pred = tf.constant([[1, 2, 3], [4.0, 5.0, 6.0], [1, 1, 1], [1, 1, 1],
                          [1, 1, 2], [1, 2, 3], [4.0, 5.0, 6.0], [1, 1, 1],
                          [1, 1, 1], [-1, -2, -3]])
    y_true = tf.constant([[1.0], [1.0], [1.0], [1.0], [3.0]])

    loss_value = loss(y_true, y_pred)
    expected_loss_value = 2.2404877798897878

    tf.debugging.assert_near(expected_loss_value, loss_value)

  def test_batch_softmax_dot_product(self):
    loss = losses.BatchSoftmax(normalization_fn=None)

    y_pred = tf.constant([[1, 2, 3], [4.0, 5.0, 6.0], [1, 1, 1], [1, 1, 1],
                          [1, 2, 3], [4.0, 5.0, 6.0], [1, 1, 1], [1, 1, 1]])
    y_true = tf.constant([1.0, 1.0, 1.0, 1.0])

    # Test both Keras-internal call and external call since behavior may be
    # slightly different due to type/shape conversion.
    loss_value = loss(y_true, y_pred)
    expected_loss_value = 10.500068

    tf.debugging.assert_near(expected_loss_value, loss_value)

    loss_value = loss.call(y_true, y_pred)
    expected_loss_value = 10.500068

    tf.debugging.assert_near(expected_loss_value, loss_value)

  def test_batch_softmax_spreadout(self):
    loss = losses.BatchSoftmax(
        spreadout_context_lambda=0.1,
        spreadout_label_lambda=0.2,
        spreadout_cross_lambda=0.3)

    y_pred = tf.constant([[1, 2, 3], [4.0, 5.0, 6.0], [-1, -2, -3], [1, 1, 1],
                          [1, 2, 3], [4.0, 5.0, 6.0], [-2, 4, 6], [1, 1, 1]])
    y_true = tf.constant([1.0, 1.0, 1.0, 1.0])

    # Test both Keras-internal call and external call since behavior may be
    # slightly different due to type/shape conversion.
    loss_value = loss(y_true, y_pred)
    expected_loss_value = 3.1863585

    tf.debugging.assert_near(expected_loss_value, loss_value)

    loss_value = loss.call(y_true, y_pred)
    expected_loss_value = 3.1863585

    tf.debugging.assert_near(expected_loss_value, loss_value)

  def test_batch_softmax_spreadout_weighted(self):
    loss = losses.BatchSoftmax(
        spreadout_context_lambda=0.1,
        spreadout_label_lambda=0.2,
        spreadout_cross_lambda=0.3)

    y_pred = tf.constant([[1, 2, 3], [4.0, 5.0, 6.0], [1, 1, 1], [1, 1, 1],
                          [1, 1, 2], [1, 2, 3], [4.0, 5.0, 6.0], [1, 1, 1],
                          [1, 1, 1], [-1, -2, -3]])
    y_true = tf.constant([1.0, 1.0, 1.0, 1.0, 3.0])

    # Test both Keras-internal call and external call since behavior may be
    # slightly different due to type/shape conversion.
    loss_value = loss(y_true, y_pred)
    expected_loss_value = 4.8267508

    tf.debugging.assert_near(expected_loss_value, loss_value)

    loss_value = loss.call(y_true, y_pred)
    expected_loss_value = 4.8267508

    tf.debugging.assert_near(expected_loss_value, loss_value)

  def test_batch_softmax_spreadout_weighted_dot_product(self):
    loss = losses.BatchSoftmax(
        spreadout_context_lambda=0.1,
        spreadout_label_lambda=0.2,
        spreadout_cross_lambda=0.3,
        normalization_fn=None)

    y_pred = tf.constant(
        [[1, 0, 0], [2, 0, 0], [0, 1, 0], [0, 2, 0], [0, 0, 1], [1, 0, 0],
         [2, 0, 0], [0, 1, 0], [0, 2, 0], [0, 0, 1]],
        dtype=tf.float32)
    # All 0 weights to test spreadout only.
    y_true = tf.constant([0.0, 0.0, 0.0, 0.0, 0.0])

    # Similarities (context, label, and cross):
    # 1, 2, 0, 0, 0
    # 2, 4, 0, 0, 0
    # 0, 0, 1, 2, 0
    # 0, 0, 2, 4, 0
    # 0, 0, 0, 0, 1

    # Off-diagonal elements:
    # 0, 2, 0, 0, 0
    # 2, 0, 0, 0, 0
    # 0, 0, 0, 2, 0
    # 0, 0, 2, 0, 0
    # 0, 0, 0, 0, 0

    # Spreadout loss value (for each of context, label, cross):
    # sqrt(4 * 2^2) = 4

    # Test both Keras-internal call and external call since behavior may be
    # slightly different due to type/shape conversion.
    loss_value = loss(y_true, y_pred)
    expected_loss_value = 2.4

    tf.debugging.assert_near(expected_loss_value, loss_value)

    loss_value = loss.call(y_true, y_pred)
    expected_loss_value = 2.4

    tf.debugging.assert_near(expected_loss_value, loss_value)

  def test_batch_softmax_spreadout_weighted_dot_product_multiple_similarities(
      self):
    loss = losses.BatchSoftmax(
        spreadout_context_lambda=0.4,
        spreadout_label_lambda=0.3,
        spreadout_cross_lambda=0.7,
        normalization_fn=None)

    y_pred = tf.constant(
        [[1, 0, 0], [2, 0, 0], [0, 1, 0], [0, 2, 0], [0, 0, 1], [1, 0, 0],
         [2, 1, 0], [0, 1, 0], [0, 2, 0], [0, 2, 1]],
        dtype=tf.float32)
    # All 0 weights to test spreadout only.
    y_true = tf.constant([0.0, 0.0, 0.0, 0.0, 0.0])

    # Context off-diagonal elements:
    # 0, 2, 0, 0, 0
    # 2, 0, 0, 0, 0
    # 0, 0, 0, 2, 0
    # 0, 0, 2, 0, 0
    # 0, 0, 0, 0, 0
    # Spreadout loss: sqrt(4 * 2^2) = 4

    # Label off-diagonal elements:
    # 0, 2, 0, 0, 0
    # 2, 0, 1, 2, 2
    # 0, 1, 0, 2, 2
    # 0, 2, 2, 0, 4
    # 0, 2, 2, 4, 0
    # Spreadout loss: sqrt(2 * 1^2 + 10 * 2^2 + 2 * 4^2) = 8.602325267

    # Cross off-diagonal elements:
    # 0, 2, 0, 0, 0
    # 2, 0, 0, 0, 0
    # 0, 1, 0, 2, 2
    # 0, 2, 2, 0, 4
    # 0, 0, 0, 0, 0
    # Spreadout loss: sqrt(1 * 1^2 + 6 * 2^2 + 1 * 4^2) = 6.4031242374

    # Weighted spreadout loss value:
    # 0.4 * 4 + 0.3 * 8.602325267 + 0.7 * 6.4031242374 = 8.6628845463

    # Test both Keras-internal call and external call since behavior may be
    # slightly different due to type/shape conversion.
    loss_value = loss(y_true, y_pred)
    expected_loss_value = 8.6628845463

    tf.debugging.assert_near(expected_loss_value, loss_value)

    loss_value = loss.call(y_true, y_pred)
    expected_loss_value = 8.6628845463

    tf.debugging.assert_near(expected_loss_value, loss_value)

  def test_batch_softmax_get_config(self):
    loss = losses.BatchSoftmax()

    config = loss.get_config()
    expected_config = {
        'normalization_fn': utils.l2_normalize_fn,
        'expect_embeddings': True,
        'spreadout_context_lambda': 0.0,
        'spreadout_label_lambda': 0.0,
        'spreadout_cross_lambda': 0.0,
        'name': None,
        'reduction': 'auto',
        'label_indices_fn': tf.range,
    }
    self.assertEqual(config, expected_config)

  def test_batch_softmax_get_config_keyword_args(self):
    label_indices_fn_instance = lambda x: tf.cast(tf.zeros((x,)), tf.int32)
    loss = losses.BatchSoftmax(
        normalization_fn=None,
        expect_embeddings=False,
        spreadout_context_lambda=0.0,
        spreadout_label_lambda=0.0,
        spreadout_cross_lambda=0.1,
        label_indices_fn=label_indices_fn_instance,
        name='my_loss')

    config = loss.get_config()
    expected_config = {
        'normalization_fn': None,
        'expect_embeddings': False,
        'spreadout_context_lambda': 0.0,
        'spreadout_label_lambda': 0.0,
        'spreadout_cross_lambda': 0.1,
        'name': 'my_loss',
        'reduction': 'auto',
        'label_indices_fn': label_indices_fn_instance,
    }
    self.assertEqual(config, expected_config)

  def test_batch_softmax_with_global_similarity(self):
    loss = losses.BatchSoftmaxWithGlobalSimilarity()

    y_pred = tf.constant([[1, 0], [0.0, 1.0], [1, 1], [0, 1], [1, 0], [1, 0]])
    y_true = tf.constant([[1], [2]])

    # Test both Keras-internal call and external call since behavior may be
    # slightly different due to type/shape conversion.
    loss_value = loss(y_true, y_pred)
    expected_loss_value = 1.3132616

    tf.debugging.assert_near(expected_loss_value, loss_value)

    loss_value = loss.call(y_true, y_pred)
    expected_loss_value = 1.3132616

    tf.debugging.assert_near(expected_loss_value, loss_value)

  def test_batch_softmax_with_global_similarity_flat_y_true(self):
    loss = losses.BatchSoftmaxWithGlobalSimilarity()

    y_pred = tf.constant([[1, 0], [0.0, 1.0], [1, 1], [0, 1], [1, 0], [1, 0]])
    y_true = tf.constant([1, 2])

    # Test both Keras-internal call and external call since behavior may be
    # slightly different due to type/shape conversion.
    loss_value = loss(y_true, y_pred)
    expected_loss_value = 1.3132616

    tf.debugging.assert_near(expected_loss_value, loss_value)

    loss_value = loss.call(y_true, y_pred)
    expected_loss_value = 1.3132616

    tf.debugging.assert_near(expected_loss_value, loss_value)

  def test_compare_batch_softmax_and_batch_softmax_with_global_similarity(self):
    loss_1 = losses.BatchSoftmaxWithGlobalSimilarity()
    loss_2 = losses.BatchSoftmax()

    y_pred_1 = tf.constant([[1, 0], [0.0, 1.0], [1, 1], [0, 1], [1, 0], [1, 0]])
    y_pred_2 = tf.constant([[1, 0], [0.0, 1.0], [0, 1], [1, 0]])
    y_true = tf.constant([[1], [2]])

    loss_value_1 = loss_1(y_true, y_pred_1)
    loss_value_2 = loss_2(y_true, y_pred_2)

    tf.debugging.assert_near(loss_value_1, loss_value_2)

  def test_batch_softmax_with_global_similarity_similarities(self):
    loss = losses.BatchSoftmaxWithGlobalSimilarity(expect_embeddings=False)

    y_pred = tf.constant([[0.7071067, 0.0, 1.0, 1.0],
                          [0.7071067, 1.0, 0.0, 0.0]])
    y_true = tf.constant([[1], [2]])

    # Test both Keras-internal call and external call since behavior may be
    # slightly different due to type/shape conversion.
    loss_value = loss(y_true, y_pred)
    expected_loss_value = 1.3132616

    tf.debugging.assert_near(expected_loss_value, loss_value)

    loss_value = loss.call(y_true, y_pred)
    expected_loss_value = 1.3132616

    tf.debugging.assert_near(expected_loss_value, loss_value)

  def test_compare_batch_softmax_and_batch_softmax_with_global_similarity_similarities(self):  # pylint: disable=line-too-long
    loss_1 = losses.BatchSoftmaxWithGlobalSimilarity(expect_embeddings=False)
    loss_2 = losses.BatchSoftmax(expect_embeddings=False)

    y_pred_1 = tf.constant([[0.7071067, 0.0, 1.0, 1.0],
                            [0.7071067, 1.0, 0.0, 0.0]])
    y_pred_2 = tf.constant([[0.0, 1.0], [1.0, 0.0]])
    y_true = tf.constant([[1], [2]])

    loss_value_1 = loss_1(y_true, y_pred_1)
    loss_value_2 = loss_2(y_true, y_pred_2)

    tf.debugging.assert_near(loss_value_1, loss_value_2)

  def test_batch_softmax_with_global_similarity_context_label_spreadout_no_embeddings(self):  # pylint: disable=line-too-long
    with self.assertRaises(ValueError):
      losses.BatchSoftmaxWithGlobalSimilarity(
          expect_embeddings=False, spreadout_context_lambda=0.1)

    with self.assertRaises(ValueError):
      losses.BatchSoftmaxWithGlobalSimilarity(
          expect_embeddings=False, spreadout_label_lambda=0.1)

    with self.assertRaises(ValueError):
      losses.BatchSoftmaxWithGlobalSimilarity(
          expect_embeddings=False,
          spreadout_context_lambda=0.1,
          spreadout_label_lambda=0.1)

  def test_batch_softmax_with_global_similarity_dot_product(self):
    loss = losses.BatchSoftmaxWithGlobalSimilarity(normalization_fn=None)

    y_pred = tf.constant([[1.0, 0.0], [0, 1], [0.7071067, 0.7071067],
                          [0.0, 1.0], [1, 0], [1.0, 0.0]])
    y_true = tf.constant([[1], [2]])

    # Test both Keras-internal call and external call since behavior may be
    # slightly different due to type/shape conversion.
    loss_value = loss(y_true, y_pred)
    expected_loss_value = 1.3132616

    tf.debugging.assert_near(expected_loss_value, loss_value)

    loss_value = loss.call(y_true, y_pred)
    expected_loss_value = 1.3132616

    tf.debugging.assert_near(expected_loss_value, loss_value)

  def test_compare_batch_softmax_and_batch_softmax_with_global_similarity_dot_product(self):  # pylint: disable=line-too-long
    loss_1 = losses.BatchSoftmaxWithGlobalSimilarity(normalization_fn=None)
    loss_2 = losses.BatchSoftmax(normalization_fn=None)

    y_pred_1 = tf.constant([[1.0, 0.0], [0, 1], [0.7071067, 0.7071067],
                            [0.0, 1.0], [1, 0], [1, 0]])
    y_pred_2 = tf.constant([[1.0, 0.0], [0, 1], [0.0, 1.0], [1, 0]])
    y_true = tf.constant([[1], [2]])

    loss_value_1 = loss_1(y_true, y_pred_1)
    loss_value_2 = loss_2(y_true, y_pred_2)

    tf.debugging.assert_near(loss_value_1, loss_value_2)

  def test_batch_softmax_with_global_similarity_spreadout(self):
    loss = losses.BatchSoftmaxWithGlobalSimilarity(
        spreadout_context_lambda=0.1,
        spreadout_label_lambda=0.2,
        spreadout_cross_lambda=0.3)

    y_pred = tf.constant([[1, 0], [0.0, 1.0], [1, 1], [0, 1], [1, 0], [1, 0]])
    y_true = tf.constant([[1], [2]])

    # Test both Keras-internal call and external call since behavior may be
    # slightly different due to type/shape conversion.
    loss_value = loss(y_true, y_pred)
    expected_loss_value = 1.7375257

    tf.debugging.assert_near(expected_loss_value, loss_value)

    loss_value = loss.call(y_true, y_pred)
    expected_loss_value = 1.7375257

    tf.debugging.assert_near(expected_loss_value, loss_value)

  def test_compare_batch_softmax_and_batch_softmax_with_global_similarity_spreadout(self):  # pylint: disable=line-too-long
    loss_1 = losses.BatchSoftmaxWithGlobalSimilarity(
        spreadout_context_lambda=0.1,
        spreadout_label_lambda=0.2,
        spreadout_cross_lambda=0.3)
    loss_2 = losses.BatchSoftmax(
        spreadout_context_lambda=0.1,
        spreadout_label_lambda=0.2,
        spreadout_cross_lambda=0.3)

    y_pred_1 = tf.constant([[1, 0], [0.0, 1.0], [1, 1], [0, 1], [1, 0], [1, 0]])
    y_pred_2 = tf.constant([[1, 0], [0.0, 1.0], [0, 1], [1, 0]])
    y_true = tf.constant([[1], [2]])

    loss_value_1 = loss_1(y_true, y_pred_1)
    loss_value_2 = loss_2(y_true, y_pred_2)

    tf.debugging.assert_near(loss_value_1, loss_value_2)

  def test_batch_softmax_with_global_similarity_spreadout_dot_product(self):
    loss = losses.BatchSoftmaxWithGlobalSimilarity(
        spreadout_context_lambda=0.1,
        spreadout_label_lambda=0.2,
        spreadout_cross_lambda=0.3,
        normalization_fn=None)

    y_pred = tf.constant([[1.0, 0.0], [0, 1], [0.7071067, 0.7071067],
                          [0.0, 1.0], [1, 0], [1.0, 0.0]])
    y_true = tf.constant([[1], [2]])

    # Test both Keras-internal call and external call since behavior may be
    # slightly different due to type/shape conversion.
    loss_value = loss(y_true, y_pred)
    expected_loss_value = 1.7375257

    tf.debugging.assert_near(expected_loss_value, loss_value)

    loss_value = loss.call(y_true, y_pred)
    expected_loss_value = 1.7375257

    tf.debugging.assert_near(expected_loss_value, loss_value)

  def test_compare_batch_softmax_and_batch_softmax_with_global_similarity_spreadout_dot_product(self):  # pylint: disable=line-too-long
    loss_1 = losses.BatchSoftmaxWithGlobalSimilarity(
        spreadout_context_lambda=0.1,
        spreadout_label_lambda=0.2,
        spreadout_cross_lambda=0.3,
        normalization_fn=None)
    loss_2 = losses.BatchSoftmax(
        spreadout_context_lambda=0.1,
        spreadout_label_lambda=0.2,
        spreadout_cross_lambda=0.3,
        normalization_fn=None)

    y_pred_1 = tf.constant([[1.0, 0.0], [0, 1], [0.7071067, 0.7071067],
                            [0.0, 1.0], [1, 0], [1, 0]])
    y_pred_2 = tf.constant([[1.0, 0.0], [0, 1], [0.0, 1.0], [1, 0]])
    y_true = tf.constant([[1], [2]])

    loss_value_1 = loss_1(y_true, y_pred_1)
    loss_value_2 = loss_2(y_true, y_pred_2)

    tf.debugging.assert_near(loss_value_1, loss_value_2)

  def test_batch_softmax_with_global_similarity_get_config(self):
    loss = losses.BatchSoftmaxWithGlobalSimilarity()

    config = loss.get_config()
    expected_config = {
        'normalization_fn': utils.l2_normalize_fn,
        'expect_embeddings': True,
        'spreadout_context_lambda': 0.0,
        'spreadout_label_lambda': 0.0,
        'spreadout_cross_lambda': 0.0,
        'name': None,
        'reduction': 'auto',
    }
    self.assertEqual(config, expected_config)

  def test_batch_softmax_with_global_similarity_get_config_keyword_args(self):
    loss = losses.BatchSoftmaxWithGlobalSimilarity(
        normalization_fn=None,
        expect_embeddings=False,
        spreadout_context_lambda=0.0,
        spreadout_label_lambda=0.0,
        spreadout_cross_lambda=0.1,
        name='my_loss')

    config = loss.get_config()
    expected_config = {
        'normalization_fn': None,
        'expect_embeddings': False,
        'spreadout_context_lambda': 0.0,
        'spreadout_label_lambda': 0.0,
        'spreadout_cross_lambda': 0.1,
        'name': 'my_loss',
        'reduction': 'auto'
    }
    self.assertEqual(config, expected_config)

  def test_global_softmax(self):
    loss = losses.GlobalSoftmax()

    y_pred = tf.constant([[1, 0], [0.0, 1.0], [1, 1], [0, 1], [1, 0], [1, 0]])
    y_true = tf.constant([[1], [2]])

    # Test both Keras-internal call and external call since behavior may be
    # slightly different due to type/shape conversion.
    loss_value = loss(y_true, y_pred)
    expected_loss_value = 2.0224552

    tf.debugging.assert_near(expected_loss_value, loss_value)

    loss_value = loss.call(y_true, y_pred)
    expected_loss_value = 2.0224552

    tf.debugging.assert_near(expected_loss_value, loss_value)

  def test_global_softmax_similarities(self):
    loss = losses.GlobalSoftmax(expect_embeddings=False)

    y_pred = tf.constant([[0.7071067, 0.0, 1.0, 1.0],
                          [0.7071067, 1.0, 0.0, 0.0]])
    y_true = tf.constant([[1], [2]])

    # Test both Keras-internal call and external call since behavior may be
    # slightly different due to type/shape conversion.
    loss_value = loss(y_true, y_pred)
    expected_loss_value = 2.0224552

    tf.debugging.assert_near(expected_loss_value, loss_value)

    loss_value = loss.call(y_true, y_pred)
    expected_loss_value = 2.0224552

    tf.debugging.assert_near(expected_loss_value, loss_value)

  def test_global_softmax_context_label_spreadout_no_embeddings(self):
    with self.assertRaises(ValueError):
      losses.GlobalSoftmax(
          expect_embeddings=False, spreadout_context_lambda=0.1)

    with self.assertRaises(ValueError):
      losses.GlobalSoftmax(expect_embeddings=False, spreadout_label_lambda=0.1)

    with self.assertRaises(ValueError):
      losses.GlobalSoftmax(
          expect_embeddings=False,
          spreadout_context_lambda=0.1,
          spreadout_label_lambda=0.1)

  def test_global_softmax_dot_product(self):
    loss = losses.GlobalSoftmax(normalization_fn=None)

    y_pred = tf.constant([[1.0, 0.0], [0, 1], [0.7071067, 0.7071067],
                          [0.0, 1.0], [1, 0], [1.0, 0.0]])
    y_true = tf.constant([[1], [2]])

    # Test both Keras-internal call and external call since behavior may be
    # slightly different due to type/shape conversion.
    loss_value = loss(y_true, y_pred)
    expected_loss_value = 2.0224552

    tf.debugging.assert_near(expected_loss_value, loss_value)

    loss_value = loss.call(y_true, y_pred)
    expected_loss_value = 2.0224552

    tf.debugging.assert_near(expected_loss_value, loss_value)

  def test_global_softmax_spreadout(self):
    loss = losses.GlobalSoftmax(
        spreadout_context_lambda=0.1,
        spreadout_label_lambda=0.2,
        spreadout_cross_lambda=0.3)

    y_pred = tf.constant([[1, 0], [0.0, 1.0], [1, 1], [0, 1], [1, 0], [1, 0]])
    y_true = tf.constant([[1], [2]])

    # Test both Keras-internal call and external call since behavior may be
    # slightly different due to type/shape conversion.
    loss_value = loss(y_true, y_pred)
    expected_loss_value = 3.0696688

    tf.debugging.assert_near(expected_loss_value, loss_value)

    loss_value = loss.call(y_true, y_pred)
    expected_loss_value = 3.0696688

    tf.debugging.assert_near(expected_loss_value, loss_value)

  def test_global_softmax_spreadout_dot_product(self):
    loss = losses.GlobalSoftmax(
        spreadout_context_lambda=0.1,
        spreadout_label_lambda=0.2,
        spreadout_cross_lambda=0.3,
        normalization_fn=None)

    y_pred = tf.constant([[1.0, 0.0], [0, 1], [0.7071067, 0.7071067],
                          [0.0, 1.0], [1, 0], [1.0, 0.0]])
    y_true = tf.constant([[1], [2]])

    # Test both Keras-internal call and external call since behavior may be
    # slightly different due to type/shape conversion.
    loss_value = loss(y_true, y_pred)
    expected_loss_value = 3.0696688

    tf.debugging.assert_near(expected_loss_value, loss_value)

    loss_value = loss.call(y_true, y_pred)
    expected_loss_value = 3.0696688

    tf.debugging.assert_near(expected_loss_value, loss_value)

  def test_global_softmax_get_config(self):
    loss = losses.GlobalSoftmax()

    config = loss.get_config()
    expected_config = {
        'normalization_fn': utils.l2_normalize_fn,
        'expect_embeddings': True,
        'spreadout_context_lambda': 0.0,
        'spreadout_label_lambda': 0.0,
        'spreadout_cross_lambda': 0.0,
        'name': None,
        'reduction': 'auto',
    }
    self.assertEqual(config, expected_config)

  def test_global_softmax_get_config_keyword_args(self):
    loss = losses.GlobalSoftmax(
        normalization_fn=None,
        expect_embeddings=False,
        spreadout_context_lambda=0.0,
        spreadout_label_lambda=0.0,
        spreadout_cross_lambda=0.1,
        name='my_loss')

    config = loss.get_config()
    expected_config = {
        'normalization_fn': None,
        'expect_embeddings': False,
        'spreadout_context_lambda': 0.0,
        'spreadout_label_lambda': 0.0,
        'spreadout_cross_lambda': 0.1,
        'name': 'my_loss',
        'reduction': 'auto'
    }
    self.assertEqual(config, expected_config)

  def test_hinge(self):
    loss = losses.Hinge()

    y_pred = tf.constant([[1, 0], [0.0, 1.0], [1, 1], [0, 1], [1, 0], [1, 0]])
    y_true = tf.constant([1.0, 1.0, 1.0])

    # Test both Keras-internal call and external call since behavior may be
    # slightly different due to type/shape conversion.
    loss_value = loss(y_true, y_pred)
    expected_loss_value = 0.552403

    tf.debugging.assert_near(expected_loss_value, loss_value)

    loss_value = loss.call(y_true, y_pred)
    expected_loss_value = 0.552403

    tf.debugging.assert_near(expected_loss_value, loss_value)

  def test_hinge_with_global_similarity(self):
    loss = losses.Hinge(use_global_similarity=True)

    y_pred = tf.constant([[1, 0], [0.0, 1.0], [1, 1], [0, 1], [1, 0], [1, 1],
                          [1, 0]])
    y_true = tf.constant([[0], [1], [3]])

    # Test both Keras-internal call and external call since behavior may be
    # slightly different due to type/shape conversion.
    loss_value = loss(y_true, y_pred)
    expected_loss_value = 0.552403

    tf.debugging.assert_near(expected_loss_value, loss_value)

    loss_value = loss.call(y_true, y_pred)
    expected_loss_value = 0.552403

    tf.debugging.assert_near(expected_loss_value, loss_value)

  def test_compare_hinge_and_hinge_with_global_similarity(self):
    loss_1 = losses.Hinge()
    loss_2 = losses.Hinge(use_global_similarity=True)

    y_pred_1 = tf.constant([[1, 0], [0.0, 1.0], [1, 1], [0, 1], [1, 0], [1, 0]])
    y_true_1 = tf.constant([1.0, 1.0, 1.0])

    y_pred_2 = tf.constant([[1, 0], [0.0, 1.0], [1, 1], [0, 1], [1, 0], [1, 1],
                            [1, 0]])
    y_true_2 = tf.constant([[0], [1], [3]])

    loss_value_1 = loss_1(y_true_1, y_pred_1)
    loss_value_2 = loss_2(y_true_2, y_pred_2)

    tf.debugging.assert_near(loss_value_1, loss_value_2)

  def test_hinge_similarities(self):
    loss = losses.Hinge(expect_embeddings=False)

    y_pred = tf.constant([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0],
                          [0.7071067, 0.7071067, 0.7071067]])
    y_true = tf.constant([1.0, 1.0, 1.0])

    # Test both Keras-internal call and external call since behavior may be
    # slightly different due to type/shape conversion.
    loss_value = loss(y_true, y_pred)
    expected_loss_value = 0.552403

    tf.debugging.assert_near(expected_loss_value, loss_value)

    loss_value = loss.call(y_true, y_pred)
    expected_loss_value = 0.552403

    tf.debugging.assert_near(expected_loss_value, loss_value)

  def test_hinge_similarities_with_global_similarity(self):
    loss = losses.Hinge(expect_embeddings=False, use_global_similarity=True)

    y_pred = tf.constant([[0.0, 1.0, 0.7071067, 1.0],
                          [1.0, 0.0, 0.7071067, 0.0],
                          [0.7071067, 0.7071067, 1.0, 0.7071067]])
    y_true = tf.constant([[0], [1], [3]])

    # Test both Keras-internal call and external call since behavior may be
    # slightly different due to type/shape conversion.
    loss_value = loss(y_true, y_pred)
    expected_loss_value = 0.552403

    tf.debugging.assert_near(expected_loss_value, loss_value)

    loss_value = loss.call(y_true, y_pred)
    expected_loss_value = 0.552403

    tf.debugging.assert_near(expected_loss_value, loss_value)

  def test_Hinge_context_label_spreadout_no_embeddings(self):
    with self.assertRaises(ValueError):
      losses.Hinge(expect_embeddings=False, spreadout_context_lambda=0.1)

    with self.assertRaises(ValueError):
      losses.Hinge(expect_embeddings=False, spreadout_label_lambda=0.1)

    with self.assertRaises(ValueError):
      losses.Hinge(
          expect_embeddings=False,
          spreadout_context_lambda=0.1,
          spreadout_label_lambda=0.1)

  def test_hinge_dot_product(self):
    loss = losses.Hinge(normalization_fn=None)

    y_pred = tf.constant([[1.0, 0.0], [0, 1], [0.7071067, 0.7071067],
                          [0.0, 1.0], [1, 0], [1.0, 0.0]])
    y_true = tf.constant([1.0, 1.0, 1.0])

    # Test both Keras-internal call and external call since behavior may be
    # slightly different due to type/shape conversion.
    loss_value = loss(y_true, y_pred)
    expected_loss_value = 0.552403

    tf.debugging.assert_near(expected_loss_value, loss_value)

    loss_value = loss.call(y_true, y_pred)
    expected_loss_value = 0.552403

    tf.debugging.assert_near(expected_loss_value, loss_value)

  def test_hinge_spreadout(self):
    loss = losses.Hinge(
        spreadout_context_lambda=0.1,
        spreadout_label_lambda=0.2,
        spreadout_cross_lambda=0.3)

    y_pred = tf.constant([[1, 0], [0.0, 1.0], [1, 1], [0, 1], [1, 0], [1, 0]])
    y_true = tf.constant([1.0, 1.0, 1.0])

    # Test both Keras-internal call and external call since behavior may be
    # slightly different due to type/shape conversion.
    loss_value = loss(y_true, y_pred)
    expected_loss_value = 1.576667

    tf.debugging.assert_near(expected_loss_value, loss_value)

    loss_value = loss.call(y_true, y_pred)
    expected_loss_value = 1.576667

    tf.debugging.assert_near(expected_loss_value, loss_value)

  def test_hinge_spreadout_with_global_similarity(self):
    loss = losses.Hinge(
        spreadout_context_lambda=0.1,
        spreadout_label_lambda=0.2,
        spreadout_cross_lambda=0.3,
        use_global_similarity=True)

    y_pred = tf.constant([[1, 0], [0.0, 1.0], [1, 1], [0, 1], [1, 0], [0, 0],
                          [1, 0]])
    y_true = tf.constant([[0], [1], [3]])

    # Test both Keras-internal call and external call since behavior may be
    # slightly different due to type/shape conversion.
    loss_value = loss(y_true, y_pred)
    expected_loss_value = 1.576667

    tf.debugging.assert_near(expected_loss_value, loss_value)

    loss_value = loss.call(y_true, y_pred)
    expected_loss_value = 1.576667

    tf.debugging.assert_near(expected_loss_value, loss_value)

  def test_hinge_spreadout_dot_product(self):
    loss = losses.Hinge(
        spreadout_context_lambda=0.1,
        spreadout_label_lambda=0.2,
        spreadout_cross_lambda=0.3,
        normalization_fn=None)

    y_pred = tf.constant([[1.0, 0.0], [0, 1], [0.7071067, 0.7071067],
                          [0.0, 1.0], [1, 0], [1.0, 0.0]])
    y_true = tf.constant([1.0, 1.0, 1.0])

    # Test both Keras-internal call and external call since behavior may be
    # slightly different due to type/shape conversion.
    loss_value = loss(y_true, y_pred)
    expected_loss_value = 1.576667

    tf.debugging.assert_near(expected_loss_value, loss_value)

    loss_value = loss.call(y_true, y_pred)
    expected_loss_value = 1.576667

    tf.debugging.assert_near(expected_loss_value, loss_value)

  def test_hinge_get_config(self):
    loss = losses.Hinge()

    config = loss.get_config()
    expected_config = {
        'normalization_fn': utils.l2_normalize_fn,
        'expect_embeddings': True,
        'spreadout_context_lambda': 0.0,
        'spreadout_label_lambda': 0.0,
        'spreadout_cross_lambda': 0.0,
        'name': None,
        'reduction': 'auto',
    }
    self.assertEqual(config, expected_config)

  def test_hinge_get_config_keyword_args(self):
    loss = losses.Hinge(
        normalization_fn=None,
        expect_embeddings=False,
        spreadout_context_lambda=0.0,
        spreadout_label_lambda=0.0,
        spreadout_cross_lambda=0.1,
        name='my_loss')

    config = loss.get_config()
    expected_config = {
        'normalization_fn': None,
        'expect_embeddings': False,
        'spreadout_context_lambda': 0.0,
        'spreadout_label_lambda': 0.0,
        'spreadout_cross_lambda': 0.1,
        'name': 'my_loss',
        'reduction': 'auto'
    }
    self.assertEqual(config, expected_config)

  def test_update_loss_with_spreadout_loss(self):
    loss = 0.1
    context_embedding = tf.constant([[1, 0], [0.0, 1.0]])
    label_embedding = tf.constant([[1.0, 1.0], [0, 1], [1, 0], [1, 0]])
    similarity = tf.constant([[1, 0, 1, 1], [1, 1, 0, 0.]])
    y_true = tf.constant([[1], [2]])

    spreadout_context_lambda = 0.1
    spreadout_label_lambda = 0.2
    spreadout_cross_lambda = 0.3

    loss_value = losses._update_loss_with_spreadout_loss(
        loss=loss,
        context_embedding=context_embedding,
        label_embedding=label_embedding,
        similarities=similarity,
        spreadout_context_lambda=spreadout_context_lambda,
        spreadout_label_lambda=spreadout_label_lambda,
        spreadout_cross_lambda=spreadout_cross_lambda,
        label_indices=tf.transpose(y_true)[0])

    expected_loss_value = 1.3365058

    tf.debugging.assert_near(expected_loss_value, loss_value)


if __name__ == '__main__':
  absltest.main()
