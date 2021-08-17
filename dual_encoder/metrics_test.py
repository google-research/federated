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

from dual_encoder import metrics
from dual_encoder import model_utils as utils


class MetricsTest(absltest.TestCase):

  def test_batch_recall(self):
    metric = metrics.BatchRecall(recall_k=2)

    y_pred = tf.constant(
        [[1, 2, 3],
         [4.0, 5.0, 6.0],
         [1, 1, 1],
         [1, 1, 1],
         [1, 1, 2],
         [1, 2, 3],
         [4.0, 5.0, 6.0],
         [1, 1, 1],
         [1, 1, 1],
         [-1, -2, -3]]
    )
    y_true = tf.constant([1.0, 1.0, 1.0, 1.0, 1.0])

    metric_value = metric(y_true, y_pred)
    expected_metric_value = 0.8

    tf.debugging.assert_near(expected_metric_value, metric_value)

  def test_batch_recall_similarities(self):
    metric = metrics.BatchRecall(recall_k=2, expect_embeddings=False)

    y_pred = tf.constant(
        [[0.9999999, 0.97463185, 0.92582005, 0.92582005, -0.9999999],
         [0.97463185, 1.0000001, 0.98692757, 0.98692757, -0.97463185],
         [0.92582005, 0.98692757, 0.99999994, 0.99999994, -0.92582005],
         [0.92582005, 0.98692757, 0.99999994, 0.99999994, -0.92582005],
         [0.98198044, 0.9770084, 0.942809, 0.942809, -0.98198044]])
    y_true = tf.constant([1.0, 1.0, 1.0, 1.0, 1.0])

    metric_value = metric(y_true, y_pred)
    expected_metric_value = 0.8

    tf.debugging.assert_near(expected_metric_value, metric_value)

  def test_batch_recall_y_true_2d(self):
    metric = metrics.BatchRecall(recall_k=2)

    y_pred = tf.constant(
        [[1, 2, 3],
         [4.0, 5.0, 6.0],
         [1, 1, 1],
         [1, 1, 1],
         [1, 1, 2],
         [1, 2, 3],
         [4.0, 5.0, 6.0],
         [1, 1, 1],
         [1, 1, 1],
         [-1, -2, -3]]
    )
    y_true = tf.constant([[1.0], [1.0], [1.0], [1.0], [1.0]])

    metric_value = metric(y_true, y_pred)
    expected_metric_value = 0.8

    tf.debugging.assert_near(expected_metric_value, metric_value)

  def test_batch_recall_dot_product(self):
    metric = metrics.BatchRecall(recall_k=2, normalization_fn=None)

    y_pred = tf.constant(
        [[1, 2, 3],
         [4.0, 5.0, 6.0],
         [1, 1, 1],
         [1, 1, 1],
         [1, 1, 2],
         [1, 2, 3],
         [4.0, 5.0, 6.0],
         [1, 1, 1],
         [1, 1, 1],
         [-1, -2, -3]]
    )
    y_true = tf.constant([1.0, 1.0, 1.0, 1.0, 1.0])

    metric_value = metric(y_true, y_pred)
    expected_metric_value = 0.4

    tf.debugging.assert_near(expected_metric_value, metric_value)

  def test_batch_recall_get_config(self):
    metric = metrics.BatchRecall()

    config = metric.get_config()
    expected_config = {
        'normalization_fn': utils.l2_normalize_fn,
        'expect_embeddings': True,
        'recall_k': 10,
        'name': 'batch_recall',
        'dtype': 'float32'
    }
    self.assertEqual(config, expected_config)

  def test_batch_recall_get_config_keyword_args(self):
    metric = metrics.BatchRecall(recall_k=5,
                                 normalization_fn=None,
                                 expect_embeddings=False)

    config = metric.get_config()
    expected_config = {
        'normalization_fn': None,
        'expect_embeddings': False,
        'recall_k': 5,
        'name': 'batch_recall',
        'dtype': 'float32'
    }
    self.assertEqual(config, expected_config)

  def test_batch_recall_with_global_similarity(self):
    metric = metrics.BatchRecallWithGlobalSimilarity(recall_k=1)

    y_pred = tf.constant(
        [[1, 0],
         [1.0, 1.0],
         [1, 1],
         [0., 1.0],
         [0, 1],
         [1, 0]]
    )
    y_true = tf.constant([[0], [2]])

    metric_value = metric(y_true, y_pred)
    expected_metric_value = 0.5

    tf.debugging.assert_near(expected_metric_value, metric_value)

  def test_compare_batch_recall_and_batch_recall_with_global_similarity(self):
    metric_1 = metrics.BatchRecallWithGlobalSimilarity(recall_k=1)
    metric_2 = metrics.BatchRecall(recall_k=1)

    y_pred_1 = tf.constant(
        [[1, 0],
         [1.0, 1.0],
         [1, 1],
         [0., 1.0],
         [0, 1],
         [1, 0]]
    )
    y_pred_2 = tf.constant(
        [[1, 0],
         [1.0, 1.0],
         [1, 1],
         [0, 1]]
    )
    y_true = tf.constant([[0], [2]])

    metric_value_1 = metric_1(y_true, y_pred_1)
    metric_value_2 = metric_2(y_true, y_pred_2)

    tf.debugging.assert_near(metric_value_1, metric_value_2)

  def test_batch_recall_with_global_similarity_similarities(self):
    metric = metrics.BatchRecallWithGlobalSimilarity(
        recall_k=1, expect_embeddings=False)

    y_pred = tf.constant(
        [[0.7071067, 0.0, 0.0, 1.0],
         [1.0, 0.7071067, 0.7071067, 0.7071067]])
    y_true = tf.constant([[0], [2]])

    metric_value = metric(y_true, y_pred)
    expected_metric_value = 0.5

    tf.debugging.assert_near(expected_metric_value, metric_value)

  def test_compare_batch_recall_and_batch_recall_with_global_similarity_similarities(self):  # pylint: disable=line-too-long
    metric_1 = metrics.BatchRecallWithGlobalSimilarity(
        recall_k=1, expect_embeddings=False)
    metric_2 = metrics.BatchRecall(
        recall_k=1, expect_embeddings=False)

    y_pred_1 = tf.constant(
        [[0.7071067, 0.0, 0.0, 1.0],
         [1.0, 0.7071067, 0.7071067, 0.7071067]])
    y_pred_2 = tf.constant(
        [[0.7071067, 0.0],
         [1.0, 0.7071067]])
    y_true = tf.constant([[0], [2]])

    metric_value_1 = metric_1(y_true, y_pred_1)
    metric_value_2 = metric_2(y_true, y_pred_2)

    tf.debugging.assert_near(metric_value_1, metric_value_2)

  def test_batch_recall_with_global_similarity_dot_product(self):
    metric = metrics.BatchRecallWithGlobalSimilarity(
        recall_k=1, normalization_fn=None)

    y_pred = tf.constant(
        [[1.0, 0.0],
         [0.7071067, 0.7071067],
         [0.7071067, 0.7071067],
         [0.0, 1.0],
         [0, 1],
         [1.0, 0.0]]
    )
    y_true = tf.constant([[0], [2]])

    metric_value = metric(y_true, y_pred)
    expected_metric_value = 0.5

    tf.debugging.assert_near(expected_metric_value, metric_value)

  def test_compare_batch_recall_and_batch_recall_with_global_similarity_dot_product(self):  # pylint: disable=line-too-long
    metric_1 = metrics.BatchRecallWithGlobalSimilarity(
        recall_k=1, normalization_fn=None)
    metric_2 = metrics.BatchRecall(
        recall_k=1, normalization_fn=None)

    y_pred_1 = tf.constant(
        [[1.0, 0.0],
         [0.7071067, 0.7071067],
         [0.7071067, 0.7071067],
         [0.0, 1.0],
         [0, 1],
         [1.0, 0.0]]
    )
    y_pred_2 = tf.constant(
        [[1.0, 0.0],
         [0.7071067, 0.7071067],
         [0.7071067, 0.7071067],
         [0, 1]]
    )
    y_true = tf.constant([[0], [2]])

    metric_value_1 = metric_1(y_true, y_pred_1)
    metric_value_2 = metric_2(y_true, y_pred_2)

    tf.debugging.assert_near(metric_value_1, metric_value_2)

  def test_batch_recall_with_global_similarity_get_config(self):
    metric = metrics.BatchRecallWithGlobalSimilarity()

    config = metric.get_config()
    expected_config = {
        'normalization_fn': utils.l2_normalize_fn,
        'expect_embeddings': True,
        'recall_k': 10,
        'name': 'batch_recall_with_global_similarity',
        'dtype': 'float32'
    }
    self.assertEqual(config, expected_config)

  def test_batch_recall_with_global_similarity_get_config_keyword_args(self):
    metric = metrics.BatchRecallWithGlobalSimilarity(
        recall_k=5, normalization_fn=None, expect_embeddings=False)

    config = metric.get_config()
    expected_config = {
        'normalization_fn': None,
        'expect_embeddings': False,
        'recall_k': 5,
        'name': 'batch_recall_with_global_similarity',
        'dtype': 'float32'
    }
    self.assertEqual(config, expected_config)

  def test_global_recall(self):
    metric = metrics.GlobalRecall(recall_k=2)

    y_pred = tf.constant(
        [[1, 0],
         [0.0, 1.0],
         [1, 1],
         [0, 1],
         [1, 0],
         [1, 0]]
    )
    y_true = tf.constant([[2], [3]])

    metric_value = metric(y_true, y_pred)
    expected_metric_value = 0.5

    tf.debugging.assert_near(expected_metric_value, metric_value)

  def test_global_recall_similarities(self):
    metric = metrics.GlobalRecall(recall_k=2, expect_embeddings=False)

    y_pred = tf.constant(
        [[0.7071067, 0.0, 1.0, 1.0],
         [0.7071067, 1.0, 0.0, 0.0]])
    y_true = tf.constant([[2], [3]])

    metric_value = metric(y_true, y_pred)
    expected_metric_value = 0.5

    tf.debugging.assert_near(expected_metric_value, metric_value)

  def test_global_recall_dot_product(self):
    metric = metrics.GlobalRecall(recall_k=2, normalization_fn=None)

    y_pred = tf.constant(
        [[1.0, 0.0],
         [0, 1],
         [0.7071067, 0.7071067],
         [0.0, 1.0],
         [1, 0],
         [1.0, 0.0]]
    )
    y_true = tf.constant([[2], [3]])

    metric_value = metric(y_true, y_pred)
    expected_metric_value = 0.5

    tf.debugging.assert_near(expected_metric_value, metric_value)

  def test_global_recall_get_config(self):
    metric = metrics.GlobalRecall()

    config = metric.get_config()
    expected_config = {
        'normalization_fn': utils.l2_normalize_fn,
        'expect_embeddings': True,
        'recall_k': 10,
        'name': 'global_recall',
        'dtype': 'float32'
    }
    self.assertEqual(config, expected_config)

  def test_global_recall_get_config_keyword_args(self):
    metric = metrics.GlobalRecall(recall_k=5,
                                  normalization_fn=None,
                                  expect_embeddings=False)

    config = metric.get_config()
    expected_config = {
        'normalization_fn': None,
        'expect_embeddings': False,
        'recall_k': 5,
        'name': 'global_recall',
        'dtype': 'float32'
    }
    self.assertEqual(config, expected_config)

  def test_batch_mean_rank(self):
    metric = metrics.BatchMeanRank()

    y_pred = tf.constant(
        [[1, 2, 3],
         [4.0, 5.0, 6.0],
         [1, 1, 1],
         [1, 1, 1],
         [1, 1, 2],
         [1, 2, 3],
         [4.0, 5.0, 6.0],
         [1, 1, 1],
         [1, 1, 1],
         [-1, -2, -3]]
    )
    y_true = tf.constant([1.0, 1.0, 1.0, 1.0, 1.0])

    metric_value = metric(y_true, y_pred)
    expected_metric_value = 1.0

    tf.debugging.assert_near(expected_metric_value, metric_value)

  def test_batch_mean_rank_similarities(self):
    metric = metrics.BatchMeanRank(expect_embeddings=False)

    y_pred = tf.constant(
        [[0.9999999, 0.97463185, 0.92582005, 0.92582005, -0.9999999],
         [0.97463185, 1.0000001, 0.98692757, 0.98692757, -0.97463185],
         [0.92582005, 0.98692757, 0.99999994, 0.99999994, -0.92582005],
         [0.92582005, 0.98692757, 0.99999994, 0.99999994, -0.92582005],
         [0.98198044, 0.9770084, 0.942809, 0.942809, -0.98198044]])
    y_true = tf.constant([1.0, 1.0, 1.0, 1.0, 1.0])

    metric_value = metric(y_true, y_pred)
    expected_metric_value = 1.0

    tf.debugging.assert_near(expected_metric_value, metric_value)

  def test_batch_mean_rank_y_true_2d(self):
    metric = metrics.BatchMeanRank()

    y_pred = tf.constant(
        [[1, 2, 3],
         [4.0, 5.0, 6.0],
         [1, 1, 1],
         [1, 1, 1],
         [1, 1, 2],
         [1, 2, 3],
         [4.0, 5.0, 6.0],
         [1, 1, 1],
         [1, 1, 1],
         [-1, -2, -3]]
    )
    y_true = tf.constant([[1.0], [1.0], [1.0], [1.0], [1.0]])

    metric_value = metric(y_true, y_pred)
    expected_metric_value = 1.0

    tf.debugging.assert_near(expected_metric_value, metric_value)

  def test_batch_mean_rank_dot_product(self):
    metric = metrics.BatchMeanRank(normalization_fn=None)

    y_pred = tf.constant(
        [[1, 2, 3],
         [4.0, 5.0, 6.0],
         [1, 1, 1],
         [1, 1, 1],
         [1, 1, 2],
         [1, 2, 3],
         [4.0, 5.0, 6.0],
         [1, 1, 1],
         [1, 1, 1],
         [-1, -2, -3]]
    )
    y_true = tf.constant([1.0, 1.0, 1.0, 1.0, 1.0])

    metric_value = metric(y_true, y_pred)
    expected_metric_value = 2.0

    tf.debugging.assert_near(expected_metric_value, metric_value)

  def test_batch_mean_rank_get_config(self):
    metric = metrics.BatchMeanRank()

    config = metric.get_config()
    expected_config = {
        'normalization_fn': utils.l2_normalize_fn,
        'expect_embeddings': True,
        'name': 'batch_mean_rank',
        'dtype': 'float32'
    }
    self.assertEqual(config, expected_config)

  def test_batch_mean_rank_get_config_keyword_args(self):
    metric = metrics.BatchMeanRank(normalization_fn=None,
                                   expect_embeddings=False)

    config = metric.get_config()
    expected_config = {
        'normalization_fn': None,
        'expect_embeddings': False,
        'name': 'batch_mean_rank',
        'dtype': 'float32'
    }
    self.assertEqual(config, expected_config)

  def test_global_mean_rank(self):
    metric = metrics.GlobalMeanRank()

    y_pred = tf.constant(
        [[1, 0],
         [0.0, 1.0],
         [1, 1],
         [0, 1],
         [1, 0],
         [1, 0]]
    )
    y_true = tf.constant([[2], [3]])

    metric_value = metric(y_true, y_pred)
    expected_metric_value = 1.5

    tf.debugging.assert_near(expected_metric_value, metric_value)

  def test_global_mean_rank_similarities(self):
    metric = metrics.GlobalMeanRank(expect_embeddings=False)

    y_pred = tf.constant(
        [[0.7071067, 0.0, 1.0, 1.0],
         [0.7071067, 1.0, 0.0, 0.0]])
    y_true = tf.constant([[2], [3]])

    metric_value = metric(y_true, y_pred)
    expected_metric_value = 1.5

    tf.debugging.assert_near(expected_metric_value, metric_value)

  def test_global_mean_rank_dot_product(self):
    metric = metrics.GlobalMeanRank(normalization_fn=None)

    y_pred = tf.constant(
        [[1.0, 0.0],
         [0, 1],
         [0.7071067, 0.7071067],
         [0.0, 1.0],
         [1, 0],
         [1.0, 0.0]]
    )
    y_true = tf.constant([[2], [3]])

    metric_value = metric(y_true, y_pred)
    expected_metric_value = 1.5

    tf.debugging.assert_near(expected_metric_value, metric_value)

  def test_global_mean_rank_get_config(self):
    metric = metrics.GlobalMeanRank()

    config = metric.get_config()
    expected_config = {
        'normalization_fn': utils.l2_normalize_fn,
        'expect_embeddings': True,
        'name': 'global_mean_rank',
        'dtype': 'float32'
    }
    self.assertEqual(config, expected_config)

  def test_global_mean_rank_get_config_keyword_args(self):
    metric = metrics.GlobalMeanRank(normalization_fn=None,
                                    expect_embeddings=False)

    config = metric.get_config()
    expected_config = {
        'normalization_fn': None,
        'expect_embeddings': False,
        'name': 'global_mean_rank',
        'dtype': 'float32'
    }
    self.assertEqual(config, expected_config)

  def test_batch_similarities_norm(self):
    metric = metrics.BatchSimilaritiesNorm()

    y_pred = tf.constant(
        [[1, 2, 3],
         [4.0, 5.0, 6.0],
         [1, 1, 1],
         [1, 1, 1],
         [1, 1, 2],
         [1, 2, 3],
         [4.0, 5.0, 6.0],
         [1, 1, 1],
         [1, 1, 1],
         [-1, -2, -3]]
    )
    y_true = tf.constant([1.0, 1.0, 1.0, 1.0, 1.0])

    metric_value = metric(y_true, y_pred)
    expected_metric_value = 4.311066

    tf.debugging.assert_near(expected_metric_value, metric_value)

  def test_batch_similarities_norm_similarities(self):
    metric = metrics.BatchSimilaritiesNorm(expect_embeddings=False)

    y_pred = tf.constant(
        [[0.9999999, 0.97463185, 0.92582005, 0.92582005, -0.9999999],
         [0.97463185, 1.0000001, 0.98692757, 0.98692757, -0.97463185],
         [0.92582005, 0.98692757, 0.99999994, 0.99999994, -0.92582005],
         [0.92582005, 0.98692757, 0.99999994, 0.99999994, -0.92582005],
         [0.98198044, 0.9770084, 0.942809, 0.942809, -0.98198044]])
    y_true = tf.constant([1.0, 1.0, 1.0, 1.0, 1.0])

    metric_value = metric(y_true, y_pred)
    expected_metric_value = 4.311066

    tf.debugging.assert_near(expected_metric_value, metric_value)

  def test_batch_similarities_norm_y_true_2d(self):
    metric = metrics.BatchSimilaritiesNorm()

    y_pred = tf.constant(
        [[1, 2, 3],
         [4.0, 5.0, 6.0],
         [1, 1, 1],
         [1, 1, 1],
         [1, 1, 2],
         [1, 2, 3],
         [4.0, 5.0, 6.0],
         [1, 1, 1],
         [1, 1, 1],
         [-1, -2, -3]]
    )
    y_true = tf.constant([[1.0], [1.0], [1.0], [1.0], [1.0]])

    metric_value = metric(y_true, y_pred)
    expected_metric_value = 4.311066

    tf.debugging.assert_near(expected_metric_value, metric_value)

  def test_batch_similarities_norm_dot_product(self):
    metric = metrics.BatchSimilaritiesNorm(normalization_fn=None)

    y_pred = tf.constant(
        [[1, 2, 3],
         [4.0, 5.0, 6.0],
         [1, 1, 1],
         [1, 1, 1],
         [1, 1, 2],
         [1, 2, 3],
         [4.0, 5.0, 6.0],
         [1, 1, 1],
         [1, 1, 1],
         [-1, -2, -3]]
    )
    y_true = tf.constant([1.0, 1.0, 1.0, 1.0, 1.0])

    metric_value = metric(y_true, y_pred)
    expected_metric_value = 70.398865

    tf.debugging.assert_near(expected_metric_value, metric_value)

  def test_batch_similarities_norm_get_config(self):
    metric = metrics.BatchSimilaritiesNorm()

    config = metric.get_config()
    expected_config = {
        'normalization_fn': utils.l2_normalize_fn,
        'expect_embeddings': True,
        'name': 'batch_similarities_norm',
        'dtype': 'float32'
    }
    self.assertEqual(config, expected_config)

  def test_batch_similarities_norm_get_config_keyword_args(self):
    metric = metrics.BatchSimilaritiesNorm(normalization_fn=None,
                                           expect_embeddings=False)

    config = metric.get_config()
    expected_config = {
        'normalization_fn': None,
        'expect_embeddings': False,
        'name': 'batch_similarities_norm',
        'dtype': 'float32'
    }
    self.assertEqual(config, expected_config)

if __name__ == '__main__':
  absltest.main()
