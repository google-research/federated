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
from absl.testing import flagsaver
import numpy as np
import tensorflow as tf

from personalization_benchmark.cross_device.algorithms import knn_per


class KnnPerTest(tf.test.TestCase):

  def test_embedding_model_returns_correct_embedding_dimension(self):
    expected_embedding_dimension = {
        'emnist': 512,
        'stackoverflow': 670,
        'landmark': 1280,
        'ted_multi': 96
    }
    for dataset_name, expected_dim in expected_embedding_dimension.items():
      model_fn, _, _, _, _ = knn_per.create_model_and_data(
          dataset_name, use_synthetic_data=True)
      keras_model = model_fn()._keras_model
      actual_dim = keras_model.layers[knn_per.embedding_layer_index(
          dataset_name)].output.shape[-1]
      self.assertEqual(expected_dim, actual_dim)

  @flagsaver.flagsaver(num_neighbors=2)
  def test_compute_knn_softmax_returns_correct_result(self):
    train_embeddings = np.array([[0.1], [0.1], [0.4], [0.4]])
    train_labels = np.array([[0], [1], [2], [0]])
    eval_embeddings = np.array([[0.0], [0.5]])
    num_labels = 3
    knn_softmax = knn_per._compute_knn_softmax(train_embeddings, train_labels,
                                               eval_embeddings, num_labels)
    expected_knn_softmax = np.array([[0.5, 0.5, 0.0], [0.5, 0.0, 0.5]])
    self.assertAllEqual(expected_knn_softmax, knn_softmax)

  @flagsaver.flagsaver(
      dataset_name='emnist',
      num_neighbors=2,
      valid_clients_per_evaluation=2,
      test_clients_per_evaluation=2,
      use_synthetic_data=True)
  def test_emnist_executes(self):
    knn_per.main([])

  @flagsaver.flagsaver(
      dataset_name='stackoverflow',
      num_neighbors=1,
      valid_clients_per_evaluation=1,
      test_clients_per_evaluation=1,
      use_synthetic_data=True)
  def test_stackoverflow_executes(self):
    knn_per.main([])

  @flagsaver.flagsaver(
      dataset_name='landmark',
      num_neighbors=1,
      valid_clients_per_evaluation=1,
      test_clients_per_evaluation=1,
      use_synthetic_data=True)
  def test_landmark_executes(self):
    knn_per.main([])

  @flagsaver.flagsaver(
      dataset_name='ted_multi',
      num_neighbors=1,
      valid_clients_per_evaluation=1,
      test_clients_per_evaluation=1,
      use_synthetic_data=True)
  def test_tedmulti_executes(self):
    knn_per.main([])


if __name__ == '__main__':
  tf.test.main()
