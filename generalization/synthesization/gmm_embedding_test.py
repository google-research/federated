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
"""Tests for gmm_embedding."""

import collections

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from generalization.synthesization import gmm_embedding

tfd = tfp.distributions


def _build_fake_dataset(num_elems=1000) -> tf.data.Dataset:
  rng = np.random.default_rng(1)
  return tf.data.Dataset.from_tensor_slices(
      collections.OrderedDict(
          image=np.array(
              rng.integers(0, 256, (num_elems, 4, 4, 3)), dtype=np.uint8),
          label=np.array(rng.integers(0, 4, (num_elems,)), dtype=np.int64)))


def _build_fake_pretrained_model() -> tf.keras.Model:
  return tf.keras.models.Sequential([tf.keras.layers.Flatten()])


def _fake_tril(dim):
  a = tf.random.normal((dim, dim))
  return tf.linalg.cholesky(
      tf.matmul(a, a, transpose_a=True) + tf.eye(dim) * 1e-6)


def _pairwise_kl_divergence_between_multivariate_normal_tril_element_wise(
    means_1: tf.Tensor, trils_1: tf.Tensor, means_2: tf.Tensor,
    trils_2: tf.Tensor) -> tf.Tensor:
  """Compute pairwise KL divergence matrix element-wise."""
  pairwise_matrix = tf.Variable(
      tf.zeros((means_1.shape[0], means_2.shape[0]), dtype=means_1.dtype))

  for i in range(means_1.shape[0]):
    for j in range(means_2.shape[0]):
      pairwise_matrix[i, j].assign(
          tfd.kl_divergence(
              tfd.MultivariateNormalTriL(means_1[i, :], trils_1[i, :, :]),
              tfd.MultivariateNormalTriL(means_2[j, :], trils_2[j, :, :])))

  return tf.convert_to_tensor(pairwise_matrix)


class PairwiseKLTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(((f'batch_size = {batch_size}', batch_size)
                                   for batch_size in [1, 2, 3, 5, 7, 11, 13]))
  def test_pairwise_kl_divergence_in_batch_returns_the_same_result_as_element_wise(
      self, batch_size):

    num_dist_1 = 7
    num_dist_2 = 11
    dim = 13

    means_1 = tf.random.normal((num_dist_1, dim))
    means_2 = tf.random.normal((num_dist_2, dim))
    trils_1 = tf.stack([_fake_tril(dim) for _ in range(num_dist_1)])
    trils_2 = tf.stack([_fake_tril(dim) for _ in range(num_dist_2)])

    obtained_pairwise_matrix = gmm_embedding.pairwise_kl_divergence_between_multivariate_normal_tril_in_batch(
        means_1, trils_1, means_2, trils_2, batch_size=batch_size)

    expected_pairwise_matrix = _pairwise_kl_divergence_between_multivariate_normal_tril_element_wise(
        means_1, trils_1, means_2, trils_2)

    self.assertAllClose(expected_pairwise_matrix, obtained_pairwise_matrix)


class GmmEmbeddingTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('case 1', None, True),
      ('case 2', 4, True),
      ('case 3', None, False),
      ('case 4', 4, False),
  )
  def test_gmm_embedding(self, pca_components, use_progressive_matching):
    dataset = _build_fake_dataset()
    num_clients = 4
    cd = gmm_embedding.synthesize_by_gmm_over_pretrained_embedding(
        dataset,
        pretrained_model_builder=_build_fake_pretrained_model,
        num_clients=num_clients,
        pca_components=pca_components,
        use_progressive_matching=use_progressive_matching)

    self.assertCountEqual(cd.client_ids, list(map(str, range(num_clients))))

    for client_id in cd.client_ids:
      client_ds = cd.create_tf_dataset_for_client(client_id)
      self.assertEqual(client_ds.element_spec, dataset.element_spec)

  @parameterized.named_parameters(
      ('case 1', None, True),
      ('case 2', 4, True),
      ('case 3', None, False),
      ('case 4', 4, False),
  )
  def test_gmm_embedding_use_seed(self, pca_components,
                                  use_progressive_matching):
    dataset = _build_fake_dataset()
    num_clients = 4

    cd1 = gmm_embedding.synthesize_by_gmm_over_pretrained_embedding(
        dataset,
        pretrained_model_builder=_build_fake_pretrained_model,
        num_clients=num_clients,
        pca_components=pca_components,
        use_progressive_matching=use_progressive_matching,
        seed=1)

    cd2 = gmm_embedding.synthesize_by_gmm_over_pretrained_embedding(
        dataset,
        pretrained_model_builder=_build_fake_pretrained_model,
        num_clients=num_clients,
        pca_components=pca_components,
        use_progressive_matching=use_progressive_matching,
        seed=1)

    self.assertCountEqual(cd1.client_ids, list(map(str, range(num_clients))))
    self.assertCountEqual(cd2.client_ids, list(map(str, range(num_clients))))

    for client_id in cd1.client_ids:
      client_ds1_list = list(cd1.create_tf_dataset_for_client(client_id))
      client_ds2_list = list(cd2.create_tf_dataset_for_client(client_id))

      self.assertEqual(len(client_ds1_list), len(client_ds2_list))

      for elem1, elem2 in zip(client_ds1_list, client_ds2_list):
        self.assertAllEqual(elem1['image'], elem2['image'])


if __name__ == '__main__':
  tf.test.main()
