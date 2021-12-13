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
"""GMM Clustering within each label."""

from typing import Callable, List, Mapping, Optional

from absl import logging
import numpy as np
import scipy.optimize
import sklearn.decomposition
import sklearn.preprocessing
import tensorflow as tf
import tensorflow_federated as tff
import tensorflow_probability as tfp

from generalization.utils import client_data_utils
from generalization.utils import logging_utils
from generalization.utils import tf_gaussian_mixture

tfd = tfp.distributions


# Intentionally remove the @tf.function decorator to avoid retracing OOM
def _pairwise_kl_divergence_between_multivariate_normal_tril_one_batch(
    means_1: tf.Tensor, trils_1: tf.Tensor, means_2: tf.Tensor,
    trils_2: tf.Tensor) -> tf.Tensor:
  """Compute pairwise KL Divergence between two batches of distribution."""

  dists_first = tfd.MultivariateNormalTriL(
      tf.expand_dims(means_1, 1), tf.expand_dims(trils_1, 1))
  dists_second = tfd.MultivariateNormalTriL(
      tf.expand_dims(means_2, 0), tf.expand_dims(trils_2, 0))

  return tfd.kl_divergence(dists_first, dists_second)


def pairwise_kl_divergence_between_multivariate_normal_tril_in_batch(
    means_1: tf.Tensor,
    trils_1: tf.Tensor,
    means_2: tf.Tensor,
    trils_2: tf.Tensor,
    batch_size: Optional[int] = 100) -> tf.Tensor:
  """Compute pairwise KL Divergence between two groups of Gaussian distribution.

  This function will accept two groups of tfd.MultivariateNormalTriL
    distributions parameterized by (means_1, trils_1) and (means_2, trils_2)
    repsectively, and return the pairwise KL divergence between the two groups.

  The (i, j)-th element of the returned pairwise_kl_matrix is the KL divergence
  of the following two distributions:
    - tfd.MultivariateNormalTriL(means_1[i,:], trils_1[i, :, :])
    - tfd.MultivariateNormalTriL(means_2[i,:], trils_2[i, :, :])

  To speedup computation we will compute in batch. At each time, (at most)
    batch_size * batch_size part of the pairwise matrix will be computed. If
    batch_size is None, the entire matrix will be computed in one batch.
    (Warning: This may incur non-trivial memory surge or OOM since the
    underlying function will attempt to allocate a tensor of shape
    batch_size * batch_size * dim * dim).

  Args:
    means_1: A `tf.Tensor` of shape [num_distributions, embedding_dim]
      representing the means of the first group of Gaussian distributions.
    trils_1: A `tf.Tensor` of shape [num_distributions, embedding_dim,
      embedding_dim] representing the trils of the first group of Gaussian
      distributions.
    means_2: A `tf.Tensor` of shape [num_distributions, embedding_dim]
      representing the means of the second group of Gaussian distributions.
    trils_2: A `tf.Tensor` of shape [num_distributions, embedding_dim,
      embedding_dim] representing the trils of the second group of Gaussian
      distributions.
    batch_size: An optional integer representing the (block) batch size. If
      None, the entire matrix will be computed in one batch.

  Returns:
    The pairwise KL-divergence matrix of `tf.Tensor` type, see above for
    details.
  """

  num_dist_1, num_dist_2 = means_1.shape[0], means_2.shape[0]
  pairwise_kl_matrix = tf.Variable(
      tf.zeros((num_dist_1, num_dist_2), dtype=means_1.dtype))

  logger = logging_utils.ProgressLogger(
      name='computing pairwise KL divergence cost matrix',
      every=(tf.size(pairwise_kl_matrix) + 9) // 10,
      total=tf.size(pairwise_kl_matrix))

  if batch_size is None:
    pairwise_kl_matrix.assign(
        _pairwise_kl_divergence_between_multivariate_normal_tril_one_batch(
            means_1, trils_1, means_2, trils_2))
  else:
    for first_chunk_base in range(0, means_1.shape[0], batch_size):
      for second_chunk_base in range(0, means_2.shape[0], batch_size):
        means_1_batch = means_1[first_chunk_base:first_chunk_base +
                                batch_size, :]
        trils_1_batch = trils_1[first_chunk_base:first_chunk_base +
                                batch_size, :, :]
        means_2_batch = means_2[second_chunk_base:second_chunk_base +
                                batch_size, :]
        trils_2_batch = trils_2[second_chunk_base:second_chunk_base +
                                batch_size, :, :]

        batch_matrix = _pairwise_kl_divergence_between_multivariate_normal_tril_one_batch(
            means_1_batch,
            trils_1_batch,
            means_2_batch,
            trils_2_batch,
        )
        pairwise_kl_matrix[first_chunk_base:first_chunk_base + batch_size,
                           second_chunk_base:second_chunk_base +
                           batch_size].assign(batch_matrix)

        logger.increment(tf.size(batch_matrix))

  return tf.convert_to_tensor(pairwise_kl_matrix)


class _GMMOverPretrainedEmbeddingSynthesizer():
  # TODO(b/210260308): Add arxiv identifier after acceptance.
  """Backend class of function synthesize_by_GMM_over_pretrained_embedding().

  Please refer to section D of paper "What Do We Mean by Generalization in
    Federated Learning?" for detailed descriptions.
  """

  def __init__(self,
               dataset: tf.data.Dataset,
               pretrained_model_builder: Callable[[], tf.keras.Model],
               num_clients: int,
               pca_components: Optional[int] = None,
               input_name: str = 'image',
               label_name: str = 'label',
               gmm_init_params: str = 'random',
               seed: Optional[int] = None):
    self._seed = seed
    self._rng = np.random.default_rng(self._seed)
    self._dataset = dataset
    self._num_clusters_per_label = num_clients
    self._client_ids = list(map(str, range(self._num_clusters_per_label)))
    self._element_spec = dataset.element_spec
    self._input_name = input_name
    self._label_name = label_name

    self._label_tensor = self._unpack_label_tensor()
    self._label_set = set(self._label_tensor.numpy())
    self._embedding_tensor = self._compute_embedding_tensor(
        pretrained_model_builder)

    self._normalize_in_place()
    # Reduce the embedding dimensions by PCA, if pca_components is not None.
    if pca_components is not None:
      self._pca_transform_in_place(pca_components)

    self._build_label_clusters(gmm_init_params=gmm_init_params)
    del self._embedding_tensor

  def _unpack_label_tensor(self) -> tf.Tensor:
    """Compute label tensor."""
    logging.info('Starting unpacking label tensor.')
    with tf.device('CPU'):
      label_tensor = tf.stack(
          [elem[self._label_name] for elem in self._dataset])

    logging.info('Finished unpacking label tensor.')
    return label_tensor

  def _compute_embedding_tensor(
      self, pretrained_model_builder=Callable[[], tf.keras.Model]) -> tf.Tensor:
    """Compute embedding tensor."""

    pretrained_model = pretrained_model_builder()

    embedding_dataset = self._dataset.batch(256).map(
        lambda batch: pretrained_model(batch[self._input_name])).unbatch()

    embedding_list = []

    logger = logging_utils.ProgressLogger(
        name='computing embedding',
        every=10000,
        total=tf.size(self._label_tensor))
    for embedding in embedding_dataset:
      embedding_list.append(embedding)
      logger.increment()

    embedding_tensor = tf.stack(embedding_list)

    return embedding_tensor

  def _normalize_in_place(self) -> None:
    """Normalize embedding in place."""
    _ = logging_utils.ProgressLogger('computing embedding normalization')
    with tf.device('CPU'):
      self._embedding_tensor = tf.convert_to_tensor(
          sklearn.preprocessing.StandardScaler(with_std=False).fit_transform(
              self._embedding_tensor))

  def _subsample_tensor_by_rows(self, tensor: tf.Tensor,
                                num_subsamples: int) -> tf.Tensor:
    if num_subsamples > tensor.shape[0]:
      return tensor
    else:
      subsamples_idx = self._rng.choice(range(tensor.shape[0]), num_subsamples)
      return tf.gather(tensor, subsamples_idx)

  def _pca_transform_in_place(self, pca_components, fit_limit=100000) -> None:
    """Computing in-place PCA."""

    # This PCA step can cause CPU/GPU OOM if fit_limit is too large.
    _ = logging_utils.ProgressLogger('computing PCA')
    logging.info('Starting computing PCA.')
    with tf.device('CPU'):
      pca = sklearn.decomposition.PCA(pca_components, random_state=self._seed)
      subsample_tensor = self._subsample_tensor_by_rows(self._embedding_tensor,
                                                        fit_limit)
      pca.fit(subsample_tensor)
      self._embedding_tensor = tf.convert_to_tensor(
          pca.transform(self._embedding_tensor))

    return None

  def _build_label_clusters(self,
                            fit_limit: int = 10000,
                            gmm_init_params: str = 'random') -> None:
    """Generate clusters for each label."""
    self._label_cluster_list = {
        label: [list() for _ in range(self._num_clusters_per_label)
               ] for label in self._label_set
    }
    self._label_cluster_means = {label: None for label in self._label_set}
    self._label_cluster_trils = {label: None for label in self._label_set}

    for current_label in self._label_set:
      logging.info('Starting clustering label %s', current_label)

      idx_list_of_current_label = tf.reshape(
          tf.where(self._label_tensor == current_label), [-1])
      embedding_tensor_of_current_label = tf.gather(self._embedding_tensor,
                                                    idx_list_of_current_label)
      subsample_embedding_tensor_of_current_label = self._subsample_tensor_by_rows(
          embedding_tensor_of_current_label, fit_limit)

      logging.info(
          '  Starting fitting GMM for label %s: '
          'n_samples (for fitting) = %d, '
          'n_features = %d, n_components = %d.', current_label,
          subsample_embedding_tensor_of_current_label.shape[0],
          subsample_embedding_tensor_of_current_label.shape[1],
          self._num_clusters_per_label)
      gm = tf_gaussian_mixture.GaussianMixture(
          self._num_clusters_per_label,
          verbose=2,
          reg_covar=1e-4,
          verbose_interval=1,
          init_params=gmm_init_params,
          kmeans_batch_size=None if gmm_init_params == 'random' else 10000,
          random_state=self._seed)
      gm.fit(subsample_embedding_tensor_of_current_label)
      logging.info('  Finished fitting GMM for label %s.', current_label)

      logging.info(
          '  Starting predicting with GMM for label %s: '
          'n_samples = %d, n_features = %d, n_components = %d.', current_label,
          embedding_tensor_of_current_label.shape[0],
          embedding_tensor_of_current_label.shape[1],
          self._num_clusters_per_label)
      cluster_arr = gm.predict(embedding_tensor_of_current_label)

      for idx, cluster in zip(idx_list_of_current_label, cluster_arr):
        self._label_cluster_list[current_label][cluster].append(idx)

      # Log min/max of each clusters.
      min_cluster = min(
          [len(cluster) for cluster in self._label_cluster_list[current_label]])
      max_cluster = max(
          [len(cluster) for cluster in self._label_cluster_list[current_label]])
      logging.info('Finished clustering label %s result: min %d, max %d',
                   current_label, min_cluster, max_cluster)

      cov_cholesky = tf.linalg.cholesky(gm.covariances_)
      with tf.device('CPU'):
        self._label_cluster_means[current_label] = tf.identity(gm.means_)
        self._label_cluster_trils[current_label] = tf.identity(cov_cholesky)

    return None

  def _build_client_data_from_dict_of_idx(
      self,
      dict_of_idx: Mapping[str,
                           List[int]]) -> tff.simulation.datasets.ClientData:
    """Build ClientData from the dict of indices of the base dataset."""

    tensor_slices_dict = dict()
    original_dataset_list = list(self._dataset)
    for client_id in self._client_ids:
      local_dataset_list = [
          original_dataset_list[idx] for idx in dict_of_idx[client_id]
      ]
      logging.info('Client %s dataset length: %d.', client_id,
                   len(local_dataset_list))
      tensor_slices_dict[
          client_id] = client_data_utils.convert_list_of_elems_to_tensor_slices(
              local_dataset_list)

    return tff.simulation.datasets.TestClientData(tensor_slices_dict)

  def build_client_data_by_random_matching(
      self) -> tff.simulation.datasets.ClientData:
    """Build federated ClientData."""

    del self._label_cluster_means
    del self._label_cluster_trils

    for label in self._label_set:
      self._rng.shuffle(self._label_cluster_list[label])

    client_data_as_dict_of_idx = {
        client_id: list() for client_id in self._client_ids
    }

    for client_id, cluster in zip(self._client_ids,
                                  range(self._num_clusters_per_label)):
      for label in self._label_cluster_list:
        client_data_as_dict_of_idx[client_id].extend(
            self._label_cluster_list[label][cluster])

    return self._build_client_data_from_dict_of_idx(client_data_as_dict_of_idx)

  def build_client_data_by_progressive_matching(
      self, kl_pairwise_batch_size=100) -> tff.simulation.datasets.ClientData:
    """Build federated ClientData by progressive matching."""
    cluster_assignment = {
        label: [None for _ in range(self._num_clusters_per_label)
               ] for label in self._label_cluster_means
    }

    unmatched_labels = list(self._label_cluster_means.keys())
    # All labels are not matched as of now.

    latest_matched_label = self._rng.choice(unmatched_labels)
    cluster_assignment[latest_matched_label] = self._client_ids

    unmatched_labels.remove(latest_matched_label)

    while unmatched_labels:
      label_to_match = self._rng.choice(unmatched_labels)
      logging.info('Matching label %s with label %s, %d out of %d processed.',
                   label_to_match, latest_matched_label,
                   len(self._label_set) - len(unmatched_labels),
                   len(self._label_set))

      cost_matrix = pairwise_kl_divergence_between_multivariate_normal_tril_in_batch(
          self._label_cluster_means[latest_matched_label],
          self._label_cluster_trils[latest_matched_label],
          self._label_cluster_means[label_to_match],
          self._label_cluster_trils[label_to_match],
          kl_pairwise_batch_size).numpy()

      logging.info('  Starting computing linear sum assignment.')
      optimal_local_assignment = scipy.optimize.linear_sum_assignment(
          cost_matrix)
      logging.info('  Finished computing linear sum assignment.')

      for c in range(self._num_clusters_per_label):
        cluster_assignment[label_to_match][optimal_local_assignment[1][
            c]] = cluster_assignment[latest_matched_label][
                optimal_local_assignment[0][c]]

      unmatched_labels.remove(label_to_match)
      latest_matched_label = label_to_match

    del self._label_cluster_means
    del self._label_cluster_trils

    client_data_as_dict_of_idx = {
        client_id: list() for client_id in self._client_ids
    }

    for label in cluster_assignment:
      for c in range(self._num_clusters_per_label):
        client_data_as_dict_of_idx[cluster_assignment[label][c]].extend(
            self._label_cluster_list[label][c])

    return self._build_client_data_from_dict_of_idx(client_data_as_dict_of_idx)


def synthesize_by_gmm_over_pretrained_embedding(
    dataset: tf.data.Dataset,
    pretrained_model_builder: Callable[[], tf.keras.Model],
    num_clients: int,
    pca_components: Optional[int] = None,
    input_name: str = 'image',
    label_name: str = 'label',
    use_progressive_matching: bool = True,
    kl_pairwise_batch_size: Optional[int] = 100,
    gmm_init_params: str = 'random',
    seed: Optional[int] = None) -> tff.simulation.datasets.ClientData:
  # TODO(b/210260308): Add arxiv identifier after acceptance.
  """Construct a federated dataset from a centralized dataset based on GMM clustering.

  Please refer to section D of paper "What Do We Mean by Generalization in
    Federated Learning?" for detailed descriptions.

  Assumptions:
    1) `dataset` should has `element_spec` of type `Mapping[str,
    tf.TensorSpec]`, with a hashable label keyed by label_name, and input keyed
    by input_name.

  Limitations:
    1) The current implementations will unpack the entire dataset into
    the memory. This could result in large memory use if the dataset is large.

  Args:
    dataset: The original tf.data.Dataset to be partitioned.
    pretrained_model_builder: A callable that returns the pre-trained
      `tf.keras.Model` to compute embedding.
    num_clients: An integer representing the number of clients to construct.
    pca_components: An optional integer representing the number of PCA
      components to be extracted from the embedding arrays for GMM. If None, the
      full embedding array will be used for GMM.
    input_name: A str representing the name of input in `dataset` elements.
    label_name: A str representing the name of label in `dataset` elements.
    use_progressive_matching: Whether to use progressive matching. If True, the
      function will progressively match the clusters of one unmatched label with
      a matched label by computing the optimal bipartite matching under pairwise
      KL divergence. If False, the function will randomly match the clusters
      across labels.
    kl_pairwise_batch_size: An optional integer representing the batch size when
      computing pairwise KL divergence. If None, the full cost matrix will be
      computed in one batch. This could result in large memory cost.
    gmm_init_params: A str representing the initialization mode of GMM, can be
      either 'random' or 'kmeans'.
    seed: An optional integer representing the random seed for all random
      procedures. If None, no random seed is used.

  Returns:
    A ClientData instance.
  """
  synthesizer = _GMMOverPretrainedEmbeddingSynthesizer(
      dataset=dataset,
      pretrained_model_builder=pretrained_model_builder,
      num_clients=num_clients,
      pca_components=pca_components,
      input_name=input_name,
      label_name=label_name,
      gmm_init_params=gmm_init_params,
      seed=seed)

  if not use_progressive_matching:
    return synthesizer.build_client_data_by_random_matching()
  else:
    return synthesizer.build_client_data_by_progressive_matching(
        kl_pairwise_batch_size=kl_pairwise_batch_size)
