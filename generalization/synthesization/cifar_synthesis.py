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
"""Synthesize a federated dataset from CIFAR10/100."""

import collections
import functools

from typing import Mapping, Optional, Tuple

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_federated as tff

from generalization.synthesization import coarse_dirichlet
from generalization.synthesization import dirichlet
from generalization.synthesization import gmm_embedding


def _cifar_consistency_postprocesser(ds: tf.data.Dataset) -> tf.data.Dataset:
  """Preprocess to keep consistency with the TFF official CIFAR dataset."""

  def elem_postprocesser(
      elem: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
    return collections.OrderedDict(image=elem['image'], label=elem['label'])

  return ds.map(elem_postprocesser)


def _load_cifar_dataset(base_dataset_name: str, include_train: bool,
                        include_test: bool) -> tf.data.Dataset:
  """Load CIFAR dataset."""
  total_ds_dict = tfds.load(base_dataset_name)

  if include_train and not include_test:
    ds = total_ds_dict['train']
  elif include_test and (not include_train):
    ds = total_ds_dict['test']
  elif include_test and include_train:
    ds = total_ds_dict['train'].concatenate(total_ds_dict['test'])
  else:
    raise ValueError('At least one of the `include_train` and'
                     '`include_test` must be True.')

  return ds


def _load_cifar_pretrained_model(efficient_net_b: int = 7) -> tf.keras.Model:
  """Load pretrained model for CIFAR."""
  model_builder = getattr(tf.keras.applications.efficientnet,
                          f'EfficientNetB{efficient_net_b}')
  base_model = model_builder(
      include_top=False,
      weights='imagenet',
      input_shape=(32, 32, 3),
  )

  inputs = tf.keras.Input(shape=(32, 32, 3))
  x = base_model(inputs, training=False)  # (None, 1, 1, 1280)
  outputs = tf.keras.layers.Flatten()(x)  # (None, 1280)

  return tf.keras.Model(inputs=inputs, outputs=outputs)


def synthesize_cifar_by_gmm_embedding(
    base_dataset_name: str,
    num_clients: int,
    efficient_net_b: int,
    pca_components: Optional[int],
    use_progressive_matching: bool,
    kl_pairwise_batch_size: int,
    gmm_init_params: str,
    include_train: bool,
    include_test: bool,
    seed: Optional[int],
) -> Tuple[tff.simulation.datasets.ClientData, str]:
  """Synthesize a federated dataset from a CIFAR-like dataset via GMM over embeddding.

  Args:
    base_dataset_name: A str representing the name of the base CIFAR-like
      dataset, can be either 'cifar10' or 'cifar100'.
    num_clients: An integer representing the number of clients to construct.
    efficient_net_b: An integer ranging from 0--7 representing the size of the
      EfficientNet pretrained model.
    pca_components: An optional integer representing the number of PCA
      components to be extracted from the embedding arrays for GMM. If None, the
      full embedding array will be used for GMM.
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
    include_train: A boolean representing whether to include training split of
      the original CIFAR dataset.
    include_test: A boolean representing whether to include test split of the
      original CIFAR dataset. At least one of the include_train and include_test
      should be True.
    seed: An optional integer representing the random seed for all random
      procedures. If None, no random seed is used.

  Returns:
    A ClientData instance holding the resulting federated dataset, and a
      str representing the name of the synthesized dataset.
  """

  dataset = _load_cifar_dataset(
      base_dataset_name, include_train=include_train, include_test=include_test)

  ds_name = base_dataset_name
  if include_train and (not include_test):
    ds_name = ds_name + '_train_only'
  elif include_test and (not include_train):
    ds_name = ds_name + '_test_only'

  name = ','.join([
      ds_name, 'gmm_embedding', f'clients={num_clients}',
      f'model=b{efficient_net_b}', f'pca={pca_components}', 'matching=' +
      ('progressive_optimal' if use_progressive_matching else 'random'),
      f'gmm_init={gmm_init_params}', f'seed={seed}'
  ])

  cd = gmm_embedding.synthesize_by_gmm_over_pretrained_embedding(
      dataset=dataset,
      pretrained_model_builder=functools.partial(
          _load_cifar_pretrained_model, efficient_net_b=efficient_net_b),
      num_clients=num_clients,
      pca_components=pca_components,
      input_name='image',
      label_name='label',
      use_progressive_matching=use_progressive_matching,
      kl_pairwise_batch_size=kl_pairwise_batch_size,
      gmm_init_params=gmm_init_params,
      seed=seed)

  cd = cd.preprocess(_cifar_consistency_postprocesser)

  return cd, name


def synthesize_cifar_by_dirichlet_over_labels(
    base_dataset_name: str, num_clients: int, concentration_factor: float,
    use_rotate_draw: bool, include_train: bool, include_test: bool,
    seed: Optional[int]) -> Tuple[tff.simulation.datasets.ClientData, str]:
  """Synthesize a federated dataset from a CIFAR-like dataset via dirichlet over labels.

  Args:
    base_dataset_name: A str representing the name of the base CIFAR-like
      dataset, can be either 'cifar10' or 'cifar100'.
    num_clients: An integer representing the number of clients to construct.
    concentration_factor:  A float-typed parameter of Dirichlet distribution.
      Each client will sample from Dirichlet(concentration_factor *
      label_relative_popularity) to get a multinomial distribution over labels.
      It controls the data heterogeneity of clients. If approaches 0, then each
      client only have data from a single category label. If approaches
      infinity, then the client distribution will approach overall popularity.
    use_rotate_draw: Whether to rotate the drawing clients. If True, each client
      will draw only one sample at once, and then rotate to the next random
      client. This is intended to prevent the last clients from deviating from
      its desired distribution. If False, a client will draw all the samples at
      once before moving to the next client.
    include_train: A boolean representing whether to include training split of
      the original CIFAR dataset.
    include_test: A boolean representing whether to include test split of the
      original CIFAR dataset. At least one of the include_train and include_test
      should be True.
    seed: An optional integer representing the random seed for all random
      procedures. If None, no random seed is used.

  Returns:
    A ClientData instance holding the resulting federated dataset, and a
      str representing the name of the synthesized dataset.
  """
  dataset = _load_cifar_dataset(
      base_dataset_name, include_train=include_train, include_test=include_test)

  ds_name = base_dataset_name
  if include_train and (not include_test):
    ds_name = ds_name + '_train_only'
  elif include_test and (not include_train):
    ds_name = ds_name + '_test_only'

  name = ','.join([
      ds_name, 'dirichlet', f'clients={num_clients}',
      f'concentration_factor={concentration_factor}',
      f'rotate={use_rotate_draw}', f'seed={seed}'
  ])

  cd = dirichlet.synthesize_by_dirichlet_over_labels(
      dataset=dataset,
      num_clients=num_clients,
      concentration_factor=concentration_factor,
      use_rotate_draw=use_rotate_draw,
      seed=seed)

  cd = cd.preprocess(_cifar_consistency_postprocesser)

  return cd, name


def synthesize_cifar100_over_coarse_and_fine_labels(
    num_clients: int, coarse_concentration_factor: float,
    fine_concentration_factor: float, use_rotate_draw: bool,
    include_train: bool, include_test: bool,
    seed: Optional[int]) -> Tuple[tff.simulation.datasets.ClientData, str]:
  """Synthesize a federated dataset from CIFAR100 via dirichlet over coarse and fine labels.

  Args:
    num_clients: An integer representing the number of clients to construct.
    coarse_concentration_factor: A float-typed parameter of Dirichlet
      distribution to draw coarse labels.
    fine_concentration_factor: A float-typed parameter of Dirichlet distribution
      to draw fine labels.
    use_rotate_draw: Whether to rotate the drawing clients. If True, each client
      will draw only one sample at once, and then rotate to the next random
      client. This is intended to prevent the last clients from deviating from
      its desired distribution. If False, a client will draw all the samples at
      once before moving to the next client.
    include_train: A boolean representing whether to include training split of
      the original CIFAR dataset.
    include_test: A boolean representing whether to include test split of the
      original CIFAR dataset. At least one of the include_train and include_test
      should be True.
    seed: An optional integer representing the random seed for all random
      procedures. If None, no random seed is used.

  Returns:
    A ClientData instance holding the resulting federated dataset, and a
      str representing the name of the synthesized dataset.
  """

  dataset = _load_cifar_dataset(
      'cifar100', include_train=include_train, include_test=include_test)

  ds_name = 'cifar100'
  if include_train and (not include_test):
    ds_name = ds_name + '_train_only'
  elif include_test and (not include_train):
    ds_name = ds_name + '_test_only'

  name = ','.join([
      ds_name, 'coarse_dirichlet', f'clients={num_clients}',
      f'coarse_factor={coarse_concentration_factor}',
      f'fine_factor={fine_concentration_factor}', f'rotate={use_rotate_draw}',
      f'seed={seed}'
  ])

  cd = coarse_dirichlet.synthesize_by_dirichlet_over_coarse_and_fine_labels(
      dataset=dataset,
      num_clients=num_clients,
      coarse_concentration_factor=coarse_concentration_factor,
      fine_concentration_factor=fine_concentration_factor,
      use_rotate_draw=use_rotate_draw,
      seed=seed)

  cd = cd.preprocess(_cifar_consistency_postprocesser)

  return cd, name
