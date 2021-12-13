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
"""Hyperparameter configuration for synthesization."""

import collections
import itertools

from typing import List, Mapping, Sequence, Union


def hyper_discrete_grid(
    grid_dict: Mapping[str, Sequence[Union[str, int, float]]]
) -> List[Mapping[str, Union[str, int, float]]]:
  """Converts a param-keyed dict of lists to a list of mapping.

  Args:
    grid_dict: A Mapping from string parameter names to lists of values.

  Returns:
    A list of parameter sweep based on the Cartesian product of
    all options in param_dict.
  """
  return [
      dict(zip(grid_dict, val))
      for val in itertools.product(*grid_dict.values())
  ]


def define_parameters() -> List[Mapping[str, Union[str, int, float]]]:
  """Returns a list of dicts of parameters defining the experiment grid."""
  # Base hyperparameters grid for all experiments

  mnist_hyper = hyper_discrete_grid(  # pylint: disable=unused-variable
      collections.OrderedDict(
          dataset=['mnist'],
          synthesization=['gmm_embedding'],
          num_clients=[10, 30, 100, 300, 1000],
          gmm_embedding_efficient_net_b=[3, 7],
          gmm_embedding_use_progressive_matching=[True, False],
          gmm_embedding_kl_pairwise_batch_size=[64],
          gmm_embedding_init_params=['random'],
          gmm_embedding_pca_components=[256]))

  fashion_mnist_hyper = hyper_discrete_grid(  # pylint: disable=unused-variable
      collections.OrderedDict(
          dataset=['fashion_mnist'],
          synthesization=['gmm_embedding'],
          num_clients=[10, 30, 100, 300, 1000],
          gmm_embedding_efficient_net_b=[3, 7],
          gmm_embedding_use_progressive_matching=[True, False],
          gmm_embedding_kl_pairwise_batch_size=[64],
          gmm_embedding_init_params=['random'],
          gmm_embedding_pca_components=[256]))

  emnist10_hyper = hyper_discrete_grid(  # pylint: disable=unused-variable
      collections.OrderedDict(
          dataset=['emnist10'],
          synthesization=['gmm_embedding'],
          num_clients=[10, 30, 100, 300, 1000],
          gmm_embedding_efficient_net_b=[3, 7],
          gmm_embedding_use_progressive_matching=[True, False],
          gmm_embedding_kl_pairwise_batch_size=[50],
          gmm_embedding_init_params=['random'],
          gmm_embedding_pca_components=[256]))

  emnist62_hyper = hyper_discrete_grid(  # pylint: disable=unused-variable
      collections.OrderedDict(
          dataset=['emnist62'],
          synthesization=['gmm_embedding'],
          num_clients=[10, 30, 100, 300],
          gmm_embedding_efficient_net_b=[3, 7],
          gmm_embedding_use_progressive_matching=[True, False],
          gmm_embedding_kl_pairwise_batch_size=[50],
          gmm_embedding_init_params=['random'],
          gmm_embedding_pca_components=[256]))

  cifar10_hyper = hyper_discrete_grid(  # pylint: disable=unused-variable
      collections.OrderedDict(
          dataset=['cifar10'],
          synthesization=['gmm_embedding'],
          num_clients=[10, 30, 100],
          gmm_embedding_efficient_net_b=[3, 7],
          include_train=[False],
          include_test=[True],
          gmm_embedding_use_progressive_matching=[True, False],
          gmm_embedding_kl_pairwise_batch_size=[64],
          gmm_embedding_init_params=['kmeans'],
          gmm_embedding_pca_components=[256]))

  cifar100_hyper = hyper_discrete_grid(  # pylint: disable=unused-variable
      collections.OrderedDict(
          dataset=['cifar100'],
          synthesization=['gmm_embedding'],
          num_clients=[10, 30, 100],
          include_train=[False],
          include_test=[True],
          gmm_embedding_efficient_net_b=[3, 7],
          gmm_embedding_use_progressive_matching=[True, False],
          gmm_embedding_kl_pairwise_batch_size=[64],
          gmm_embedding_init_params=['kmeans'],
          gmm_embedding_pca_components=[256]))

  cifar10_dirichlet = hyper_discrete_grid(  # pylint: disable=unused-variable
      collections.OrderedDict(
          dataset=['cifar10'],
          synthesization=['dirichlet'],
          num_clients=[300],
          dirichlet_concentration_factor=[100, 1000],
      ))

  cifar100_coarse_dirichlet = hyper_discrete_grid(  # pylint: disable=unused-variable
      collections.OrderedDict(
          dataset=['cifar100'],
          synthesization=['coarse_dirichlet'],
          num_clients=[100],
          coarse_dirichlet_coarse_concentration_factor=[10],
          coarse_dirichlet_fine_concentration_factor=[1000],
      ))

  emnist62_dirichlet = hyper_discrete_grid(  # pylint: disable=unused-variable
      collections.OrderedDict(
          dataset=['emnist62'],
          synthesization=['dirichlet'],
          num_clients=[3000],
          dirichlet_concentration_factor=[100, 1000],
      ))

  return list(itertools.chain(cifar10_hyper, cifar100_hyper))
