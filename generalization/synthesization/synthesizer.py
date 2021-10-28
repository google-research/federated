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
"""Binary for synthesizing federated datasets."""

import os
from typing import Sequence

from absl import app
from absl import flags
from absl import logging

from generalization.synthesization import cifar_synthesis
from generalization.synthesization import mnist_synthesis
from generalization.utils import sql_client_data_utils

flags.DEFINE_string('root_output_dir', '/tmp/dataset',
                    'Root directory for the output dataset')

flags.DEFINE_integer('num_clients', 3000, 'Number of clients to synthesize.')

flags.DEFINE_integer('seed', 1, 'Random seed.')

flags.DEFINE_enum(
    'dataset', None,
    ['mnist', 'fashion_mnist', 'emnist10', 'emnist62', 'cifar10', 'cifar100'],
    'Which dataset to synthesize clients from.')

flags.DEFINE_enum('synthesization', None,
                  ['dirichlet', 'coarse_dirichlet', 'gmm_embedding'],
                  'Which synthesization scheme to use.')

# Dirichlet synthesization scheme flags.
flags.DEFINE_float('dirichlet_concentration_factor', 10,
                   'Concentration factor in Dirichlet sampling process.')
flags.DEFINE_boolean(
    'dirichlet_rotate_draw', True,
    'Whether to rotate the drawing clients every sample when '
    'using dirichlet synthesization.')

# Coarse-fine dirichlet synthesization scheme flags.
flags.DEFINE_float(
    'coarse_dirichlet_coarse_concentration_factor', 10,
    'Coarse concentration factor in coarse-fine Dirichlet sampling process.')
flags.DEFINE_float(
    'coarse_dirichlet_fine_concentration_factor', 10,
    'Fine concentration factor in coarse-fine Dirichlet sampling process.')
flags.DEFINE_boolean(
    'coarse_dirichlet_rotate_draw', True,
    'Whether to rotate the drawing clients every sample when '
    'using coarse-fine dirichlet synthesization.')

# GMM over embedding synthesis scheme flags.
flags.DEFINE_integer(
    'gmm_embedding_efficient_net_b', 7,
    'Efficient Net pre-trained model to be used for generating embeddding.')
flags.DEFINE_integer(
    'gmm_embedding_pca_components', None,
    'An optional integer representing the number of PCA components for reducing '
    'the dimensions of embedding. If None, the full-dimension raw embedding will be used.'
)
flags.DEFINE_boolean(
    'gmm_embedding_use_progressive_matching', True,
    'Whether to use progressive matching for gmm embedding matching.')
flags.DEFINE_enum('gmm_embedding_init_params', 'random', ['random', 'kmeans'],
                  'Initialization for GMM procedure.')
flags.DEFINE_integer('gmm_embedding_kl_pairwise_batch_size', 10,
                     'Block batch size when computing KL cost matrix.')
FLAGS = flags.FLAGS


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  logging.info('Starting synthesizing dataset.')
  if FLAGS.dataset in ('mnist', 'fashion_mnist', 'emnist10', 'emnist62'):
    if FLAGS.synthesization == 'dirichlet':
      cd, cd_name = mnist_synthesis.synthesize_mnist_by_dirichlet_over_labels(
          base_dataset_name=FLAGS.dataset,
          num_clients=FLAGS.num_clients,
          concentration_factor=FLAGS.dirichlet_concentration_factor,
          use_rotate_draw=FLAGS.dirichlet_rotate_draw,
          seed=FLAGS.seed)
    elif FLAGS.synthesization == 'gmm_embedding':
      cd, cd_name = mnist_synthesis.synthesize_mnist_by_gmm_embedding(
          base_dataset_name=FLAGS.dataset,
          efficient_net_b=FLAGS.gmm_embedding_efficient_net_b,
          num_clients=FLAGS.num_clients,
          pca_components=FLAGS.gmm_embedding_pca_components,
          use_progressive_matching=FLAGS.gmm_embedding_use_progressive_matching,
          kl_pairwise_batch_size=FLAGS.gmm_embedding_kl_pairwise_batch_size,
          gmm_init_params=FLAGS.gmm_embedding_init_params,
          seed=FLAGS.seed,
      )
    else:
      raise ValueError(
          f'Unknown synthesization scheme "{FLAGS.synthesization}" for dataset f{FLAGS.dataset}.'
      )
  elif FLAGS.dataset in ('cifar10', 'cifar100'):
    if FLAGS.synthesization == 'dirichlet':
      cd, cd_name = cifar_synthesis.synthesize_cifar_by_dirichlet_over_labels(
          base_dataset_name=FLAGS.dataset,
          num_clients=FLAGS.num_clients,
          concentration_factor=FLAGS.dirichlet_concentration_factor,
          use_rotate_draw=FLAGS.dirichlet_rotate_draw,
          seed=FLAGS.seed)
    elif FLAGS.synthesization == 'gmm_embedding':
      cd, cd_name = cifar_synthesis.synthesize_cifar_by_gmm_embedding(
          base_dataset_name=FLAGS.dataset,
          efficient_net_b=FLAGS.gmm_embedding_efficient_net_b,
          num_clients=FLAGS.num_clients,
          pca_components=FLAGS.gmm_embedding_pca_components,
          use_progressive_matching=FLAGS.gmm_embedding_use_progressive_matching,
          kl_pairwise_batch_size=FLAGS.gmm_embedding_kl_pairwise_batch_size,
          gmm_init_params=FLAGS.gmm_embedding_init_params,
          seed=FLAGS.seed,
      )
    elif FLAGS.dataset == 'cifar100' and FLAGS.synthesization == 'coarse_dirichlet':
      cd, cd_name = cifar_synthesis.synthesize_cifar100_over_coarse_and_fine_labels(
          num_clients=FLAGS.num_clients,
          coarse_concentration_factor=FLAGS
          .coarse_dirichlet_coarse_concentration_factor,
          fine_concentration_factor=FLAGS
          .coarse_dirichlet_fine_concentration_factor,
          use_rotate_draw=FLAGS.coarse_dirichlet_rotate_draw,
          seed=FLAGS.seed)
    else:
      raise ValueError(
          f'Unknown synthesization scheme "{FLAGS.synthesization}" for dataset {FLAGS.dataset}.'
      )
  else:
    raise ValueError(f'Unknown dataset {FLAGS.dataset}.')
  logging.info('Finished synthesizing dataset.')

  database_filepath = os.path.join(FLAGS.root_output_dir, cd_name + '.db')

  logging.info('Starting saving sql dataset.')
  sql_client_data_utils.save_to_sql_client_data_from_client_data(
      cd, database_filepath=database_filepath, allow_overwrite=True)


if __name__ == '__main__':
  app.run(main)
