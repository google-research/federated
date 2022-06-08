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
"""Library containing strategies for initializing SGD procedure."""
from typing import Any, List, Mapping

import tensorflow as tf

from dp_matrix_factorization import matrix_constructors

SUPPORTED_STRATEGIES = [
    'binary_tree', 'extended_binary_tree', 'identity',
    'random_binary_tree_structure', 'double_h_solution'
]


def _ensure_has_entries(param_dict: Mapping[str, Any], entries: List[str]):
  for entry in entries:
    if param_dict.get(entry) is None:
      raise ValueError(f'Expected parameter dict to have entry {entry}; found '
                       f'instead keys {param_dict.keys()}')


def get_initial_h(strategy: str, params: Mapping[str, Any]) -> tf.Tensor:
  """Initializes H according to `strategy` and `params`.

  The following initializations are supported:

  * binary_tree: Initializes H to be the binary tree matrix with a given
    number of leaves, specified as a base-2 logarithm.
  * extended_binary_tree: Initializes H to be a binary tree specified as above,
    in addition to extra rows of zeros. Intended for empirically investigating
    potential effects of higher-dimensional intermediate space.
  * identity: Initializes H to be the identity matrix of given dimensionality,
    again specified as a base-2 logarithm
  * random_normal_binary_tree_structure: Initializes H to be a matrix of similar
    shape and dtype as the binary tree matrix, with similar lower-triangular
    structure; that is, if an zero entry appears above and to the right of all
    nonzero entries in the binary tree matrix, it will be zero in this random
    matrix as well.
  * double_h_solution: Stamps out the supplued h_to_double matrix along the
    block-diagonal of a new matrix, thus creating a matrix with doubled rows
    and doubled columns.

  Args:
    strategy: A string indicating the strategy enumerated above with which to
      initialize H.
    params: An ordered dict containing the appropriate parameters to use in
      initialization.

  Returns:
    A rank-2 tensor constructed according to the specification above.
  """
  if strategy == 'binary_tree':
    _ensure_has_entries(params, ['log_2_leaves'])
    return tf.constant(
        matrix_constructors.binary_tree_matrix(
            log_2_leaves=params['log_2_leaves']))
  elif strategy == 'extended_binary_tree':
    _ensure_has_entries(params, ['log_2_leaves', 'num_extra_rows'])
    return matrix_constructors.extended_binary_tree(
        log_2_leaves=params['log_2_leaves'],
        num_extra_rows=params['num_extra_rows'])
  elif strategy == 'identity':
    _ensure_has_entries(params, ['log_2_leaves'])
    return tf.eye(2**params['log_2_leaves'])
  elif strategy == 'random_binary_tree_structure':
    _ensure_has_entries(params, ['log_2_leaves'])
    return matrix_constructors.random_normal_binary_tree_structure(
        log_2_leaves=params['log_2_leaves'])
  elif strategy == 'double_h_solution':
    _ensure_has_entries(params, ['h_to_double'])
    return matrix_constructors.double_h_solution(params['h_to_double'])
  else:
    raise ValueError(f'Unknown initialization strategy: {strategy}. Expected '
                     f'one of {SUPPORTED_STRATEGIES}.')
