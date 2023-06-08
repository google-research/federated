# Copyright 2023, Google LLC.
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
"""Library defining matrices relevant to prefix-sum factorization."""
from typing import Any

import numpy as np
import scipy
import tensorflow as tf

from multi_epoch_dp_matrix_factorization import constraint_builders
from multi_epoch_dp_matrix_factorization import matrix_factorization_query


def _double_leaves(tree_matrix: np.ndarray) -> np.ndarray:
  """Duplicates `tree_matrix` along the diagonal, adding a row of 1s."""
  rows, cols = tree_matrix.shape
  return np.block([
      [tree_matrix, np.zeros(shape=(rows, cols))],
      [np.zeros(shape=(rows, cols)), tree_matrix],
      [np.ones(shape=(2 * cols))],
  ])


def binary_tree_matrix(*, log_2_leaves: int) -> tf.Tensor:
  r"""Constructs a matrix representing binary-tree aggregation.

  By binary-tree aggregation, here we mean: take a vector
  `x = (x_1, x_2, ... x_n)`. Assume `n % 2 == 0, and consider the entries `x_i`
  as leaves of a binary tree. Assign to the intermediate nodes of this tree the
  sum of the values of its children, so that e.g. the node with children
  corresponding to `x_1` and `x_2` has value `x_1 + x2`, and so on.

  Then, interpreting the vector `x` above as a column vector, multiplying x (on
  the left) by the matrix returned by this function will result in a vector
  which contains the same values as those decorating the nodes of the tree
  described above.

  The description we give above is not quite a full specification of this
  function; indeed, permuting the rows of the returned matrix still satisfies
  this description. We choose to normalize so that multiplying `x` by the matrix
  we return here corresponds to a postorder traversal of the tree described
  above.

  As an example, for the vector `(x_1, x_2, x_3, x_4)`, we would construct the
  tree:
                                  x_1+x_2+x_3+x_4
                                 /               \
                              x_1+x_2           x_3+x_4
                             /       \         /       \
                           x_1       x_2     x_3       x_4
  with corresponding matrix:

  [[1, 0, 0, 0],
   [0, 1, 0, 0],
   [1, 1, 0, 0],
   [0, 0, 1, 0],
   [0, 0, 0, 1],
   [0, 0, 1, 1],
   [1, 1, 1, 1]]

  Args:
    log_2_leaves: An integer defining the log-base-two of the number of leaves
      to include in the binary tree construction described above (IE, `n = 2 **
      log_2_leaves`). Must be passed as a keyword.

  Returns:
    A rank-two `numpy ndarray`, representing a matrix as described above.

  Raises:
    ValueError: If `log_2_leaves` is negative.
  """
  if log_2_leaves < 0:
    raise ValueError(
        'Expected log-base-two of the number of leaves to be a '
        f'nonnegative integer; found {log_2_leaves}'
    )
  m = np.array([[1.0]])
  for _ in range(log_2_leaves):
    m = _double_leaves(m)
  return tf.constant(m)


def _compute_h_mask(h_with_zeros: tf.Tensor) -> tf.Tensor:
  h_mask = np.zeros(h_with_zeros.shape)
  for row_idx in range(h_mask.shape[0]):
    for col_idx in range(h_mask.shape[1]):
      if np.sum(np.abs(h_with_zeros[: row_idx + 1, col_idx])) > 0:
        h_mask[row_idx, col_idx] = 1
  return tf.constant(h_mask)


def _compute_matrix_vars_for_streaming(h_matrix: tf.Tensor) -> tf.Tensor:
  """Computes a rank-2 boolean tensor indicating nonzero entries for W.

  The specification of this function can be found in the docstring of
  `compute_masks_for_streaming`.

  Args:
    h_matrix: two-dimensional numpy ndarray or eager TensorFlow tensor, used to
      compute streaming requirements for the matrix `W` as described above.

  Returns:
    A rank-2 boolean Tensor representing the constraints on W as specified
    above.
  """
  observation_idx_available = (
      constraint_builders.compute_observations_available(h_matrix)
  )
  constructed_bool_matrix = []
  num_measurements, num_observations = h_matrix.shape

  def _measurement_available(observation_index, measurement_index):
    return observation_idx_available[measurement_index] <= observation_index

  for i in range(num_observations):
    constructed_row = [
        _measurement_available(i, j) for j in range(num_measurements)
    ]
    constructed_bool_matrix.append(constructed_row)
  return tf.constant(constructed_bool_matrix)


def compute_masks_for_streaming(
    h_matrix: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
  """Returns masks ensuring that a factorization respects streaming constraints.

  That is, for an incoming `h_matrix`, the lower-triangular structure of this
  matrix determines a compatibility with streaming algorithmic requirements.
  For a row i, the rightmost nonzero column j of this row introduces the
  requirement that precisely j entries of the vector x = (x_1, ... x_n) must be
  available before element i of the product h_matrix @ x can be computed
  (assuming the x_i materialize sequentially in time).

  This function computes these requirements, and the inferred requirements on
  W. The first tensor returned represents the lower-triangular structure of
  `h_matrix`, so that every entry 'below or to the left' of a nonzero element of
  `h_matrix` corresponds to a 1 in the firest return tensor, and al else zeros.

  The second corresponds to the elements of  W. If an entry in this matrix is
  `1`, the corresponding element of the `W` matrix may be nonzero; if it is
  `0`, the corresponding element must be zero.

  If a matrix pair (`W`, `H`) is constructed which satisfies these constraints,
  then, computing the ith element of the vector `WHx` requires that at most `i`
  of the elements of the vector `x` have materialized; that is, that the product
  `WHx` can be computed in a streaming fashion.

  Args:
    h_matrix: H matrix to be used to compute the masks as described above.

  Returns:
    A two-tuple of tensors, representing inferred structural constraints on W
    and H to respect streaming constraints, as described above.
  """
  h_mask = tf.cast(_compute_h_mask(h_matrix), h_matrix.dtype)
  w_mask = tf.cast(_compute_matrix_vars_for_streaming(h_matrix), tf.int32)
  return h_mask, w_mask


def random_normal_binary_tree_structure(*, log_2_leaves: int) -> tf.Tensor:
  """Constructs a random matrix similar to the binary tree matrix.

  More precisely, the constructed matrix here will have similar shape, dtype,
  and lower-triangular structure to the matrix returned by the binary tree
  matrix constructor above.

  Args:
    log_2_leaves: The base-2 logarithm of the number of columns of the
      constructed matrix.

  Returns:
    A matrix constructed according to the specifications above.
  """
  bin_tree_matrix = binary_tree_matrix(log_2_leaves=log_2_leaves)
  shape = bin_tree_matrix.shape
  random_matrix = tf.random.normal(shape=shape, dtype=bin_tree_matrix.dtype)
  mask = _compute_h_mask(bin_tree_matrix)
  return tf.constant(random_matrix * mask)


def extended_binary_tree(
    *, log_2_leaves: int, num_extra_rows: int
) -> tf.Tensor:
  bin_tree = binary_tree_matrix(log_2_leaves=log_2_leaves)
  row_additions = tf.zeros(
      shape=[num_extra_rows, bin_tree.shape[1]], dtype=bin_tree.dtype
  )
  return tf.concat([bin_tree, row_additions], axis=0)


def double_h_solution(h_to_double: tf.Tensor) -> tf.Tensor:
  first_rows_of_larger_h = tf.concat(
      [h_to_double, tf.zeros_like(h_to_double)], axis=1
  )
  second_rows_of_larger_h = tf.concat(
      [tf.zeros_like(h_to_double), h_to_double], axis=1
  )
  return tf.concat([first_rows_of_larger_h, second_rows_of_larger_h], axis=0)


class MomentumWithLearningRatesResidual(matrix_factorization_query.OnlineQuery):
  """Residuals the matrix produced by `momentum_sgd_matrix(...)`."""

  def __init__(self, tensor_specs, momentum: float, learning_rates: Any = None):
    """Constructs an efficient implementation of the residual function.

    Args:
      tensor_specs: Specification for the model structure (and momentum buffer).
      momentum: Should match momentum_sgd_matrix.
      learning_rates: Should match momentum_sgd_matrix.
    """
    self._tensor_specs = tensor_specs
    self._momentum = momentum
    if learning_rates is None:
      self._learning_rates = None
    else:
      self._learning_rates = learning_rates

  @tf.function
  def initial_state(self) -> Any:
    zeros = tf.nest.map_structure(
        lambda x: tf.zeros(shape=x.shape, dtype=x.dtype), self._tensor_specs
    )
    return (tf.constant(0, dtype=tf.int32), zeros)

  @tf.function
  def compute_query(self, state, observation):
    """Returns (result, updated_state)."""
    round_num, momentum_buf = state
    momentum_buf = tf.nest.map_structure(
        lambda x: x * tf.cast(self._momentum, x.dtype), momentum_buf
    )
    momentum_buf = tf.nest.map_structure(tf.add, momentum_buf, observation)
    if self._learning_rates is not None:
      # Ensure this is a tensor so we can index using autograph:
      rates = tf.constant(self._learning_rates)
      learning_rate = rates[round_num]
    else:
      learning_rate = tf.constant(1.0)
    result = tf.nest.map_structure(
        lambda x: tf.cast(learning_rate, x.dtype) * x, momentum_buf
    )
    return (result, (round_num + 1, momentum_buf))


def momentum_sgd_matrix(
    num_iters: int, momentum: float, learning_rates: Any = None
) -> np.ndarray:
  """Returns a matrix representing momentum SGD.

  Momentum can be expressed as a linear combination of previous gradients
  (or updates in the case of FL). This function returns a matrix `S` with the
  property that (for a one-dimensional problem on gradient array `g`):
  ```
  theta = -learning_rate * S @ np.array(g[0:t-1])
  ```

  is equivalent to the iterates produced by:

  ```
  theta[0] = 0.0
  momentumb_buf = 0.0
  for t in range(num_iters):
    momentum_buf = momentum * momentum_buf + grads[t]
    theta[t+1] = theta[t] - learning_rate * momentum_buf
  theta = theta[1:]
  ```

  Args:
    num_iters: The number of iterations for which to construct the matrix.
    momentum: The momentum parameter, in [0, 1).
    learning_rates: Optional array of learning rates of length num_iters. Note
      that these learning rates should generally be used to express the desired
      learning rate schedule, but not the overall magnitude, which will be
      provided by the server learning rate. Hence, capping these values at one
      is usually a good practice.
  Returns: A np.ndarray of shape [num_iters, num_iters].
  """
  if learning_rates is None:
    learning_rates = np.ones(num_iters)
  if len(learning_rates) != num_iters:
    raise ValueError(
        'learning_rates must have length equal to num_iters, '
        + f'found {len(learning_rates)} vs {num_iters}'
    )

  if np.min(learning_rates) <= 0.0:
    raise ValueError(
        'Learning rates must be positive (zero learning rates may break fixed '
        f'point factorization into W H. Found {learning_rates}'
    )

  # Banded matrix that computes the t'th value of the momentum buffer
  m_powers = [momentum**i for i in range(num_iters)]
  m_buf_matrix = scipy.sparse.diags(
      m_powers,
      offsets=-np.arange(num_iters, dtype=np.int32),
      shape=(num_iters, num_iters),
  ).toarray()

  # Lower triangular matrix with nonzeros in column i equal to learning_rates[i]
  lr_matrix = np.tri(num_iters) @ np.diag(learning_rates)
  return lr_matrix @ m_buf_matrix
