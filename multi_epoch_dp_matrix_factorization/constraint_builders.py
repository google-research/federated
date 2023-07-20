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
"""Library for building constraint matrices for vectorized versions of prefix-sum matrix factorixations."""
from typing import Union

import numpy as np
import tensorflow as tf


@tf.function
def create_vectorized_constraint_matrix(h_matrix: tf.Tensor) -> tf.Tensor:
  """Construct a constraint matrix for a vectorized full-matrix factorization.

  Given a matrix H, the matrix M returned from this function can be used to
  represent the linear set of constraints

  W @ H = S

  for square S. That is, the matrix expression above is satisfied precisely
  when the vector-matrix equation

  M @ tf.reshape(W, [-1, 1]) = tf.reshape(S, [-1, 1])

  is satisfied.

  Further constraints on the matrix W (e.g., that only some of its elements
  are nonzero) can be represented by a postprocessing of the returned matrix M.

  Args:
    h_matrix: Matrix defining the right-hand side of the factorization problem
      described above.

  Returns:
    A matrix M representing a vectorized version of the constraints represented
    by the matrix factorization problem, as described above.
  """
  h_column_dim = h_matrix.shape[1]
  blocks = [
      tf.linalg.LinearOperatorFullMatrix(tf.transpose(h_matrix))
      for _ in range(h_column_dim)
  ]
  block_diag = tf.linalg.LinearOperatorBlockDiag(blocks)
  return block_diag.to_dense()


def _convert_matrix_to_numpy(
    h_matrix: Union[tf.Tensor, np.ndarray]
) -> np.ndarray:
  """Verifies argument is rank 2, and converts to Numpy."""
  if hasattr(h_matrix, 'numpy'):
    h_matrix = h_matrix.numpy()
  elif not isinstance(h_matrix, np.ndarray):
    raise TypeError(
        'Expected eager tensor or numpy ndarray; encountered '
        f'unknown h_matrix argument type: {type(h_matrix)}'
    )
  if len(h_matrix.shape) != 2:
    raise ValueError(
        f'Expected a rank-2 ndarray or tensor; found shape {h_matrix.shape}'
    )
  return h_matrix


def compute_observations_available(
    h_matrix: Union[tf.Tensor, np.ndarray]
) -> list[np.generic]:
  """Computes first index an observation is available.

  Assumes lower-triangular structure of the desired matrix.

  That is, assumes that if an element in row i is nonzero in column j of
  `h_matrix`, then all elements in rows k for k >= i may be assumed to be
  potentially nonzero.

  This assumption is reflected in the monotonically nondecreasing nature of the
  returned list.


  Args:
    h_matrix: H matrix of factorization problem.

  Returns:
    A monotonically nondecreasing list of indices, representing the time steps
    after which the rows of `h_matrix` are fully available, subject to the
    lower-triangular assumption described above.
  """
  h_matrix = _convert_matrix_to_numpy(h_matrix)
  num_measurements = h_matrix.shape[0]
  # Compute observation index where the necessary elements of this measurement
  # have all materialized.
  nonzero_elements = [
      np.nonzero(h_matrix[i, :]) for i in range(num_measurements)
  ]

  observation_idx_available = [np.max(x, initial=-1) for x in nonzero_elements]

  monotonic_availability = []
  last_val = -1
  for elem in observation_idx_available:
    new_elem = max(last_val, elem)
    monotonic_availability.append(new_elem)
    last_val = new_elem
  return monotonic_availability


def compute_flat_vars_for_streaming(
    h_matrix: Union[tf.Tensor, np.ndarray]
) -> list[bool]:
  """Computes a list of booleans indicating nonzero entries for a flattened W.

  That is, for an incoming `h_matrix`, the lower-diagonal structure of this
  matrix determines a compatibility with streaming algorithmic requirements.
  For a row i, the rightmost nonzero column j of this row introduces the
  requirement that precisely j entries of the vector x = (x_1, ... x_n) must be
  available before element i of the product h_matrix @ x can be computed
  (assuming the x_i materialize sequentially in time).

  This function computes these requirements, producing a list of booleans
  corresponding to the elements of a flattened matrix W, flattened by using
  column as the quickly varying index (which is standard in reshape and
  flatten implementations). If an index is `True` in this list, the
  corresponding element of the `W` matrix may be nonzero; if it is `False`,
  the corresponding element must be zero. If a matrix `W` is constructed which
  satisfies these constraints, computing the ith element of the vector `WHx`
  requires that at most `i` of the elements of the vector `x` have
  materialized; that is, that the product `WHx` can be computed in a streaming
  fashion.


  Args:
    h_matrix: two-dimensional numpy ndarray or eager TensorFlow tensor, used to
      compute streaming requirements for the matrix `W` as described above.

  Returns:
    A list of Booleans, representing the constraints on the matrix W as
    specified above.
  """
  h_matrix = _convert_matrix_to_numpy(h_matrix)
  observation_idx_available = compute_observations_available(h_matrix)
  streaming_mask = []
  num_measurements, num_observations = h_matrix.shape
  for i in range(num_observations):
    for j in range(num_measurements):
      if observation_idx_available[j] <= i:
        streaming_mask.append(True)
      else:
        streaming_mask.append(False)
  return streaming_mask


def filter_constraints(
    *,
    full_constraint_matrix: tf.Tensor,
    target_vector: tf.Tensor,
    variable_mask: list[bool],
) -> tuple[tf.Tensor, tf.Tensor]:
  """Filters constraint matrix and target to reflect dropping of variables.

  That is, for some instantiations of our matrix-factorization problem, we wish
  to fix entries of the left-hand matrix W to be zero. There are several ways we
  might accomplish this. One might be to extend the constraint matrix with
  appropriate rows containing a single one, and extend the target vector with
  zeros. This approach, however, destroys invertibility of the symmetrized
  matrix (B @ B.T for B the matrix constructed here).

  We take here an alternative approach, where we drop the dimensionality of the
  problem by deleting the columns in the full constraint matrix which
  correspond to the dropped variables. Performing this operation may result in
  rows which themselves are all 0s; these must correspond to zero elements in
  the target if the factorization problem is solvable. To preserve the
  invertibility of the symmetrized matrix, we drop these rows.

  We conjecture that if we begin with an invertible full_constraint_matrix
  and variable mask computed from this constraint matrix via
  `compute_flat_vars_for_streaming`, this operation preserves invertibility
  of the symmetrized matrix. We are yet to prove this statement, but it seems
  to hold empirically.

  Args:
    full_constraint_matrix: A matrix representing all constraints on the
      flattened LHS of a matrix factorization problem, as returned from
      `create_vectorized_constraint_matrix`. Represents for that reason a matrix
      which can applied via multiplication on the right to a flattened version
      of the matrix W.
    target_vector: A flattened version of the target matrix in the factorization
      problem.
    variable_mask: A list of booleans, corresponding to flattened LHS of our
      factorization problem. An entry in this list is true precisely when the
      associated element of the matrix can be nonzero.

  Returns:
    A filtered version of the constraint matrix and target vector in a tuple
    (matrix first), as described above.
  """
  # That this filtering + dropping rows maintains the invariant B @ B.T
  # invertible is empirically true, but I have no proof of it.
  columns_deleted = tf.experimental.numpy.compress(
      np.array(variable_mask), full_constraint_matrix, axis=1
  )
  column_max = tf.reduce_max(tf.math.abs(columns_deleted), axis=1)
  zero_row_filter = tf.greater(column_max, 0)
  filtered_s = tf.boolean_mask(target_vector, zero_row_filter)
  filtered_matrix = tf.boolean_mask(columns_deleted, zero_row_filter)
  return filtered_matrix, filtered_s
