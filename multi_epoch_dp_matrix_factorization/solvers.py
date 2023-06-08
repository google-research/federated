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
"""Solvers for W given H and S in matrix factorization S = WH."""

import tensorflow as tf


@tf.function
def solve_directly_for_optimal_w(
    *, constraint_matrix: tf.Tensor, target_vector: tf.Tensor
) -> tf.Tensor:
  """Computes optimum for factorized DP prefix sum estimation given fixed H.

  This function implements the closed-form expressions for optimum of a
  constrained quadratic to be found in "An improved closed-form solution for
  the constrained minimization of the root of a quadratic functional"; see
  https://www.sciencedirect.com/science/article/pii/S0377042712001744.

  Args:
    constraint_matrix: A matrix representing all constraints on the vectorized
      version of W constructed from tf.reshape(W, [1, -1]) implied by the
      equation WH = S, and optionally extra constraints on possible nonzero
      elements of W. For `constraint_matrix` `X`, the matrix `X @
      tf.transpose(X)` must be invertible.
    target_vector: A vector representing the flattened version of the matrix S,
      constructed from tf.reshape(S, [-1, 1]).

  Returns:
    A vectorized version of the optimal W matrix.
  """
  # Notice that we have special block-diagonal structure in our
  # `constraint_matrix`; we plan to eventually leverage this for scalability,
  # but preserve this implementation for reference.
  u_matrix = tf.matmul(constraint_matrix, tf.transpose(constraint_matrix))
  # The closed-form expression we use relies on invertibility of the matrix U.
  # If it is not invertible this line will raise.
  inv_u = tf.linalg.inv(u_matrix)
  inv_matmul = tf.transpose(constraint_matrix) @ inv_u
  reshaped_target = tf.reshape(target_vector, [-1, 1])
  return inv_matmul @ reshaped_target


@tf.function
def _construct_e_matrix(
    row_constraint: tf.Tensor, row_dim: tf.Tensor
) -> tf.Tensor:
  r"""Constructs projection operation for constrained pseudoinversion.

  The matrix E constructed is one part of the so-called
  'constrained generalized inverse'; the constrained generalized inverse of A
  with respect to the (axis-aligned) constraints on the solution vector
  represented by the boolean `row_constraint` will be `(EA)^\dagger`.

  For a derivation of the expressions used here, see section 3.6 of Campbell
  and Meyer's 'Generalized Inverses of Linear Transformations'

  Args:
    row_constraint: A rank-1 Boolean tensor indicating which elements may be
      nonzero.
    row_dim: Integer tensor representing the intended number of rows of the
      matrix `A` as described above.

  Returns:
    A rank-2 tensor E as described above.
  """
  num_false_elem = tf.reduce_sum(
      tf.ones(shape=row_constraint.shape, dtype=tf.int32)
      - tf.cast(row_constraint, tf.int32)
  )
  if num_false_elem == 0:
    e_matrix = tf.eye(row_dim)
  else:
    c_matrix = tf.zeros(shape=[num_false_elem, tf.size(row_constraint)])
    col_idx = tf.size(row_constraint) - num_false_elem
    updates = [1]
    for row_idx in range(num_false_elem):
      indices = [[row_idx, col_idx]]
      c_matrix = tf.tensor_scatter_nd_add(c_matrix, indices, updates)
      col_idx += 1
    e_matrix = tf.eye(row_dim) - tf.linalg.pinv(c_matrix) @ c_matrix
  return e_matrix


@tf.function
def solve_for_constrained_w_row_with_pseudoinv(
    h_matrix: tf.Tensor, s_row: tf.Tensor, row_constraint: tf.Tensor
) -> tf.Tensor:
  r"""Solves for a single row of W with minimal Frobenius norm.

  Args:
    h_matrix: rank-2 tensor representing the term H in the factorization WH = S.
    s_row: rank-1 tensor representing the term S in the factorization WH = S.
    row_constraint: rank-1 tensor of Booleans representing constraints on the
      matrix W.

  Returns:
    The matrix W above; the unique rank-2 tensor of minimal Frobenius norm
    satisfying WH = S, with all elements corresponding to entries in
    constraint_matrix which are False guaranteed to be zero.
  """
  e_matrix = _construct_e_matrix(row_constraint, h_matrix.shape[0])
  row_soln = tf.squeeze(
      tf.reshape(tf.cast(s_row, h_matrix.dtype), [1, -1])
      @ tf.linalg.pinv(tf.cast(e_matrix, h_matrix.dtype) @ h_matrix)
  )
  return row_soln


@tf.function
def solve_for_constrained_w_with_pseudoinv(
    h_matrix: tf.Tensor, s_matrix: tf.Tensor, w_constraint_matrix: tf.Tensor
) -> tf.Tensor:
  r"""Solves for W of minimal Frobenius norm satisfying WH = S, subject to Boolean constraints.

  This implementation leverages the
  goemetric properties of the Moore-Penrose pseudoinverse  to reduce complexity
  in computing the W of minimal Frobenius norm. In particular, this algorithm
  essentially translates Campbell & Meyer's Thm 3.6.3 to our setting.

  Args:
    h_matrix: rank-2 tensor representing the term H in the factorization WH = S.
    s_matrix: rank-2 tensor representing the term S in the factorization WH = S.
    w_constraint_matrix: rank-2 tensor of Booleans representing constraints on
      the matrix W.

  Returns:
    The matrix W above; the unique rank-2 tensor of minimal Frobenius norm
    satisfying WH = S, with all elements corresponding to entries in
    constraint_matrix which are False guaranteed to be zero.
  """
  # We construct W row-by-row.
  w_rows = []
  for outer_row_idx in range(w_constraint_matrix.shape[0]):
    row_soln = solve_for_constrained_w_row_with_pseudoinv(
        h_matrix,
        s_matrix[outer_row_idx, :],
        w_constraint_matrix[outer_row_idx, :],
    )
    w_rows.append(row_soln)
  # Stack the rows into a single matrix.
  return tf.stack(w_rows)


@tf.function
def solve_directly_for_optimal_full_batch_w(
    h_matrix: tf.Tensor, s_matrix: tf.Tensor
) -> tf.Tensor:
  r"""Solves for W of minimal Frobenius norm satisfying WH = S.

  The resulting W matrix may be dense; this implementation leverages the
  goemetric properties of the Moore-Penrose pseudoinverse, along with the
  symmetry of the full-batch problem, to reduce complexity in computing the W
  of minimal Frobenius norm. In particular, for a vector x satisfying xA = B,
  the l^2 norm of x is at least the l^2 norm of BA^\dagger; IE, the
  Moore-Penrose pseudoinverse yields the *minimal* solution of the problem
  xA = B in l^2 norm. See Ch 2 of Campbell and Meyer's
  _Generalized Inverses of Linear Transformations_ for further discussion.

  Full-batch factorization can leverage this observation directly by applying
  it to each of the rows of W, interpreted as vectors.

  Args:
    h_matrix: rank-2 tensor representing the term H in the factorization WH = S.
    s_matrix: rank-2 tensor representing the term S in the factorization WH = S.

  Returns:
    The matrix W above; the unique rank-2 tensor of minimal Frobenius norm
    satisfying WH = S.
  """
  return tf.reshape(
      tf.cast(s_matrix, h_matrix.dtype), [-1, h_matrix.shape[1]]
  ) @ tf.linalg.pinv(h_matrix)
