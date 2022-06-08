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
"""Tests for constraint_builders."""
from absl.testing import parameterized
import tensorflow as tf

from dp_matrix_factorization import constraint_builders


def _get_constraint_factorization_parameters(seed: int,
                                             num_parameters: int = 5):
  """Returns name, h_matrix, s_matrix tuples for testing constraint generation.

  For some set of dimensionalities, this function returns an invertible matrix
  H, along with a dimension-compatible matrix S. Since H is invertible, there
  is a unique solution W to WH = S; that is, W = SH^{-1}.

  Args:
    seed: Integer representing seed to use for matrix generation.
    num_parameters: The number of distinct parameters to generate.

  Returns:
    A list of three-tuples, first element being a descriptive name, with the
    second and third elements representing the matrices H and S described above.
  """
  tf.random.set_seed(seed)
  tuples = []
  for idx in range(1, num_parameters + 1):
    name = f'h_dim_{idx}x{idx}_s_dim_{idx}x{idx}'
    s_matrix = tf.random.normal(shape=[idx, idx])
    # h_matrix will be invertible with probability 1
    h_matrix = tf.random.normal(shape=[idx, idx])
    tuples.append((name, h_matrix, s_matrix))
  return tuples


class VectorizedConstraintMatrixTest(tf.test.TestCase, parameterized.TestCase):

  def test_constructs_identity_from_identity(self):
    matrix_dim = 10
    identity = tf.eye(matrix_dim)
    constraint_matrix = constraint_builders.create_vectorized_constraint_matrix(
        identity)
    # The vectorized version of the implied constraint equation has n ** 2
    # variables. The equation itself is flatten(W) * I = flatten(S), since
    # multiplication by the identity matrix leaves every element in W alone.
    high_dim_identity = tf.eye(matrix_dim**2)
    self.assertAllClose(constraint_matrix, high_dim_identity)

  def test_vectorizing_nonsquare_constraints(self):
    # If W is nxd, and H is dxn, then S is nxn.
    num_columns_h = 10
    num_rows_h = 20
    num_rows_w = num_columns_h
    num_columns_w = num_rows_h

    tril = tf.constant(1., shape=[num_columns_h, num_columns_h])
    lower_triangular_ones = tf.linalg.LinearOperatorLowerTriangular(
        tril).to_dense()
    constant_ones = tf.constant(
        1., shape=[num_rows_h - lower_triangular_ones.shape[0], num_columns_h])
    nonsquare_constraints = tf.concat([lower_triangular_ones, constant_ones],
                                      axis=0)

    vectorized_constraints = constraint_builders.create_vectorized_constraint_matrix(
        nonsquare_constraints)
    # If W is nxd, and H is dxn, then S is nxn.
    flat_target = tf.ones(shape=[num_rows_w * num_columns_h, 1])
    # With nonsquare constraints, we take a pseudoinverse.
    flat_w = tf.linalg.pinv(vectorized_constraints) @ flat_target
    reshaped_w = tf.reshape(flat_w, shape=[num_rows_w, num_columns_w])
    reshaped_target = tf.reshape(flat_target, shape=[num_rows_w, num_columns_h])
    computed_target = reshaped_w @ nonsquare_constraints
    self.assertAllClose(computed_target, reshaped_target)

  @parameterized.named_parameters(
      *_get_constraint_factorization_parameters(seed=0))
  def test_solution_for_matrix_equation_solves_vector_equation(
      self, h_matrix, s_matrix):
    unique_w = s_matrix @ tf.linalg.inv(h_matrix)
    constraint_matrix = constraint_builders.create_vectorized_constraint_matrix(
        h_matrix)
    flat_w = tf.reshape(unique_w, [-1, 1])
    flat_s = tf.reshape(s_matrix, [-1, 1])
    self.assertAllEqual(constraint_matrix.shape,
                        [flat_w.shape[0], flat_s.shape[0]])
    self.assertAllClose(constraint_matrix @ flat_w, flat_s, atol=1e-5)

  @parameterized.named_parameters(
      *_get_constraint_factorization_parameters(seed=1))
  def test_solution_for_vector_equation_solves_matrix_equation(
      self, h_matrix, s_matrix):
    flat_s = tf.reshape(s_matrix, [-1, 1])
    constraint_matrix = constraint_builders.create_vectorized_constraint_matrix(
        h_matrix)
    unique_flat_w = tf.linalg.inv(constraint_matrix) @ flat_s
    reconstructed_w = tf.reshape(unique_flat_w,
                                 [s_matrix.shape[1], h_matrix.shape[0]])
    self.assertAllClose(reconstructed_w @ h_matrix, s_matrix, atol=1e-5)


class StreamingConstraintsTest(tf.test.TestCase, parameterized.TestCase):

  def test_raises_with_vector(self):
    with self.assertRaises(ValueError):
      constraint_builders.compute_flat_vars_for_streaming(tf.ones(shape=[10]))

  def test_identity_constraints_give_lower_triangular_all_true(self):
    matrix_dim = 10
    identity = tf.eye(matrix_dim)
    tril = tf.constant(True, shape=identity.shape)
    lower_triangular_true = tf.linalg.LinearOperatorLowerTriangular(
        tril).to_dense()
    flat_lower_triangular_true = tf.reshape(lower_triangular_true, [-1])
    split_lower_triangular_true = tf.squeeze(
        tf.split(flat_lower_triangular_true, num_or_size_splits=matrix_dim**2))
    flat_vars = constraint_builders.compute_flat_vars_for_streaming(identity)
    self.assertAllEqual(flat_vars, split_lower_triangular_true)

  def test_dense_constraints_yields_only_final_row_true(self):
    matrix_dim = 10
    dense_matrix = tf.ones(shape=[matrix_dim, matrix_dim])
    flat_vars = constraint_builders.compute_flat_vars_for_streaming(
        dense_matrix)
    # All entries of `dense_matrix @ x` depend on all elements of `x`; therefore
    # only the last row of `W` may be nonzero.
    expected_true_list = [False for _ in range(matrix_dim)
                         ] * (matrix_dim - 1) + [True] * matrix_dim
    self.assertAllEqual(flat_vars, expected_true_list)

  def test_zero_constraints_yields_all_true(self):
    matrix_dim = 10
    zero_matrix = tf.zeros(shape=[matrix_dim, matrix_dim])
    flat_vars = constraint_builders.compute_flat_vars_for_streaming(zero_matrix)
    expected_true_list = [True for _ in range(matrix_dim**2)]
    self.assertAllEqual(flat_vars, expected_true_list)

  @parameterized.named_parameters(
      tuple((str(i) + 'th_row_ones', i)) for i in range(10))
  def test_dense_constraints(self, row_to_insert_ones):
    matrix_dim = 10
    dense_zero_matrix = tf.zeros(shape=[matrix_dim, matrix_dim])
    scattered_ones = tf.scatter_nd(
        indices=[[row_to_insert_ones]],
        updates=tf.ones(shape=[1, matrix_dim]),
        shape=dense_zero_matrix.shape)
    dense_matrix = dense_zero_matrix + scattered_ones
    flat_vars = constraint_builders.compute_flat_vars_for_streaming(
        dense_matrix)

    # Once we encounter a row of 1s in H, we must force the columns of W to be
    # zero until the last row.
    expected_true_list = [
        True if i < row_to_insert_ones else False for i in range(matrix_dim)
    ] * (matrix_dim - 1) + [True] * matrix_dim
    self.assertAllEqual(flat_vars, expected_true_list)


class FilterConstraintsTest(tf.test.TestCase, parameterized.TestCase):

  def test_all_false_variable_mask_returns_empty_tensors(self):
    dimensionality = 10
    constraint_matrix = tf.eye(dimensionality)
    target_vector = tf.ones(shape=[10])
    variable_mask = [False for _ in range(dimensionality)]
    filtered_matrix, filtered_target = constraint_builders.filter_constraints(
        full_constraint_matrix=constraint_matrix,
        target_vector=target_vector,
        variable_mask=variable_mask)
    self.assertEqual(filtered_matrix.shape, [0, 0])
    self.assertEqual(filtered_target.shape, [0])

  @parameterized.named_parameters(
      tuple((str(i) + 'th_variable_true', i)) for i in range(10))
  def test_one_variable_true_preserves_only_one_column(self, idx):
    dimensionality = 10
    # Build a matrix with 1, ...10 in columns idx.
    column_tensor = tf.constant([list(range(1, dimensionality + 1))])
    constraint_matrix = tf.transpose(
        tf.scatter_nd(
            indices=[[idx]],
            updates=column_tensor,
            shape=[dimensionality, dimensionality]))
    target_vector = tf.ones(shape=[10])
    variable_mask = [False] * idx + [True] + [False] * (
        dimensionality - idx - 1)
    filtered_matrix, filtered_target = constraint_builders.filter_constraints(
        full_constraint_matrix=constraint_matrix,
        target_vector=target_vector,
        variable_mask=variable_mask)
    self.assertEqual(filtered_matrix.shape, [10, 1])
    self.assertAllEqual(filtered_matrix, tf.reshape(column_tensor, [10, 1]))
    self.assertEqual(filtered_target.shape, [10])
    self.assertAllEqual(filtered_target, tf.ones(shape=[10]))

  @parameterized.named_parameters(
      tuple((str(i) + 'th_permutation', i)) for i in range(10))
  def test_returned_matrix_symmetrized_remains_invertible(
      self, permutation_seed):
    matrix_dim = 10
    identity = tf.eye(matrix_dim)
    permutation = tf.random.shuffle(identity, seed=permutation_seed)
    flat_vars = constraint_builders.compute_flat_vars_for_streaming(permutation)
    target = tf.ones(matrix_dim**2)
    constraint_matrix = constraint_builders.create_vectorized_constraint_matrix(
        permutation)
    filtered_matrix, _ = constraint_builders.filter_constraints(
        full_constraint_matrix=constraint_matrix,
        target_vector=target,
        variable_mask=flat_vars)
    # Doesn't raise
    tf.linalg.inv(filtered_matrix @ tf.transpose(filtered_matrix))
    self.assertAllClose(filtered_matrix @ tf.transpose(filtered_matrix),
                        tf.eye(filtered_matrix.shape[0]))


if __name__ == '__main__':
  tf.test.main()
