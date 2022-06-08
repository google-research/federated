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
"""Tests for matrix_constructors."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from dp_matrix_factorization import matrix_constructors


class BinaryTreeTest(parameterized.TestCase):

  def test_raises_negative_log(self):
    with self.assertRaises(ValueError):
      matrix_constructors.binary_tree_matrix(log_2_leaves=-1)

  @parameterized.named_parameters(
      ('one_leaf', 0, np.array([[1]])),
      ('two_leaves', 1, np.array([[1, 0], [0, 1], [1, 1]])),
      ('four_leaves', 2,
       np.array([
           [1, 0, 0, 0],
           [0, 1, 0, 0],
           [1, 1, 0, 0],
           [0, 0, 1, 0],
           [0, 0, 0, 1],
           [0, 0, 1, 1],
           [1, 1, 1, 1],
       ])),
      ('eight_leaves', 3,
       np.array([
           [1, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0, 0],
           [1, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 1, 1, 0, 0, 0, 0],
           [1, 1, 1, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 1],
           [0, 0, 0, 0, 0, 0, 1, 1],
           [0, 0, 0, 0, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1],
       ])),
  )
  def test_returns_expected_results(self, log_2_leaves, expected_matrix):
    tree_matrix = matrix_constructors.binary_tree_matrix(
        log_2_leaves=log_2_leaves)
    np.testing.assert_array_equal(expected_matrix, tree_matrix)

  @parameterized.named_parameters(
      ('sixteen_leaves', 4),
      ('1024_leaves', 10),
      ('4096_leaves', 12),
  )
  def test_returns_matrix_with_expected_column_sums(self, log_2_leaves):
    # The expected column sum is identical here to the height of the
    # constructed binary tree, which is the log-base-two of the number of leaves
    # plus 1 (e.g., for a binary tree of 1 element, 2 ** 0 = 1, and the tree is
    # of height 1).
    expected_column_sum = log_2_leaves + 1

    tree_matrix = matrix_constructors.binary_tree_matrix(
        log_2_leaves=log_2_leaves)
    self.assertLen(tree_matrix.shape, 2)
    for column_idx in range(tree_matrix.shape[1]):
      column_sum = np.sum(tree_matrix[:, column_idx])
      self.assertEqual(expected_column_sum, column_sum)


class RandomBinaryTreeTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('sixteen_leaves', 4),
      ('thirty_two_leaves', 5),
      ('sixty_four_leaves', 6),
  )
  def test_returns_appropriate_shape_and_type(self, log_2_leaves):
    bin_tree_matrix = matrix_constructors.binary_tree_matrix(
        log_2_leaves=log_2_leaves)
    random_tree_matrix = matrix_constructors.random_normal_binary_tree_structure(
        log_2_leaves=log_2_leaves)
    self.assertEqual(bin_tree_matrix.shape, random_tree_matrix.shape)
    self.assertEqual(bin_tree_matrix.dtype, random_tree_matrix.dtype)

  @parameterized.named_parameters(
      ('one_leaf', 0, np.array([[True]])),
      ('two_leaves', 1, np.array([[True, False], [True, True], [True, True]])),
      ('four_leaves', 2,
       np.array([
           [True, False, False, False],
           [True, True, False, False],
           [True, True, False, False],
           [True, True, True, False],
           [True, True, True, True],
           [True, True, True, True],
           [True, True, True, True],
       ])),
  )
  def test_returns_matrix_with_appropriate_elements_zero(
      self, log_2_leaves, expected_mask):
    random_tree_matrix = matrix_constructors.random_normal_binary_tree_structure(
        log_2_leaves=log_2_leaves)
    actual_mask = matrix_constructors._compute_h_mask(random_tree_matrix)

    self.assertLen(random_tree_matrix.shape, 2)
    np.testing.assert_array_equal(expected_mask, actual_mask)


class MatrixStreamingConstraintsTest(tf.test.TestCase, parameterized.TestCase):

  def test_raises_with_vector(self):
    with self.assertRaises(ValueError):
      matrix_constructors._compute_matrix_vars_for_streaming(
          tf.ones(shape=[10]))

  def test_zero_h_matrix_gives_all_true(self):
    matrix_dim = 10
    zeros = tf.zeros(shape=[matrix_dim, matrix_dim])
    all_true = tf.constant(True, shape=zeros.shape)
    matrix_vars = matrix_constructors._compute_matrix_vars_for_streaming(zeros)
    self.assertAllEqual(matrix_vars, all_true)

  def test_extended_binary_tree_gives_expected_result(self):
    log_2_leaves = 1
    extended_bin_tree = matrix_constructors.extended_binary_tree(
        log_2_leaves=log_2_leaves, num_extra_rows=1)
    expected_result = tf.constant([[True, False, False, False],
                                   [True, True, True, True]])
    matrix_vars = matrix_constructors._compute_matrix_vars_for_streaming(
        extended_bin_tree)
    self.assertAllEqual(expected_result, matrix_vars)

  def test_identity_constraints_give_lower_triangular_all_true(self):
    matrix_dim = 10
    identity = tf.eye(matrix_dim)
    tril = tf.constant(True, shape=identity.shape)
    lower_triangular_true = tf.linalg.LinearOperatorLowerTriangular(
        tril).to_dense()
    matrix_vars = matrix_constructors._compute_matrix_vars_for_streaming(
        identity)
    self.assertAllEqual(matrix_vars, lower_triangular_true)

  def test_dense_constraints_yields_only_final_row_true(self):
    matrix_dim = 10
    dense_matrix = tf.ones(shape=[matrix_dim, matrix_dim])
    matrix_vars = matrix_constructors._compute_matrix_vars_for_streaming(
        dense_matrix)
    # All entries of `dense_matrix @ x` depend on all elements of `x`; therefore
    # only the last row of `W` may be nonzero.
    expected_true_matrix = [[False] * matrix_dim] * (matrix_dim - 1) + [
        [True] * matrix_dim
    ]
    self.assertAllEqual(matrix_vars, expected_true_matrix)

  def test_zero_constraints_yields_all_true(self):
    matrix_dim = 10
    zero_matrix = tf.zeros(shape=[matrix_dim, matrix_dim])
    matrix_vars = matrix_constructors._compute_matrix_vars_for_streaming(
        zero_matrix)
    expected_true_matrix = [[True] * matrix_dim] * matrix_dim
    self.assertAllEqual(matrix_vars, expected_true_matrix)

  def test_binary_tree_h_yields_expected_structure(self):
    h_matrix = matrix_constructors.binary_tree_matrix(log_2_leaves=2)
    matrix_vars = matrix_constructors._compute_matrix_vars_for_streaming(
        h_matrix)
    expected_true_matrix = [
        [True, False, False, False, False, False, False],
        [True, True, True, False, False, False, False],
        [True, True, True, True, False, False, False],
        [True, True, True, True, True, True, True],
    ]
    self.assertAllEqual(matrix_vars, expected_true_matrix)

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
    matrix_vars = matrix_constructors._compute_matrix_vars_for_streaming(
        dense_matrix)

    # Once we encounter a row of 1s in H, we must force the columns of W to be
    # zero until the last row. All others can be True. This is due to the
    # lower-triangular assumption.
    expected_true_list = [[
        True if i < row_to_insert_ones else False for i in range(matrix_dim)
    ]] * (matrix_dim - 1) + [[True] * matrix_dim]
    self.assertAllEqual(matrix_vars, expected_true_list)

  def _compute_momentum_iterates(self, gradients, momentum, learning_rates):
    # Directly compute iterates using the standard momentum algorithm
    thetas = [0.0]
    momentum_buf = 0.0
    for i in range(len(gradients)):
      # Note: Some implementations scale the RH term by (1 - momentum) I think.
      momentum_buf = momentum * momentum_buf + gradients[i]
      thetas.append(thetas[-1] - learning_rates[i] * momentum_buf)
    return np.array(thetas[1:])

  @parameterized.product(momentum=[0.0, 0.85, 0.99])
  def test_momentum_matrix_no_learning_rates(self, momentum):
    gradients = np.array([0.1] * 4 + [0.3] * 5 + [-0.2] * 6 + [0.0] * 2)
    n = len(gradients)
    direct_iterates = self._compute_momentum_iterates(
        gradients, momentum, learning_rates=np.ones(n))
    s_matrix = matrix_constructors.momentum_sgd_matrix(n, momentum)
    s_iterates = -s_matrix @ gradients
    np.testing.assert_allclose(direct_iterates, s_iterates)

  @parameterized.product(momentum=[0.0, 0.85, 0.99])
  def test_momentum_matrix_learning_rates(self, momentum):
    gradients = np.array([0.1] * 4 + [0.3] * 5 + [-0.2] * 6 + [0.0] * 2)
    n = len(gradients)
    learning_rates = np.linspace(1.0, 0.05, num=n)
    direct_iterates = self._compute_momentum_iterates(gradients, momentum,
                                                      learning_rates)
    s_matrix = matrix_constructors.momentum_sgd_matrix(n, momentum,
                                                       learning_rates)
    s_iterates = -s_matrix @ gradients
    np.testing.assert_allclose(direct_iterates, s_iterates)

  @parameterized.product(
      momentum=[0.0, 0.85, 0.99], learning_rate_str=['none', 'decreasing'])
  def test_momentum_residuals(self, momentum, learning_rate_str):
    gradients = np.array([0.1] * 4 + [0.3] * 5 + [-0.2] * 6 + [0.0] * 2)
    n = len(gradients)
    if learning_rate_str == 'none':
      learning_rates = None
    else:
      learning_rates = np.linspace(1.0, 0.05, num=n)
    s_matrix = matrix_constructors.momentum_sgd_matrix(n, momentum,
                                                       learning_rates)
    iterates = s_matrix @ gradients
    expected_residuals = (iterates - np.concatenate([[0.0], iterates[:-1]]))

    # We transform gradients into a tuple of values to
    # test support for structures, scaling the 2nd entry by 2x.
    ts = tf.TensorSpec([], dtype=tf.float64)
    query = matrix_constructors.MomentumWithLearningRatesResidual(
        (ts, ts), momentum, learning_rates)

    residuals = []
    state = query.initial_state()
    for i in range(n):
      result, state = query.compute_query(state,
                                          (gradients[i], 2 * gradients[i]))
      np.testing.assert_allclose(result[0], result[1] / 2)
      residuals.append(result[0])
    residuals = np.array(residuals)

    np.testing.assert_allclose(expected_residuals, residuals, rtol=1e-5)


if __name__ == '__main__':
  absltest.main()
