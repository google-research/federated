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
"""Tests for solvers."""
import itertools

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from dp_matrix_factorization import constraint_builders
from dp_matrix_factorization import matrix_constructors
from dp_matrix_factorization import solvers


def _create_full_lower_triangular_matrix(*, dim) -> tf.Tensor:
  return tf.constant(np.tril(np.ones(shape=(dim, dim))))


def _create_full_lower_triangular_vector(*, dim) -> tf.Tensor:
  return tf.reshape(_create_full_lower_triangular_matrix(dim=dim), [-1])


class ConstrainedSolverTest(tf.test.TestCase, parameterized.TestCase):

  def test_identity_constraints_return_target(self):
    dimensionality = 100
    constraints = tf.eye(dimensionality)
    target = tf.reshape(
        tf.constant(list(range(dimensionality)), dtype=tf.float32), [-1, 1])
    self.assertAllEqual(
        solvers.solve_directly_for_optimal_w(
            constraint_matrix=constraints, target_vector=target), target)

  def test_twodim_binary_tree_h_streaming_returns_honaker_streaming(self):
    log_2_leaves = 1
    binary_tree_h = matrix_constructors.binary_tree_matrix(
        log_2_leaves=log_2_leaves)
    full_target = _create_full_lower_triangular_vector(
        dim=binary_tree_h.shape[1])
    streaming_vars = constraint_builders.compute_flat_vars_for_streaming(
        binary_tree_h)
    vectorized_constraints = constraint_builders.create_vectorized_constraint_matrix(
        binary_tree_h)
    filtered_constraints, filtered_target = constraint_builders.filter_constraints(
        full_constraint_matrix=vectorized_constraints,
        target_vector=full_target,
        variable_mask=streaming_vars)
    optimal_w = solvers.solve_directly_for_optimal_w(
        constraint_matrix=filtered_constraints, target_vector=filtered_target)
    # Flattening of the nonzero elements in the matrix
    # [[1, 0, 0], [1/3, 1/3, 2/3]]
    self.assertAllEqual(optimal_w, [[1.], [1 / 3], [1 / 3], [2 / 3]])

  def test_twodim_binary_tree_h_streaming_returns_honaker_full_batch(self):
    log_2_leaves = 1
    binary_tree_h = matrix_constructors.binary_tree_matrix(
        log_2_leaves=log_2_leaves)
    full_target = _create_full_lower_triangular_vector(
        dim=binary_tree_h.shape[1])
    vectorized_constraints = constraint_builders.create_vectorized_constraint_matrix(
        binary_tree_h)
    optimal_w = solvers.solve_directly_for_optimal_w(
        constraint_matrix=vectorized_constraints, target_vector=full_target)
    # Flattening of the matrix
    # [[-2/3, 1/3, 1/3], [1/3, 1/3, 2/3]]
    self.assertAllEqual(optimal_w,
                        [[2 / 3], [-1 / 3], [1 / 3], [1 / 3], [1 / 3], [2 / 3]])

  @parameterized.named_parameters((str(n) + '_leaves', n) for n in range(1, 6))
  def test_solution_satisfies_constraints_binary_tree_streaming(
      self, log_2_leaves):
    binary_tree_h = matrix_constructors.binary_tree_matrix(
        log_2_leaves=log_2_leaves)
    full_target = _create_full_lower_triangular_vector(
        dim=binary_tree_h.shape[1])
    streaming_vars = constraint_builders.compute_flat_vars_for_streaming(
        binary_tree_h)
    vectorized_constraints = constraint_builders.create_vectorized_constraint_matrix(
        binary_tree_h)
    filtered_constraints, filtered_target = constraint_builders.filter_constraints(
        full_constraint_matrix=vectorized_constraints,
        target_vector=full_target,
        variable_mask=streaming_vars)
    optimal_w = solvers.solve_directly_for_optimal_w(
        constraint_matrix=filtered_constraints, target_vector=filtered_target)
    self.assertAllClose(
        tf.reshape(filtered_constraints @ optimal_w, filtered_target.shape),
        filtered_target)

  @parameterized.named_parameters((str(n) + '_leaves', n) for n in range(1, 6))
  def test_solution_satisfies_constraints_binary_tree_full_batch(
      self, log_2_leaves):
    binary_tree_h = matrix_constructors.binary_tree_matrix(
        log_2_leaves=log_2_leaves)
    full_target = _create_full_lower_triangular_vector(
        dim=binary_tree_h.shape[1])
    vectorized_constraints = constraint_builders.create_vectorized_constraint_matrix(
        binary_tree_h)
    optimal_w = solvers.solve_directly_for_optimal_w(
        constraint_matrix=vectorized_constraints, target_vector=full_target)
    self.assertAllClose(
        tf.reshape(vectorized_constraints @ optimal_w, full_target.shape),
        full_target)

  @parameterized.named_parameters((str(n) + '_seed', n) for n in range(1, 6))
  def test_solution_satisfies_constraints_random_matrix(self, seed):
    tf.random.set_seed(seed)
    dim = 10
    h_matrix = tf.random.normal(shape=[2 * dim, dim], dtype=tf.float64)
    full_target = _create_full_lower_triangular_vector(dim=h_matrix.shape[1])
    vectorized_constraints = constraint_builders.create_vectorized_constraint_matrix(
        h_matrix)
    optimal_w = solvers.solve_directly_for_optimal_w(
        constraint_matrix=vectorized_constraints, target_vector=full_target)
    self.assertAllClose(
        tf.reshape(vectorized_constraints @ optimal_w, full_target.shape),
        full_target)


class FullBatchSolverTest(tf.test.TestCase, parameterized.TestCase):

  def test_identity_h_returns_target(self):
    dimensionality = 10
    h_matrix = tf.eye(dimensionality)
    target = tf.reshape(
        tf.constant(list(range(dimensionality**2)), dtype=tf.float32),
        [dimensionality, dimensionality])
    self.assertAllClose(
        solvers.solve_directly_for_optimal_full_batch_w(
            h_matrix=h_matrix, s_matrix=target), target)

  def test_twodim_binary_tree_h_returns_honaker_full_batch(self):
    log_2_leaves = 1
    binary_tree_h = matrix_constructors.binary_tree_matrix(
        log_2_leaves=log_2_leaves)
    target_matrix = _create_full_lower_triangular_matrix(
        dim=binary_tree_h.shape[1])
    optimal_w = solvers.solve_directly_for_optimal_full_batch_w(
        h_matrix=binary_tree_h, s_matrix=target_matrix)
    # [[2/3, -1/3, 1/3], [1/3, 1/3, 2/3]]
    self.assertAllClose(optimal_w,
                        [[2 / 3, -1 / 3, 1 / 3], [1 / 3, 1 / 3, 2 / 3]])

  @parameterized.named_parameters((str(n) + '_leaves', n) for n in range(1, 6))
  def test_solution_satisfies_constraints_binary_tree(self, log_2_leaves):
    binary_tree_h = matrix_constructors.binary_tree_matrix(
        log_2_leaves=log_2_leaves)
    full_target = _create_full_lower_triangular_matrix(
        dim=binary_tree_h.shape[1])
    optimal_w = solvers.solve_directly_for_optimal_full_batch_w(
        h_matrix=binary_tree_h, s_matrix=full_target)
    self.assertAllClose(
        tf.reshape(optimal_w @ binary_tree_h, full_target.shape), full_target)

  @parameterized.named_parameters((str(n) + '_seed', n) for n in range(1, 6))
  def test_solution_satisfies_constraints_random_matrix(self, seed):
    tf.random.set_seed(seed)
    dim = 10
    h_matrix = tf.random.normal(shape=[2 * dim, dim], dtype=tf.float64)
    full_target = _create_full_lower_triangular_matrix(dim=h_matrix.shape[1])
    optimal_w = solvers.solve_directly_for_optimal_full_batch_w(
        h_matrix=h_matrix, s_matrix=full_target)
    self.assertAllClose(
        tf.reshape(optimal_w @ h_matrix, full_target.shape), full_target)


class PseudoinvSolverTest(tf.test.TestCase, parameterized.TestCase):

  def test_all_entries_available_returns_target(self):
    dimensionality = 100
    h_matrix = tf.eye(dimensionality)
    # Create a matrix of all-True booleans
    constraints = tf.cast(
        tf.ones(shape=[dimensionality, dimensionality]), tf.bool)
    target = tf.ones(shape=[dimensionality, dimensionality])
    self.assertAllEqual(
        solvers.solve_for_constrained_w_with_pseudoinv(
            h_matrix=h_matrix, s_matrix=target,
            w_constraint_matrix=constraints), target)

  def test_twodim_binary_tree_h_streaming_returns_honaker_streaming(self):
    log_2_leaves = 1
    h_matrix = matrix_constructors.binary_tree_matrix(log_2_leaves=log_2_leaves)
    full_target = _create_full_lower_triangular_matrix(dim=h_matrix.shape[1])
    w_constraints = matrix_constructors._compute_matrix_vars_for_streaming(
        h_matrix)
    expected_w = tf.constant([[1., 0, 0], [1 / 3., 1 / 3., 2 / 3.]])
    constructed_w = solvers.solve_for_constrained_w_with_pseudoinv(
        h_matrix, full_target, w_constraints)
    self.assertAllClose(expected_w, constructed_w)

  def test_fourdim_binary_tree_h_streaming_returns_honaker_streaming(self):
    h_matrix = matrix_constructors.binary_tree_matrix(log_2_leaves=2)
    full_target = _create_full_lower_triangular_matrix(dim=h_matrix.shape[1])
    w_constraints = matrix_constructors._compute_matrix_vars_for_streaming(
        h_matrix)
    expected_w = tf.constant(
        [[1., 0., 0., 0., 0., 0., 0.],
         [0.33333333, 0.33333333, 0.66666667, 0., 0., 0., 0.],
         [0.33333333, 0.33333333, 0.66666667, 1., 0., 0., 0.],
         [
             0.14285714, 0.14285714, 0.28571429, 0.14285714, 0.14285714,
             0.28571429, 0.57142857
         ]])
    constructed_w = solvers.solve_for_constrained_w_with_pseudoinv(
        h_matrix, full_target, w_constraints)
    self.assertAllClose(expected_w, constructed_w)

  def test_twodim_binary_tree_h_unconstrained_returns_honaker_full_batch(self):
    log_2_leaves = 1
    h_matrix = matrix_constructors.binary_tree_matrix(log_2_leaves=log_2_leaves)
    full_target = _create_full_lower_triangular_matrix(dim=h_matrix.shape[1])
    # All-True yields unconstrained solution.
    w_constraints = tf.cast(
        tf.ones(shape=[h_matrix.shape[1], h_matrix.shape[0]]), tf.bool)

    expected_w = tf.constant([[2 / 3., -1 / 3, 1 / 3], [1 / 3, 1 / 3, 2 / 3]])
    constructed_w = solvers.solve_for_constrained_w_with_pseudoinv(
        h_matrix, full_target, w_constraints)

    self.assertAllClose(expected_w, constructed_w)

  @parameterized.named_parameters(
      (f'{n}_leaves_streaming_{streaming}', n, streaming)
      for n, streaming in itertools.product(range(1, 6), [True, False]))
  def test_solution_satisfies_constraints_binary_tree(self, log_2_leaves,
                                                      streaming):
    binary_tree_h = matrix_constructors.binary_tree_matrix(
        log_2_leaves=log_2_leaves)
    full_target = _create_full_lower_triangular_matrix(
        dim=binary_tree_h.shape[1])
    if streaming:
      w_constraints = matrix_constructors._compute_matrix_vars_for_streaming(
          binary_tree_h)
    else:
      w_constraints = tf.cast(
          tf.ones(shape=[binary_tree_h.shape[1], binary_tree_h.shape[0]]),
          tf.bool)

    constructed_w = solvers.solve_for_constrained_w_with_pseudoinv(
        binary_tree_h, full_target, w_constraints)

    self.assertAllClose(constructed_w @ binary_tree_h, full_target)

  @parameterized.named_parameters((f'seed_{n}', n) for n in range(1, 6))
  def test_solution_satisfies_constraints_random_matrix(self, seed):
    tf.random.set_seed(seed)
    dim = 10
    h_matrix = tf.random.normal(shape=[2 * dim, dim], dtype=tf.float64)
    full_target = _create_full_lower_triangular_matrix(dim=h_matrix.shape[1])
    w_constraints = tf.cast(
        tf.ones(shape=[h_matrix.shape[1], h_matrix.shape[0]]), tf.bool)

    constructed_w = solvers.solve_for_constrained_w_with_pseudoinv(
        h_matrix, full_target, w_constraints)

    self.assertAllClose(constructed_w @ h_matrix, full_target)


if __name__ == '__main__':
  tf.test.main()
