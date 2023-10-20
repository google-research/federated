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
"""Tests for loops."""

from absl.testing import parameterized
from jax import config
from jax import numpy as jnp
import numpy as np
import tensorflow as tf

from dp_matrix_factorization import constraint_builders
from dp_matrix_factorization import loops
from dp_matrix_factorization import matrix_constructors

config.update('jax_enable_x64', True)


def _make_prefixsum_s(dimensionality) -> tf.Tensor:
  return tf.constant(np.tril(np.ones(shape=(dimensionality, dimensionality))))


class LoopsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('random_true', True),
      ('random_false', False),
  )
  def test_streaming_results_respect_streaming_constraints(self, random_init):
    log_2_observations = 3
    s_matrix = _make_prefixsum_s(2**log_2_observations)
    if not random_init:
      initial_h = matrix_constructors.binary_tree_matrix(
          log_2_leaves=log_2_observations)
    else:
      initial_h = matrix_constructors.random_normal_binary_tree_structure(
          log_2_leaves=log_2_observations)
    solution = loops.learn_h_sgd(
        initial_h=initial_h,
        s_matrix=s_matrix,
        n_iters=10,
        optimizer=tf.keras.optimizers.SGD(0.1),
        streaming=True)
    binary_tree_h = matrix_constructors.binary_tree_matrix(
        log_2_leaves=log_2_observations)
    h_mask = matrix_constructors._compute_h_mask(binary_tree_h)
    streaming_vars_for_w = constraint_builders.compute_flat_vars_for_streaming(
        binary_tree_h)

    flat_w = np.reshape(solution['W'], [-1])
    for idx, value in enumerate(streaming_vars_for_w):
      if not value:
        self.assertEqual(flat_w[idx], 0.)

    disallowed_h_element_indicator = np.ones(shape=h_mask.shape) - h_mask
    self.assertEqual(np.max(disallowed_h_element_indicator * solution['H']), 0)

  @parameterized.named_parameters(
      ('streaming_false', False),
      ('streaming_true', True),
  )
  def test_factorizes_prefix_sum_matrix_single_iteration(self, streaming):
    log_2_observations = 1
    expected_s = _make_prefixsum_s(2**log_2_observations)
    initial_h = matrix_constructors.binary_tree_matrix(
        log_2_leaves=log_2_observations)
    solution = loops.learn_h_sgd(
        initial_h=initial_h,
        s_matrix=expected_s,
        n_iters=1,
        optimizer=tf.keras.optimizers.SGD(0),
        streaming=streaming)
    self.assertAllClose(solution['W'] @ solution['H'], expected_s)

  @parameterized.named_parameters(
      ('streaming_false', False),
      ('streaming_true', True),
  )
  def test_reduces_loss_while_factorizing_prefix_sum_small_matrix(
      self, streaming):
    log_2_observations = 1
    s_matrix = _make_prefixsum_s(2**log_2_observations)
    initial_h = matrix_constructors.binary_tree_matrix(
        log_2_leaves=log_2_observations)
    expected_s = np.array([[1., 0.], [1., 1.]])
    solution = loops.learn_h_sgd(
        initial_h=initial_h,
        s_matrix=s_matrix,
        n_iters=10,
        optimizer=tf.keras.optimizers.SGD(0.001),
        streaming=streaming)
    self.assertAllClose(solution['W'] @ solution['H'], expected_s)
    self.assertLess(solution['loss_sequence'][-1], solution['loss_sequence'][0])

  @parameterized.named_parameters(
      ('streaming_false', False),
      ('streaming_true', True),
  )
  def test_reduces_loss_while_factorizing_prefix_sum_medium_matrix(
      self, streaming):
    log_2_observations = 5
    expected_s = _make_prefixsum_s(2**log_2_observations)
    initial_h = matrix_constructors.binary_tree_matrix(
        log_2_leaves=log_2_observations)
    solution = loops.learn_h_sgd(
        initial_h=initial_h,
        s_matrix=expected_s,
        n_iters=10,
        optimizer=tf.keras.optimizers.SGD(0.0001),
        streaming=streaming)
    self.assertAllClose(solution['W'] @ solution['H'], expected_s)
    self.assertLess(solution['loss_sequence'][-1], solution['loss_sequence'][0])


class FixedPointIterationsTest(tf.test.TestCase, parameterized.TestCase):

  def test_factorization_factorizes_target(self):
    log_2_observations = 1
    target_s = _make_prefixsum_s(2**log_2_observations).numpy()
    solution = loops.compute_h_fixed_point_iteration(
        s_matrix=target_s, target_relative_duality_gap=1e-6)
    w, h = solution['W'], solution['H']
    direct_loss = loops.compute_loss_w_h(w, h)
    self.assertAllClose(solution['losses'][-1], direct_loss)
    # In seemingly a numerical accident, the 2x2 optimum loss
    # is the golden ratio squared.
    golden_ratio = (1 + np.sqrt(5)) / 2
    self.assertAllClose(solution['losses'][-1], golden_ratio**2)

    # Make sure we actually have a factorization of target_s
    self.assertAllClose(w @ h, target_s)

    # Assert that we took the positive factorization and are lower triangular
    self.assertGreater(w[0, 0], 0.)
    self.assertGreater(w[1, 1], 0.)
    self.assertEqual(w[0, 1], 0.)

  def test_lowering_tolerance_increases_solution_quality(self):
    log_2_observations = 2
    target_s = _make_prefixsum_s(2**log_2_observations).numpy()
    high_tol_solution = loops.compute_h_fixed_point_iteration(
        s_matrix=target_s, target_relative_duality_gap=0.1, iters_per_eval=1)
    low_tol_solution = loops.compute_h_fixed_point_iteration(
        s_matrix=target_s, target_relative_duality_gap=0.001, iters_per_eval=1)
    self.assertLess(low_tol_solution['losses'][-1],
                    high_tol_solution['losses'][-1])

  def test_losses_non_increasing_4x4(self):
    target_s = np.tri(4)
    soln = loops.compute_h_fixed_point_iteration(
        s_matrix=target_s,
        target_relative_duality_gap=0.0,
        max_fixed_point_iterations=10,
        iters_per_eval=1)
    losses = soln['losses']
    self.assertLen(losses, 10)
    dual_obj_vals = soln['dual_obj_vals']
    self.assertLen(dual_obj_vals, 10)
    dual_obj = dual_obj_vals[0]
    loss = losses[0]
    for i in range(1, 10):
      # Dual obj lower bounds primal obj:
      self.assertLessEqual(dual_obj, loss)
      # Losses are decreasing:
      self.assertLess(losses[i], loss)
      # Dual obj is increasing:
      self.assertGreater(dual_obj_vals[i], dual_obj)
      loss = losses[i]
      dual_obj = dual_obj_vals[i]

  @parameterized.named_parameters(('n=2', 2), ('n=5', 5), ('n=8', 8))
  def test_lagrangian(self, n):
    s_matrix = jnp.tri(n)
    target = s_matrix.T @ s_matrix
    lagrange_multiplier = jnp.linspace(0.1, 1.0, num=n)
    x_matrix = loops.x_matrix_from_dual(lagrange_multiplier, target=target)
    lb1 = loops.lagrangian_fn(x_matrix, lagrange_multiplier, target=target)
    # From the Eq. in the paper
    diag_v = jnp.diag(lagrange_multiplier)
    lb2 = jnp.trace(diag_v @ (2 * x_matrix - jnp.eye(n)))
    upper_bound = loops.compute_loss_for_x(x_matrix, target)
    self.assertAllClose(lb1, lb2)
    self.assertLessEqual(lb1, upper_bound)


if __name__ == '__main__':
  tf.random.set_seed(2)
  np.random.seed(2)
  tf.test.main()
