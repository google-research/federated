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
"""Tests for optimization."""
from jax import numpy as jnp
import numpy as np
import optax
import tensorflow as tf
import tensorflow_federated as tff

from multi_epoch_dp_matrix_factorization.multiple_participations import contrib_matrix_builders
from multi_epoch_dp_matrix_factorization.multiple_participations import lagrange_terms
from multi_epoch_dp_matrix_factorization.multiple_participations import optimization


class OptimizationTest(tf.test.TestCase):

  def test_sqrt_and_sqrt_inverse(self):
    n = 10
    a = np.linspace(0.1, 0.3, num=n * n).reshape((n, n)) + np.eye(n)
    a = a.T @ a  # PSD
    s, s_inv = optimization.sqrt_and_sqrt_inv(a)
    np.testing.assert_allclose(s @ s_inv, jnp.eye(10), atol=1e-7)
    np.testing.assert_allclose(s @ s, a, atol=1e-7)

    s, s_inv = optimization.sqrt_and_sqrt_inv(a, compute_inverse=False)
    self.assertIsNone(s_inv)
    np.testing.assert_allclose(s @ s, a, atol=1e-7)

  def test_max_min_sensitivity_squared_for_x(self):
    n = 3
    # x_matrix should be PD
    x_matrix = jnp.diag(jnp.array([5.0, 0.1, 2.0]))
    lt = lagrange_terms.init_lagrange_terms(contrib_matrix=jnp.eye(n))
    max_sens, min_sens = optimization.max_min_sensitivity_squared_for_x(
        x_matrix, lt
    )
    self.assertAlmostEqual(max_sens, 5.0)
    self.assertEqual(min_sens, 0.1)

  def test_x_and_x_inv_from_dual(self):
    n = 8
    contrib_matrix = jnp.eye(n)
    lagrange_multiplier = jnp.arange(1, n + 1)
    s_matrix = np.tri(n)
    target = s_matrix.T @ s_matrix
    lt = lagrange_terms.LagrangeTerms(
        lagrange_multiplier=lagrange_multiplier, contrib_matrix=contrib_matrix
    )
    x, x_inv = optimization.x_and_x_inv_from_dual(
        lt, target=target, compute_inv=True
    )
    np.testing.assert_allclose(np.eye(n), x @ x_inv, atol=1e-10)

  def test_optax_update(self):
    n = 10
    s_matrix = np.tri(n)
    target = s_matrix.T @ s_matrix
    contrib_matrix = contrib_matrix_builders.epoch_participation_matrix(
        n=n, num_epochs=2
    )
    lt = lagrange_terms.LagrangeTerms(
        lagrange_multiplier=jnp.ones(contrib_matrix.shape[1]),
        contrib_matrix=contrib_matrix,
        nonneg_multiplier=2 * jnp.ones(shape=(n, n)),
    )

    update_fn = optimization.OptaxUpdate(nonneg_optimizer=optax.sgd(0.1), lt=lt)
    lt = update_fn(lt, target)
    lt.assert_valid()

  def test_solves_simple_problem(self):
    n = 20
    s_matrix = jnp.tri(n)
    contrib_matrix = jnp.eye(n)
    lt = lagrange_terms.init_lagrange_terms(contrib_matrix)
    r = optimization.solve_lagrange_dual_problem(
        s_matrix=s_matrix,
        lt=lt,
        iters_per_eval=4,
        update_langrange_terms_fn=optimization.OptaxUpdate(
            nonneg_optimizer=optax.sgd(0.1), lt=lt
        ),
        target_relative_duality_gap=0.001,
    )
    self.assertLessEqual(r['relative_duality_gap'], 0.001)

  def test_with_program_state_manager(self):
    state_manager = tff.program.FileProgramStateManager(
        root_dir=self.create_tempdir('program_state')
    )
    n = 3
    s_matrix = jnp.tri(n)
    contrib_matrix = jnp.eye(n)
    lt = lagrange_terms.init_lagrange_terms(contrib_matrix)
    r = optimization.solve_lagrange_dual_problem(
        s_matrix=s_matrix,
        lt=lt,
        iters_per_eval=2,
        max_iterations=2,
        program_state_manager=state_manager,
    )
    self.assertEqual(r['n_iters'], 2)
    # max_iterations isn't part of the state,
    # so we can resume and run more iterations:
    with self.assertLogs(level='INFO') as logs:
      r = optimization.solve_lagrange_dual_problem(
          s_matrix=s_matrix,
          lt=lt,
          iters_per_eval=2,
          max_iterations=4,
          program_state_manager=state_manager,
      )
      self.assertEqual(r['n_iters'], 4)
      self.assertIn(
          'Restored at iteration 2 from state checkpoint', str(logs.output)
      )


if __name__ == '__main__':
  tf.test.main()
