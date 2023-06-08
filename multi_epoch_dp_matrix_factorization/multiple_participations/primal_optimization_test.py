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
"""Tests for primal_optimization."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from multi_epoch_dp_matrix_factorization.multiple_participations import contrib_matrix_builders
from multi_epoch_dp_matrix_factorization.multiple_participations import lagrange_terms
from multi_epoch_dp_matrix_factorization.multiple_participations import optimization
from multi_epoch_dp_matrix_factorization.multiple_participations import primal_optimization


# Disabling pylint so we can use single-letter capitals to denote matrices
# as per notation table in README.md
# pylint:disable=invalid-name
def check_symmetric(X):
  return np.allclose(X, X.T)


def check_psd(X):
  """Checks if a given matrix is PSD."""
  return np.all(np.linalg.eigvals(X) > 0)


def check_banded(X, bands):
  """Checks if a given matrix is banded."""
  n = X.shape[0]
  for i in range(n):
    for j in range(n):
      if abs(i - j) > bands and X[i, j] != 0:
        return False
  return True


def check_toeplitz(X):
  """Checks if a given matrix is Toeplitz."""
  n = X.shape[0]
  for i in range(n):
    for k in range(n - i):
      if X[i + k, k] != X[i, 0]:
        return False
      if X[k, i + k] != X[0, i]:
        return False
  return True


def check_equal_norm(h_matrix):
  """Checks that column norms of matrix are equal."""
  norms = np.linalg.norm(h_matrix, axis=0)
  return max(norms) - min(norms) <= 1e-8


def check_sensitivity(X, epochs):
  n = X.shape[0]
  contrib_matrix = contrib_matrix_builders.epoch_participation_matrix(n, epochs)
  lt = lagrange_terms.init_lagrange_terms(contrib_matrix)
  mx, mn = optimization.max_min_sensitivity_squared_for_x(X, lt)
  return abs(mx - 1) <= 1e-8 and abs(mn - 1) <= 1e-8


def check_improved(X, opt):
  init = np.eye(opt._n) / opt._k
  loss1 = opt.loss_and_gradient(init)[0]
  loss2 = opt.loss_and_gradient(X)[0]
  return loss2 <= loss1


def check_optimal(X, opt):
  _, projected_grad = opt.loss_and_gradient(X)
  return np.abs(projected_grad).max()


def is_mask(mask):
  return np.all((mask == 0) | (mask == 1))


class PrimalOptimizationTest(parameterized.TestCase):

  # pylint:disable=bad-whitespace
  # pyformat:disable
  @parameterized.named_parameters(
      ('single_epoch', 1,  None, False, False),
      ('full_batch',   16, None, False, False),
      ('multi_epoch',  4,  None, False, False),
      ('banded',       4,  3,    False, False),
      ('toeplitz',     4,  None, True,  False),
      ('equal_norm',   4,  None, False, True))
  # pyformat:enable
  # pylint:enable=bad-whitespace
  def test_optimization_worked(self, epochs, bands, toeplitz, equal_norm):
    A = np.tri(16)
    opt = primal_optimization.MatrixFactorizer(
        A, epochs=epochs, bands=bands, toeplitz=toeplitz, equal_norm=equal_norm
    )
    X = opt.optimize(
        256,
        termination_fn=primal_optimization.build_pg_tol_termination_fn(1e-8),
    )
    self.assertTrue(check_symmetric(X))
    self.assertTrue(check_psd(X))
    self.assertTrue(check_sensitivity(X, epochs))
    self.assertLess(check_optimal(X, opt), 1e-3)
    self.assertTrue(check_improved(X, opt))

  def test_identity(self):
    n = 16
    A = np.eye(n)
    opt = primal_optimization.MatrixFactorizer(A, epochs=1)
    X = opt.optimize(100)
    self.assertTrue(np.allclose(X, A))

  def test_termination_fn_called_every_step(self):
    num_times_called = 0

    def no_termination(X, dX, loss) -> bool:
      del X, dX, loss  # Unused, always return false.
      nonlocal num_times_called
      num_times_called += 1
      return False

    n = 16
    A = np.eye(n)
    opt = primal_optimization.MatrixFactorizer(A, epochs=1)
    opt.optimize(100, termination_fn=no_termination)
    self.assertEqual(num_times_called, 100)

  def test_initial_X_checked_for_entries_zero(self):
    n = 16
    A = np.eye(n)
    # 16 epochs means that any off-diagonal nonzero entry in X is undesirable.
    opt = primal_optimization.MatrixFactorizer(A, epochs=16)
    with self.assertRaisesRegex(ValueError, 'nonzero in indices i, j'):
      opt.optimize(100, initial_X=np.ones(shape=(16, 16)))

  def test_single_band(self):
    n = 16
    A = np.tri(n)
    opt = primal_optimization.MatrixFactorizer(A, epochs=1, bands=0)
    X = opt.optimize(100)
    self.assertTrue(np.allclose(X, np.eye(n)))

    opt = primal_optimization.MatrixFactorizer(A, epochs=n, bands=0)
    X = opt.optimize(100)
    self.assertAlmostEqual(np.diag(X).sum(), 1.0)
    self.assertTrue(check_banded(X, 1))

  def test_toeplitz(self):
    n = 16
    A = np.tri(n)
    opt = primal_optimization.MatrixFactorizer(A, epochs=4, toeplitz=True)
    X = opt.optimize(100)
    self.assertTrue(check_toeplitz(X))

  def test_colnorm(self):
    n = 16
    A = np.tri(n)
    opt = primal_optimization.MatrixFactorizer(A, epochs=4, equal_norm=True)
    X = opt.optimize(100)
    C = np.linalg.cholesky(X).T
    self.assertTrue(check_equal_norm(C))

  def test_orth_masks(self):
    n = 16
    orth_mask = primal_optimization.get_orthogonal_mask(n, 4)
    self.assertTrue(is_mask(orth_mask))


# pylint:enable=invalid-name
if __name__ == '__main__':
  tf.test.main()
