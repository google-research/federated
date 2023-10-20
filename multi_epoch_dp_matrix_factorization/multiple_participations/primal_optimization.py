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
"""Primal optimization algorithms for multi-epoch matrix factorization."""

from collections.abc import Callable
import functools
from typing import Optional

from jax import config
from jax import jit
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import scipy as sp
from typing_extensions import Protocol

config.update('jax_enable_x64', True)

# We use math-style names here to conform better to the paper, though the Python
# linter does not like it.
# See README.md for a notation table.
# pylint:disable=invalid-name


def get_toeplitz_idx(n: int) -> jnp.ndarray:
  """Computes a Toeplitz matrix where entries are integer indices [0, ..., n-1].

  In particular, T_{ij} = | i - j |.  For example, get_toeplitz_idx(4) returns:
  ```
  [0 1 2 3]
  [1 0 1 2]
  [2 1 0 1]
  [3 2 1 0]
  ```

  Args:
    n: the size of the Toeplitz matrix

  Returns:
    A Toeplitz matrix with integer indices as entries.
  """
  return jnp.array(sp.linalg.toeplitz(np.arange(n)))


def get_orthogonal_mask(n: int, epochs: int = 1) -> jnp.ndarray:
  """Computes a mask that imposes orthognality constraints on the optimization.

  This is specific to the fixed-epoch-order (k, b)-participation schema of
  https://arxiv.org/pdf/2211.06530.pdf, where participations are separated by
  exactly b-1 steps, and b = n / epochs.

  This mask sets entry M_{ij} = 0 if i == j (mod n/epochs) and M_{ij} = 1
  otherwise.  Sensitivity for any matrix with 0s in these entries is easy to
  calculate as only a function of the diagonal.  Moreover, the sensitivity is
  equal for all possible {-1,1} participation vectors.

  Args:
    n: the size of the mask
    epochs: The number of epochs

  Returns:
    A 0/1 mask
  """
  mask = np.ones((n, n))
  for i in range(n // epochs):
    mask[i :: n // epochs, i :: n // epochs] = np.eye(epochs)
  return jnp.array(mask)


def _initialize_X_to_normalized_identity(
    *, n_steps: int, epochs: int
) -> jnp.ndarray:
  """Initializes X matrix to be optimized to a normalized identity matrix.

  The identity constructed here is normalized to have sensitivity (squared)
  exactly 1 under the (k, b) training scheme specified by n_steps and epochs.
  Also trivially sets all indices [i, j] to 0 in cases where a user can
  participate in iteration i and j.

  Args:
    n_steps: Integer specifying the number of steps for which we wish to train
      the downstream ML model.
    epochs: Integer specifying the number of epochs which will be used in
      training for n_steps.

  Returns:
    A scaled identity matrix, with sensitivity 1 under the (k, b) participation
    pattern specified by the parameters.
  """
  return jnp.eye(n_steps, dtype=jnp.float64) / epochs


class TerminationFn(Protocol):
  """Boolean-returning function used for early stopping in optimization."""

  def __call__(self, *, X: jnp.ndarray, dX: jnp.ndarray, loss: float) -> bool:
    """Indicates whether termination is desired. Signature may evolve."""


def build_pg_tol_termination_fn(pg_tol: float) -> TerminationFn:
  def _terminate(X, dX, loss) -> bool:
    del X, loss  # Unused
    return jnp.abs(dX).max() <= pg_tol

  return _terminate


class MatrixFactorizer:
  """Class for factorizing matrices."""

  def __init__(
      self,
      workload_matrix: jnp.ndarray,
      epochs: int = 1,
      equal_norm: bool = False,
  ):
    """Initializes this MatrixFactorizer object.

    Note: Although this implementation of MatrixFactorizer supports optimization
      of structured matrices, it does nothing to exploit their structure to
      speed up optimization.

    Note: Currently, this class only supports the canonical (k,b) participation
      pattern where a user contributes k times total, every b iterations.
      Currently, this class specifically supports the fixed-epoch-order
      (k, b)-participation schema of https://arxiv.org/pdf/2211.06530.pdf,
      where participations are separated by exactly b-1 steps, and
      b = num_iterations / num_epochs.

    Note: In this class we are always imposing certain orthogonality constraints
    to make the optimization problem easier to solve and tractable for large
    numbers of epochs.  Specifically, we require that X_{ij} = 0 if i != j and
    a user can appear in both iteration i and iteration j.  This ensures that
    columns i and j of C are orthogonal, and that their L2^2 sensitivites simply
    add up. We have found empirically that this structure is optimal for the
    Prefix workload and we conjecture it may be optimal more generally.

    Args:
      workload_matrix: The input workload, a n x n lower triangular matrix.
      epochs: A positive integer in the range [1,n] that evenly divides n. Is
        used to determine the participation pattern and define the sensitivity
        of a given matrix.
      equal_norm: [Optional] A flag to indicate if columns of C should have
        equal norm (i.e., X_ii = 1/epochs).
    """
    self._A = jnp.array(workload_matrix)
    self._n = self._A.shape[1]
    self._k = epochs
    self._equal_norm = equal_norm
    # These masks determine which entries of X are allowed to be non-zero.
    orth_mask = get_orthogonal_mask(self._n, epochs)
    self._mask = orth_mask

  @functools.partial(jit, static_argnums=(0,))
  def project_update(self, dX: jnp.ndarray) -> jnp.ndarray:
    r"""Project dX so that X + alpha*dX satisfies constraints for any alpha.

    Note: this function assumes that X already satisfies the constraints.

    This function does multiple things:
      1. To ensure the sensitivity of the resulting mechanism remains 1:
        a. It sets $dX[i,j] = 0$ if $i \neq j$ and a user can appear in both
          round $i$ and $j$.
        b. It normalizs $sum_i dX[i,i] = 0$, where sum is taken over rounds a
          single user can participate in.  This ensures that sum_i X[i,i]
          remains equal to 1.
      2. If banded constraints are given, sets dX[i,j] = 0 if |i-j| > # bands.
      3. If Toeplitz structure is needed, ensures dX is a toeplitz matrix.  This
        also ensures X is Toeplitz because Toeplitz + Toeplitz = Toeplitz.

    Args:
      dX: an n x n matrix, representing the gradient with respect to X.

    Returns:
      an n x n matrix, representing the projected gradient.
    """
    if self._equal_norm:
      diag = 0
    else:  # Implement 1(b) from above:
      dsum = jnp.diag(dX).reshape(self._k, -1).sum(axis=0) / self._k
      diag = jnp.diag(dX) - jnp.kron(jnp.ones(self._k), dsum)
    dX = dX.at[jnp.diag_indices(self._n)].set(diag)
    # Implement 1(a) and 2 from above:
    dX = dX * self._mask
    if self._toeplitz:  # Implement 3 from above:
      # We sum the gradient along each (pair of) diagonals, which by chain rule
      # gives the derivate wrt the Toeplitz coefficients.  Toeplitz matrix with
      # these coefficients is constructed so the shape is compatible with X.
      locs = self._Tidx.flatten()
      weights = dX.flatten()
      dX = jnp.bincount(locs, weights, length=self._n)[self._Tidx]
    return dX

  @functools.partial(jit, static_argnums=(0,))
  def loss_and_gradient(self, X: jnp.ndarray) -> tuple[float, jnp.ndarray]:
    r"""Computes the matrix mechanism loss and gradient.

    This function computes $\tr[A^T A X^{-1}]$ and the associated gradient
    $dX = -X^{-1} A^T A X^{-1}$.  It assumes that $X$ is a symmetric positive
    definite matrix.  For efficiency, no error is thrown if this assumption is
    not satisfied, but the returned loss or gradient may contain NaN's if this
    is the case.

    Args:
      X: The current iterate, an n x n matrix

    Returns:
      loss: a real-valued number
      gradient: the gradient of the loss w.r.t. X, an n x n matrix
    """
    H = jsp.linalg.solve(X, self._A.T, assume_a='pos')
    return jnp.trace(H @ self._A), self.project_update(-H @ H.T)

  @functools.partial(jit, static_argnums=(0,))
  def _lbfgs_direction(
      self, X: jnp.ndarray, dX: jnp.ndarray, X1: jnp.ndarray, dX1: jnp.ndarray
  ) -> jnp.ndarray:
    """Computes the LBFGS search direction.

    Given the current/previous iterates (X and X1) and the current/previous
    gradients (dX and dX1), compute a search direction (Z) according to the
    LBFGS update rule.

    Args:
      X: The current iterate, an n x n matrix
      dX: The current gradient, an n x n matrix
      X1: The previous iterate, an n x n matrix
      dX1: The previous gradient, an n x n matrix

    Returns:
      The (negative) search direction, an n x n matrix
    """
    S = X - X1
    Y = dX - dX1
    rho = 1.0 / jnp.sum(Y * S)
    alpha = rho * jnp.sum(S * dX)
    gamma = jnp.sum(S * Y) / jnp.sum(Y**2)
    Z = gamma * (dX - rho * jnp.sum(S * dX) * Y)
    beta = rho * jnp.sum(Y * Z)
    Z = Z + S * (alpha - beta)
    return Z

  def optimize(
      self,
      iters: int = 1000,
      metric_callback: Optional[Callable[[int, dict[str, float]], None]] = None,
      initial_X: Optional[jnp.ndarray] = None,
      termination_fn: TerminationFn = build_pg_tol_termination_fn(pg_tol=1e-3),
  ) -> jnp.ndarray:
    """Optimize the strategy matrix with an iterative gradient-based method.

    Args:
      iters: The number of iterations to run the optimization.
      metric_callback: A function for logging that must consume a dictionary of
        metrics.
      initial_X: Matrix to use as the starting value for initialization. Assumed
        to have sensitivity 1 under the (k, b) participation pattern for which
        `self` is configured. Sensitivity will remain unchanged for this pattern
        during the course of optimization. In (k, b) participation, any entries
        in initial_X at index [i, j] with i != j will remain unchanged if a user
        can participate in both iteration i and j. Such entries must be set to
        0. If `None`, defaults to a normalized identity.
      termination_fn: Function which controls early termination. Must take X,
        dX, and loss as keyword arguments. Returning true indicates that X
        should be immediately returned from the optimization procedure.

    Returns:
      A matrix X that approximately minimizes the objective tr[A^T A X^{-1}] and
      satisfies the sensitivity and structural constraints.
    """
    if initial_X is None:
      X = _initialize_X_to_normalized_identity(n_steps=self._n, epochs=self._k)
    else:
      # It may be desirable to check / raise on sensitivity here. For now,
      # assume our callers performed this check.
      X = initial_X
    if not np.all((1 - self._mask) * X == 0):
      raise ValueError(
          'Initial X matrix is nonzero in indices i, j where '
          'i != j and some user can participate in rounds i and '
          'j. Such entries being zero is generally assumed by the '
          'optimization code here and downstream consumers in '
          'order to easily reason about sensitivity.'
      )
    loss, dX = self.loss_and_gradient(X)
    X1 = X  # X at previous iteration
    dX1 = dX  # dX at previous iteration
    loss1 = loss  # Loss at previous iteration
    Z = dX  # The negative search direction (different from dX in general)

    for step in range(iters):
      step_size = 1.0
      for _ in range(30):
        X = X1 - step_size * Z
        loss, dX = self.loss_and_gradient(X)
        if jnp.isnan(loss).any() or jnp.isnan(dX).any():
          step_size *= 0.25
        elif loss < loss1:
          loss1 = loss
          break
      if termination_fn(X=X, dX=dX, loss=loss):
        # Early-return triggered; return X immediately.
        return X
      if metric_callback is not None:
        metric_callback(step, {'loss': loss})

      Z = self._lbfgs_direction(X, dX, X1, dX1)
      X1 = X
      dX1 = dX
    return X


# pylint:enable=invalid-name
