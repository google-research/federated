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
"""Loops for learning factorizations for prefix-sum problem."""

import asyncio
import collections
from collections.abc import Mapping, Sequence
import functools
import time
from typing import Any, Optional

from absl import logging
from jax import numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from multi_epoch_dp_matrix_factorization import fixed_point_library
from multi_epoch_dp_matrix_factorization import matrix_constructors
from multi_epoch_dp_matrix_factorization import solvers


@tf.function
def _compute_h_loss(h_matrix) -> tf.Tensor:
  column_norms = tf.linalg.norm(h_matrix, axis=0)
  return tf.reduce_max(column_norms)


def _compute_loss_squared_grad_row_by_row(
    h_var: tf.Variable, s_matrix: tf.Tensor, constraint_matrix: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  r"""Computes the gradient of the squared loss wrt h in a row-by-row manner.

  The loss we are interested in here is the Frobenius norm of W multiplied by
  the max-l^2-column norm of H. The square of this loss can be computed as the
  sum of (squared) Frobenius norms of the rows of W, multiplied by the squared
  max column norm of H. Therefore the gradient of this squared loss is identical
  to the sum of the gradients of the gradients of these terms. That is:

  l ** 2 = \sum_{k=0}^n l_k ** 2  =>
  d(l ** 2) / dH = \sum_{k=0}^n d(l_k ** 2) /dH

  This function leverages this row-by-row decomposition to compute the gradients
  of the squared loss in a memory efficient manner, allowing constrained
  optimization to scale on GPUs to O(thousand)-dimensional vector spaces.

  Args:
    h_var: Rank-2 TF Variable representing the 'current' H term in our
      optimization procedure; the variable with respect to which we
      differentiate.
    s_matrix: Rank-2 tensor, the matrix to be factorized.
    constraint_matrix: Rank-2 boolean tensor representing (axis-aligned)
      constraints on the solutions to the factorization problem S = WH. An entry
      in the solution matrix W may be nonzero precisely when the corresponding
      entry in `constraint_matrix` is `True`.

  Returns:
    A three-tuple of tensors. The first represents the gradient described above;
    the second represents the (unsquared) value of the loss in our
    factorization problem; the third represents the optimal solution `W` to our
    factorization problem for a fixed value of `h_var`.
  """
  # TODO(b/208139462): Add some coverage for this function, promote to public.
  grad = tf.zeros(shape=h_var.shape, dtype=h_var.dtype)
  loss_squared = tf.constant(0.0, dtype=h_var.dtype)
  w_rows = []
  for row_idx in range(s_matrix.shape[0]):
    with tf.GradientTape() as tape:
      w_row = solvers.solve_for_constrained_w_row_with_pseudoinv(
          h_var,
          tf.constant(s_matrix)[row_idx, :],
          tf.constant(constraint_matrix)[row_idx, :],
      )
      w_loss_square = tf.norm(w_row) ** 2
      h_loss_square = _compute_h_loss(h_var) ** 2
      loss = h_loss_square * w_loss_square
      loss_for_grad = w_loss_square
      loss_squared = tf.cast(loss, loss_squared.dtype) + loss_squared
    grad = tf.cast(tape.gradient(loss_for_grad, h_var), grad.dtype) + grad
    w_rows.append(w_row)
  return grad, loss_squared**0.5, tf.stack(tf.squeeze(w_rows))


def _compute_grad(h_var, s_matrix) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Optimized grad-computing function for full-batch factorization problem."""

  with tf.GradientTape() as tape:
    w_matrix = solvers.solve_directly_for_optimal_full_batch_w(
        h_var, tf.constant(s_matrix)
    )
    w_loss = tf.norm(w_matrix)
    h_loss = _compute_h_loss(h_var)
    # We compute the squared-loss here to keep consistent with the row-by-row
    # decomposition above.
    loss = (h_loss * w_loss) ** 2
  grad = tape.gradient(loss, h_var)
  return grad, loss**0.5, w_matrix


def learn_h_sgd(
    *,
    initial_h: tf.Tensor,
    s_matrix: tf.Tensor,
    streaming: bool = True,
    n_iters: int = 10,
    optimizer: tf.keras.optimizers.Optimizer,
    program_state_manager: Optional[tff.program.ProgramStateManager] = None,
    rounds_per_saving_program_state: int = 1,
    metrics_managers: Sequence[tff.program.ReleaseManager] = (),
) -> Mapping[str, Any]:
  """Performs gradient descent on prefix-sum reconstruction loss over H.

  That is, this function: takes some initial value for H, and computes the W
  of minimal Frobenius norm which satisfies WH = S; computes loss from this
  (H, W)-pair and perform gradient
  descent with respect to this loss over the parameters of H.

  Args:
    initial_h: tf.Tensor of rank 2 (IE, matrix) representing the value of H with
      which to initialize the factorization learning procedure.
    s_matrix: The matrix to be factorized.
    streaming: Boolean indicating whether or not to constrain H, W matrices to a
      streaming structure. Here we define 'streaming structure' for H to be
      essentially similar lower-triangular structure to the binary-tree matrix
      (IE, learnable variables are below the highest nonzero entries in this
      matrix), and streaming constraints for W inherited from this choice.
      Notice that this is not necessarily optimal, simply one natural choice.
    n_iters: Number of steps for which to train this SGD procedure.
    optimizer: Instance of `tf.keras.optimizers.Optimizer` to use for SGD
      training procedure.
    program_state_manager: Optional instance of
      `tff.program.ProgramStateManager` to manage saving program state for
      long-running computations.
    rounds_per_saving_program_state: The number of training rounds to run
      between saving program state.
    metrics_managers: Sequence of `tff.program.ReleaseManager` to manage metric
      (in particular, loss) writing for large experiment grids.

  Returns:
    A dict of 'H', 'W' matrices, and a sequence of losses observed
    during the training procedure.
  """
  loop = asyncio.get_event_loop()

  if streaming:
    # We keep a mask to project the gradient. It is not clear that this is
    # optimal.
    grad_mask, w_mask = matrix_constructors.compute_masks_for_streaming(
        initial_h
    )
    # The remainder of the library is programmed assuming boolean constraint
    # matrix.
    constraint_matrix = tf.cast(w_mask, tf.bool)
  else:
    grad_mask = np.ones(initial_h.shape)
    # We use an optimized implementation for this case, bypassing the need to
    # construct a constraint matrix.

  if program_state_manager is None:
    iteration = -1
  else:
    restored_h, iteration = loop.run_until_complete(
        program_state_manager.load_latest(initial_h)
    )
    if iteration is None:
      logging.info('Starting training from scratch.')
      iteration = -1
    else:
      # We found a checkpoint. Restore from it.
      logging.info(
          'Restoring from checkpoint saved at iteration: %s', iteration
      )
      initial_h = restored_h

  # For square H, we can use the optimized path, since there is an unambiguous
  # inverse.
  h_shape = tf.shape(initial_h)
  h_is_square = h_shape[0] == h_shape[1]

  h_var = tf.Variable(initial_value=initial_h)
  loss_sequence = []

  for j in range(iteration + 1, n_iters, 1):
    if streaming and not h_is_square:
      grad, loss, w_matrix = _compute_loss_squared_grad_row_by_row(
          h_var, s_matrix, constraint_matrix
      )
    else:
      # TODO(b/229428010): In the square case, we can actually implement this
      # faster, since we can use real inverse or solver for an appropriate
      # linear system. A clear place to optimize this code if necessary.
      grad, loss, w_matrix = _compute_grad(h_var, s_matrix)

    loss_sequence.append(loss.numpy())
    logging.info('Loss at iteration %s: %s', j, loss_sequence[-1])
    metrics = {'loss': loss_sequence[-1]}
    metrics_type = tff.StructType(
        [
            ('loss', tff.TensorType(loss.dtype, loss.shape)),
        ]
    )
    loop.run_until_complete(
        asyncio.gather(
            *[m.release(metrics, metrics_type, j) for m in metrics_managers]
        )
    )

    if j < n_iters - 1:
      # Skipping the last gradient step ensures we finish in a consistent state.
      h_grad = grad_mask * grad
      optimizer.apply_gradients(zip([h_grad], [h_var]))

    if program_state_manager is not None:
      if j % rounds_per_saving_program_state == 0:
        loop.run_until_complete(
            program_state_manager.save(h_var.read_value(), j)
        )

  return collections.OrderedDict(
      H=h_var.read_value().numpy(),
      W=w_matrix.numpy(),
      loss_sequence=loss_sequence,
  )


@functools.lru_cache(maxsize=20)
def _make_permutation_matrix(n: int) -> np.ndarray:
  """Constructs a matrix of all-ones on antidiagonal, all zeros elsewhere."""
  arr = []
  for row_idx in range(n):
    row_elements = []
    for col_idx in range(n):
      if col_idx == n - row_idx - 1:
        row_elements.append(1.0)
      else:
        row_elements.append(0)
    arr.append(row_elements)
  return jnp.array(arr)


def _permute_lower_triangle(h_lower):
  """Computes PXP^T for P permutation matrix above."""
  perm = _make_permutation_matrix(h_lower.shape[0])
  return perm @ h_lower @ perm.T


def x_matrix_from_dual(
    lagrange_multiplier: jnp.ndarray, target: jnp.ndarray
) -> jnp.ndarray:
  """Computes the matrix X that optimizes the Lagrangian."""
  inv_diag_sqrt = jnp.diag(lagrange_multiplier ** -(0.5))
  diag_sqrt = jnp.diag(lagrange_multiplier**0.5)

  # We need to ensure the smallest eigenvalue of `inner_term` below
  # is bounded away from 0 before we compute the Cholesky decomposition.
  # Following Thm. 7 of Merikoski and Kumar 2004
  # (https://www.emis.de/journals/AMEN/2004/040205-3.pdf),
  # noting lambda_n is the smallest eigenvalue in their notation, we have:
  min_eigenvalue = jnp.linalg.eigvalsh(target)[0] * jnp.min(lagrange_multiplier)

  inner_term = diag_sqrt @ target.astype(diag_sqrt.dtype) @ diag_sqrt
  sqrt_inner_term = fixed_point_library.diagonalize_and_take_jax_matrix_sqrt(
      inner_term, min_eigenvalue
  )
  # Note: We might be able to speed things up a little bit by computing
  # the inverse of X here at the same time.
  return inv_diag_sqrt @ sqrt_inner_term @ inv_diag_sqrt


def w_and_h_from_x_and_s(
    x_matrix: jnp.ndarray, s_matrix: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
  h_lower = jnp.linalg.cholesky(_permute_lower_triangle(x_matrix))
  h_matrix = _permute_lower_triangle(h_lower.T).astype(s_matrix.dtype)
  w_matrix = s_matrix @ jnp.linalg.inv(h_matrix)
  return w_matrix, h_matrix


def lagrangian_fn(
    x_matrix: jnp.ndarray,
    lagrange_multiplier: jnp.ndarray,
    target: jnp.ndarray,
    x_inv: Optional[jnp.ndarray] = None,
) -> float:
  """Evaluates the Lagrangian function."""
  x_inv = jnp.linalg.inv(x_matrix) if x_inv is None else x_inv
  return jnp.trace(target @ x_inv) + lagrange_multiplier @ (
      jnp.diagonal(x_matrix) - 1
  )


def compute_loss_for_x(
    x_matrix: jnp.ndarray,
    target: jnp.ndarray,
    x_inv: Optional[jnp.ndarray] = None,
) -> float:
  """Computes the primal objective for x."""
  x_inv = jnp.linalg.inv(x_matrix) if x_inv is None else x_inv
  # Primal objective
  m = target @ x_inv
  # Apply global scaling up of objective if constraints are violated.
  max_diag = jnp.maximum(jnp.max(jnp.diag(x_matrix)), 1.0)
  return jnp.trace(m) * max_diag


def compute_loss_w_h(w: jnp.ndarray, h: jnp.ndarray) -> float:
  """Computes the loss for a (W, H) factorization."""
  sensitivity = jnp.max(jnp.linalg.norm(h, axis=0))
  w_norm = jnp.linalg.norm(w)
  return sensitivity**2 * w_norm**2


def compute_h_fixed_point_iteration(
    *,
    s_matrix: np.ndarray,
    target_relative_duality_gap: float = 0.001,
    max_fixed_point_iterations: int = 1000000,
    iters_per_eval: int = 10,
) -> Mapping[str, Any]:
  """Computes square H, optimal for factorizing S = WH, by fixed-point iteration.

  The fixed-point characterization of optimal H can be found in
  Theorem 3.1 of https://arxiv.org/abs/2202.08312; this function computes the
  appropriate fixed
  point and maps back to square lower-triangular H, assuming S is
  lower-triangular.

  Args:
    s_matrix: Matrix to factorize.
    target_relative_duality_gap: Continue fixed point iterations until
      `max_fixed_point_iterations` is reached, or `suboptimality /
      optimal_loss_lower_bound <= target_relative_duality_gap.`
    max_fixed_point_iterations: The maximum number of fixed-point iterations
      used to compute the approximate fixed point of phi.
    iters_per_eval: At this interval, evaluate the primal and dual objectives to
      assess the stopping criteria.

  Returns:
    An ordered dict containing keys:
      `H`, `W`: The factorization of s_matrix
      `losses`: Sequence of losses at each eval, losses[-1] is the loss
         of the final factorization.
      'dual_obj_vals`: Sequence of dual objective values at each eval.
      'n_iters': Total number of fixed-point iterations.
  """
  start_time = time.time()
  target = s_matrix.T @ s_matrix
  num_iters = 0

  lagrange_multiplier = fixed_point_library.init_fixed_point(target)
  losses = []
  dual_obj_vals = []

  while num_iters < max_fixed_point_iterations:
    lagrange_multiplier = fixed_point_library.compute_phi_fixed_point(
        target, lagrange_multiplier, iterations=iters_per_eval
    )
    num_iters += iters_per_eval
    x_matrix = x_matrix_from_dual(lagrange_multiplier, target)  # pytype: disable=wrong-arg-types  # jnp-array
    x_inv = jnp.linalg.inv(x_matrix)
    loss = compute_loss_for_x(x_matrix, target, x_inv=x_inv)
    # Evaluating dual at the optimal X for the given lagrange_multiplier
    # gives a lower bound on the primal objevtive (loss).
    dual_obj = lagrangian_fn(  # pytype: disable=wrong-arg-types  # jnp-array
        x_matrix, lagrange_multiplier, target=target, x_inv=x_inv
    )
    assert dual_obj > 0
    duality_gap = loss - dual_obj
    relative_gap = duality_gap / dual_obj
    losses.append(loss)
    dual_obj_vals.append(dual_obj)

    total_time = time.time() - start_time
    time_per_iter = total_time / num_iters
    log_str = (
        f'{num_iters:5d}  loss {loss:6.2f} dual obj {dual_obj:6.2f} '
        f'gap {duality_gap:6.2g} relative {relative_gap:6.2g}, '
        f'total {total_time:.0f} seconds, or {time_per_iter:.2f} per iter'
    )
    logging.info(log_str)
    assert duality_gap >= -1e-10
    if relative_gap <= target_relative_duality_gap:
      logging.info('Reached target duality gap.')
      break

  w, h = w_and_h_from_x_and_s(x_matrix, s_matrix)  # pytype: disable=wrong-arg-types  # jnp-array

  assert jnp.all(jnp.isfinite(w))
  assert jnp.all(jnp.isfinite(h))
  return collections.OrderedDict(
      H=h, W=w, losses=losses, dual_obj_vals=dual_obj_vals, n_iters=num_iters
  )
