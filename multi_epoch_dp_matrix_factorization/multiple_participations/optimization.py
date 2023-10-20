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
"""Library for the optimization functions inlined from Brendan's colab."""
import asyncio
import collections
from collections.abc import Mapping, Sequence
import dataclasses
import functools
import time
from typing import Any, Optional

from absl import flags
from absl import logging
import jax
from jax import config
from jax.experimental import host_callback
import jax.numpy as jnp
import optax
import tensorflow_federated as tff

from multi_epoch_dp_matrix_factorization import loops
from multi_epoch_dp_matrix_factorization.multiple_participations import lagrange_terms

# With large matrices, the extra precision afforded by performing all
# computations in float64 is critical.
config.update('jax_enable_x64', True)

_MIN_INNER_EIGENVALUE = flags.DEFINE_float(
    'min_inner_eigenvalue',
    1e-20,
    'Minimum eigenvalue in the inner term. Important for numerical stability.',
)


@functools.partial(jax.jit, static_argnums=(2,))
def sqrt_and_sqrt_inv(
    matrix: jnp.ndarray, min_eigenval: float = 0.0, compute_inverse: bool = True
) -> jnp.ndarray:
  """Matrix square root for positive-definite, Hermitian matrices."""

  def print_warning(v):
    v1, v2 = v
    # This is almost always a sign of numerical issues / problems.
    logging.info('WARNING: eigenval %s less than minimum %s', v1, v2)

  evals, evecs = jnp.linalg.eigh(matrix)
  min_eval = jnp.min(evals)
  # The callback allows us to compile this function
  jax.lax.cond(
      min_eval < min_eigenval,
      lambda: host_callback.call(print_warning, (min_eval, min_eigenval)),
      lambda: None)  # pyformat: disable
  eval_sqrt = jnp.maximum(evals, min_eigenval) ** 0.5
  sqrt = evecs @ jnp.diag(eval_sqrt) @ evecs.T
  sqrt_inv = None
  if compute_inverse:
    sqrt_inv = evecs @ jnp.diag(1 / eval_sqrt) @ evecs.T
  else:
    sqrt_inv = None
  return sqrt, sqrt_inv  # pytype: disable=bad-return-type  # jnp-array


def x_and_x_inv_from_dual(
    lt: lagrange_terms.LagrangeTerms,
    target: jnp.ndarray,
    compute_inv: bool = True,
) -> jnp.ndarray:
  """Computes the matrix X that optimizes the Lagrangian, and X^{-1}.

  Args:
    lt: The LagrangeTerms that summarize the constraints and current dual
      solution.
    target: The target matrix, A.T @ A.
    compute_inv: Whether or not to compute X^{-1}.

  Returns:
   A (likely infeasible) X matrix and X^{-1} if requested.
  """
  u_total = lt.u_total()  # Not easily jax-jittable.

  @functools.partial(jax.jit, static_argnums=(2,))
  def _impl(u_total, target, compute_inv):
    # With the nonnegativity constraints, we can end up with a u_total that
    # is not PD. It's not quite clear that projection onto the set of
    # PD matricies is "allowed", but it is probably better than nothing.
    # TODO(b/241453645): Validate or improve on this.
    diag_sqrt, inv_diag_sqrt = sqrt_and_sqrt_inv(u_total, min_eigenval=1e-10)

    # TODO(b/241453645): Compute a proper lower bound on the
    # smallest eigenvalue that works with U-matrix summaries and
    # non-negativity constraints.
    inner_term = diag_sqrt @ target @ diag_sqrt
    sqrt_inner_term, inv_sqrt_inner_term = sqrt_and_sqrt_inv(
        inner_term, _MIN_INNER_EIGENVALUE.value
    )
    x = inv_diag_sqrt @ sqrt_inner_term @ inv_diag_sqrt
    if compute_inv:
      x_inv = diag_sqrt @ inv_sqrt_inner_term @ diag_sqrt
    else:
      x_inv = None
    return (x, x_inv)

  return _impl(u_total, target, compute_inv)


class OptaxUpdate:
  """Implements optax optimizer interface for extended fixed-point updates."""

  def __init__(
      self,
      nonneg_optimizer=None,
      lt: Optional[lagrange_terms.LagrangeTerms] = None,
      multiplicative_update: bool = False,
  ):
    # N.B. A multiplicative_update for the
    # nonneg terms will *NOT UPDATE* if the lagrange multiplier is
    # initialized to zero, and so correct initialization is necessary.
    self.multiplicative_update = multiplicative_update
    self.nonneg_opt_state = None
    self.nonneg_optimizer = None
    if nonneg_optimizer and lt and lt.nonneg_multiplier is not None:
      self.nonneg_opt_state = nonneg_optimizer.init(lt.nonneg_multiplier)
      self.nonneg_optimizer = nonneg_optimizer

  def __call__(self, lt: lagrange_terms.LagrangeTerms, target: jnp.ndarray):
    x_matrix, _ = x_and_x_inv_from_dual(lt, target, compute_inv=False)
    lt_updates = {}

    if lt.lagrange_multiplier is not None:
      mult_update = jnp.diag(lt.contrib_matrix.T @ x_matrix @ lt.contrib_matrix)  # pytype: disable=attribute-error  # jnp-array
      assert jnp.all(
          mult_update >= 0
      ), f'x_matrix may not be PD, mult_update:\n {mult_update}'
      # Letting v be lt.lagrange_multiplier, we have
      #  new_v = mult_update * v
      #        = v + (mult_update - 1) * v
      #        = v + eta * (mult_update - 1) * v
      # when we choose a learning_rate eta = 1.
      # Note that (mult_update - 1) is in fact the gradient of Lagrange
      # dual function, so this is a multiplicative gradient update.
      # We could introduce the ability to adjust this learning rate,
      # or even a general optax optimizer.
      lt_updates['lagrange_multiplier'] = lt.lagrange_multiplier * mult_update
      assert jnp.all(lt_updates['lagrange_multiplier'] > 0)

    # Summary-U terms
    # u_multipliers are a mutable list, so we don't need to use lt_updates
    if lt.u_matrices is not None:
      assert len(lt.u_multipliers) == len(lt.u_matrices)
      for i, u_matrix in enumerate(lt.u_matrices):
        lt.u_multipliers[i] = lt.u_multipliers[i] * jnp.trace(  # pytype: disable=unsupported-operands  # jnp-array
            u_matrix @ x_matrix
        )

    # Non-negativity constraints
    # TODO(b/241453645): If we use too high a learning rate here, or equivalent,
    # we can easily produce a u_total() matrix that is not PD.
    # It would be nice to explicitly prevent this somehow, e.g. projection.
    if lt.nonneg_multiplier is not None:
      nonneg_multiplier = lt.nonneg_multiplier
      # Scaling x_matrix by the current nonneg_multiplier would
      # give the multiplicative update version.
      grads = x_matrix
      if self.multiplicative_update:
        grads *= nonneg_multiplier
      updates, new_nonneg_opt_state = self.nonneg_optimizer.update(  # pytype: disable=attribute-error  # jnp-array
          grads, self.nonneg_opt_state
      )
      nonneg_multiplier += updates
      self.nonneg_opt_state = new_nonneg_opt_state
      # Project to keep non-negative
      nonneg_multiplier = jnp.maximum(nonneg_multiplier, 0.0)
      lt_updates['nonneg_multiplier'] = nonneg_multiplier

    return dataclasses.replace(lt, **lt_updates)


def lagrangian_fn(
    x_matrix: jnp.ndarray, lt: lagrange_terms.LagrangeTerms
) -> float:
  """Evaluates the Lagrangian function, where x_matrix corresponds to lt."""
  return 2 * jnp.trace(lt.u_total() @ x_matrix) - lt.multiplier_sum()


def max_min_sensitivity_squared_for_x(
    x_matrix: jnp.ndarray, lt: lagrange_terms.LagrangeTerms
):
  """Returns sensitivity**2 for x_matrix based on constraints in `lt`.

  Args:
    x_matrix: The active x-matrix (may or may not e feasible).
    lt: LagrangeTerms from which to extract sensitivity constraints.

  Returns:
    A tuple (min_sensivity^2, max_sensitivity^2) where the min and max are
    over all the participation vectors `u` (columns of lt.contrib_matrix)
    as well as the `mixture` terms in u_matrices.
  """
  sens_list = []
  if lt.contrib_matrix is not None:
    h = lt.contrib_matrix
    s = jnp.diag(h.T @ x_matrix @ h)
    max_s, min_s = jnp.max(s), jnp.min(s)
    assert min_s >= -1e-10, min_s
    sens_list.extend([max_s, min_s])
  if lt.u_matrices is not None:
    sens_list.extend(
        [jnp.trace(u_matrix @ x_matrix) for u_matrix in lt.u_matrices]
    )
  assert sens_list
  return max(sens_list), min(sens_list)


def solve_lagrange_dual_problem(
    *,
    s_matrix: jnp.ndarray,
    lt: lagrange_terms.LagrangeTerms,
    target_relative_duality_gap: float = 0.001,
    update_langrange_terms_fn=None,
    max_iterations: int = 1000000,
    iters_per_eval: int = 1,
    program_state_manager: Optional[tff.program.ProgramStateManager] = None,
    metric_release_managers: Sequence[tff.program.ReleaseManager] = (),
) -> Mapping[str, Any]:
  """Solves the lagrange dual problem given by `lt` for the given `s_matrix`."""

  if update_langrange_terms_fn is None:
    logging.info('Using default OptaxUpdate.')
    update_langrange_terms_fn = OptaxUpdate(
        nonneg_optimizer=optax.sgd(0.01, momentum=0.95),
        lt=lt,
        multiplicative_update=False,
    )

  target = s_matrix.T @ s_matrix
  start_time = time.time()
  losses = []
  dual_obj_vals = []
  num_iters = 0

  def get_state_structure():
    return (start_time, dataclasses.asdict(lt))

  if program_state_manager is not None:
    # Possibly restore state from program_state_manager
    state_structure, state_num_iters = asyncio.run(
        program_state_manager.load_latest(get_state_structure())
    )
    if state_structure is not None:
      assert state_num_iters > 0
      num_iters = state_num_iters
      # We assume this is the way it was packaged, which we will force below.
      start_time, lt_dict = state_structure
      lt = lagrange_terms.LagrangeTerms(**lt_dict)
      logging.info('Restored at iteration %s from state checkpoint', num_iters)

  while True:
    for _ in range(iters_per_eval):
      lt = update_langrange_terms_fn(lt, target)
      lt.assert_valid()
      num_iters += 1

    x_matrix, _ = x_and_x_inv_from_dual(lt, target, compute_inv=False)

    # To compute the primal_obj, we want a feasible x_matrix,
    # which requires addressing sensitivity and non-negativity constraints:
    feasible_x_matrix = x_matrix.copy()

    if lt.nonneg_multiplier is not None:
      # Non-negativity constraints, project:
      feasible_x_matrix = jnp.maximum(feasible_x_matrix, 0.0)

    # Activity sensitivity based on the constraints tracked in the current
    # LagrangeTerms:
    active_sens_squared, min_sens_squared = max_min_sensitivity_squared_for_x(
        feasible_x_matrix, lt
    )
    # Scale the matrix so constraints are satisfied.
    feasible_x_matrix /= jnp.maximum(active_sens_squared, 1.0)
    feasible_x_inv = jnp.linalg.inv(feasible_x_matrix)

    # The Frobenius norm^2 objective, for a feasible X. Trace of a positive
    # definite matrix, so positive.
    primal_obj = jnp.trace(target @ feasible_x_inv)

    # Evaluating dual at the optimal (possibly non-feasible) X for the given
    # lagrange_multiplier gives a lower bound on the primal objevtive (loss).
    dual_obj = lagrangian_fn(x_matrix, lt)  # Possibly negative.

    duality_gap = primal_obj - dual_obj
    relative_gap = duality_gap / primal_obj

    losses.append(primal_obj)
    dual_obj_vals.append(dual_obj)

    total_time = time.time() - start_time
    time_per_iter = total_time / (num_iters + 1)

    max_mult = 'N/A'
    if lt.lagrange_multiplier is not None:
      max_mult = f'{jnp.max(lt.lagrange_multiplier):7.2f}'
    max_nn_mult = 'N/A'
    if lt.nonneg_multiplier is not None:
      max_nn_mult = f'{jnp.max(lt.nonneg_multiplier):7.2f}'
    log_str = (
        f'{num_iters:5d}  primal obj {primal_obj:8.2f} dual obj'
        f' {dual_obj:8.2f} gap {duality_gap:8.2f} relative {relative_gap:7.2g},'
        f' min(x)={jnp.min(x_matrix):8.6f}, max v {max_mult}, max nonneg v'
        f' {max_nn_mult}, max/min act sens^2 {active_sens_squared:10.6f},'
        f' {min_sens_squared:10.6f}'
        + f'total {total_time:.0f} seconds, or {time_per_iter:.2f} per iter'
    )
    logging.info(log_str)
    loss_structure = collections.OrderedDict(
        # These are displayed alphabetically in tensorboard, so
        # try to group in a natural way
        duality_gap=duality_gap,
        duality_gap_relative=relative_gap,
        objective_val_dual=dual_obj,
        objective_val_primal=primal_obj,
        sensitivity_squared_max=active_sens_squared,
        sensitivity_squared_min=min_sens_squared,
        x_matrix_max=jnp.max(x_matrix),
        x_matrix_min=jnp.min(x_matrix),
    )
    if lt.lagrange_multiplier is not None:
      loss_structure['lagrange_multiplier_min'] = jnp.min(
          lt.lagrange_multiplier
      )
      loss_structure['lagrange_multiplier_max'] = jnp.max(
          lt.lagrange_multiplier
      )
    loss_type = tff.types.type_from_tensors(loss_structure)
    for manager in metric_release_managers:
      asyncio.run(manager.release(loss_structure, loss_type, num_iters))

    assert duality_gap >= -1e-10, (
        f'duality_gap {duality_gap}, dual_obj {dual_obj}, primal_obj'
        f' {primal_obj}'
    )

    if program_state_manager is not None:
      # Saving after we've updated num_iters ensures that we don't attempt to
      # overwrite existing state.
      asyncio.run(program_state_manager.save(get_state_structure(), num_iters))

    if relative_gap <= target_relative_duality_gap:
      logging.info('Reached target duality gap.')
      break

    if num_iters >= max_iterations:
      logging.info('Reached maximum number of iterations.')
      break

  w_matrix, h_matrix = loops.w_and_h_from_x_and_s(feasible_x_matrix, s_matrix)
  if lt.nonneg_multiplier is not None:
    assert jnp.all(feasible_x_matrix >= 0), jnp.min(feasible_x_matrix)
  assert jnp.all(jnp.isfinite(w_matrix))
  assert jnp.all(jnp.isfinite(h_matrix))
  return collections.OrderedDict(
      W=w_matrix,
      H=h_matrix,
      U=lt.u_total(),
      losses=losses,
      dual_obj_vals=dual_obj_vals,
      n_iters=num_iters,
      lagrange_terms=lt,
      x_matrix=x_matrix,
      relative_duality_gap=relative_gap,
  )
