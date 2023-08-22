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
"""Definition of, and helpers for, the lagrange terms class."""
from typing import Optional

import flax
from jax import numpy as jnp


@flax.struct.dataclass
class LagrangeTerms:
  """Dataclass for summarizing Lagrange multiplier terms.

  All members should be jnp/np arrays so they can be serialized easily.
  """

  # Individual terms v u' X u terms
  # Multiplier for columns of contrib_matrix
  lagrange_multiplier: Optional[jnp.ndarray] = None
  contrib_matrix: Optional[jnp.ndarray] = None  # Each column is a u vectors

  # Summary-U terms  u_multiplier[i] tr( U_i @ X )
  # If u_multipliers is length k, then u_matrices is a (k, n , n) shaped tensor.
  u_multipliers: Optional[jnp.ndarray] = None
  u_matrices: Optional[jnp.ndarray] = None

  # Non-negativity  -tr(nonneg_multiplier @ X)
  nonneg_multiplier: jnp.ndarray = (
      None  # nxn matrix, non-negative  # pytype: disable=annotation-type-mismatch  # jnp-array
  )

  @property
  def num_iters(self):
    """Returns `n`, the number of iterations."""
    n_list = []
    if self.contrib_matrix is not None:
      n_list.append(self.contrib_matrix.shape[0])
    if self.u_matrices is not None:
      n_list.append(self.u_matrices.shape[1])
    if self.nonneg_multiplier is not None:
      n_list.append(self.nonneg_multiplier.shape[0])
    if not n_list:
      raise ValueError('No way to determine num_iters')
    n_list = jnp.array(n_list)
    assert jnp.all(n_list == n_list[0]), n_list
    return n_list[0]

  def u_total(self):
    """Summarize as a single tr(u_total() @ X) term in the Lagrangian."""
    n = self.num_iters
    u_total = jnp.zeros(shape=(n, n))

    if self.lagrange_multiplier is not None:
      assert self.contrib_matrix is not None
      h = self.contrib_matrix
      assert len(self.lagrange_multiplier) == h.shape[1]
      u_total += h @ jnp.diag(self.lagrange_multiplier) @ h.T

    if self.u_matrices is not None:
      assert self.u_multipliers is not None
      u_total += jnp.tensordot(self.u_matrices, self.u_multipliers, axes=(0, 0))

    if self.nonneg_multiplier is not None:
      u_total -= self.nonneg_multiplier

    return u_total

  def assert_valid(self):
    _ = self.num_iters  # Basic shape checks
    assert (self.lagrange_multiplier is not None) or (
        self.u_matrices is not None
    )
    if self.lagrange_multiplier is not None:
      assert jnp.all(self.lagrange_multiplier >= 0), self.lagrange_multiplier
    if self.u_multipliers is not None:
      assert jnp.all(self.u_multipliers >= 0.0)
    if self.nonneg_multiplier is not None:
      assert jnp.all(self.nonneg_multiplier >= 0.0), jnp.min(
          self.nonneg_multiplier
      )

  def multiplier_sum(self):
    s = 0.0
    if self.lagrange_multiplier is not None:
      s += jnp.sum(self.lagrange_multiplier)
    if self.u_multipliers is not None:
      s += sum(self.u_multipliers)
    assert s > 0.0
    return s


def summarize(lt: LagrangeTerms) -> LagrangeTerms:
  """Summarizes vector terms with one new matrix term.

  Intended for use in active-set algorithms, this replaces individual
  multipliers on u-vectors (the lt.langrange_multipliers) with a single
  U-matrix with a new multiplier.
  TODO(b/241453645): Link to details in writeup when available, or remove
  if we end up not using this approach.

  Args:
    lt: The LagrangeTerms to summarize.

  Returns:
    A LagrangeTerms with an additional lt.u_multiplier dual variable.
  """
  assert (
      lt.lagrange_multiplier is not None
  ), 'No per-vector multipliers to summarize'
  n = lt.num_iters
  multiplier_sum = jnp.sum(lt.lagrange_multiplier).item()
  u_matrix = (
      lt.contrib_matrix
      @ jnp.diag(lt.lagrange_multiplier)
      @ lt.contrib_matrix.T  # pytype: disable=attribute-error  # jnp-array
      / multiplier_sum
  )
  u_matrices = u_matrix.reshape((1, n, n))
  u_multipliers = jnp.array([multiplier_sum])
  if lt.u_matrices is not None:
    assert lt.u_multipliers is not None
    u_matrices = jnp.vstack([lt.u_matrices, u_matrices])
    u_multipliers = jnp.concatenate([lt.u_multipliers, u_multipliers])

  return LagrangeTerms(
      lagrange_multiplier=None,
      contrib_matrix=None,
      u_multipliers=u_multipliers,
      u_matrices=u_matrices,
  )


def init_lagrange_terms(contrib_matrix):
  return LagrangeTerms(
      lagrange_multiplier=jnp.ones(contrib_matrix.shape[1]),
      contrib_matrix=contrib_matrix,
  )
