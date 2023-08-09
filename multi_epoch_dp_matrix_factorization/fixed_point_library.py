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
"""Functions for computing lagrange multiplier settings for DP-MatFac."""
import jax
from jax import numpy as jnp
from jax.config import config

# With large matrices, the extra precision afforded by performing all
# computations in float64 is critical.
config.update('jax_enable_x64', True)


@jax.jit
def diagonalize_and_take_jax_matrix_sqrt(
    matrix: jnp.ndarray, min_eigenval: float = 0.0
) -> jnp.ndarray:
  """Matrix square root for positive-definite, Hermitian matrices."""
  evals, evecs = jnp.linalg.eigh(matrix)
  eval_sqrt = jnp.maximum(evals, min_eigenval) ** 0.5
  sqrt = evecs @ jnp.diag(eval_sqrt) @ evecs.T
  return sqrt


def init_fixed_point(target):
  """Initializes lagrange_multipliers from target matrix."""
  # TODO(b/233405976): Consider different initializations and document how
  # initialization was chosen.
  target_sqrt = diagonalize_and_take_jax_matrix_sqrt(target)
  return jnp.diag(target_sqrt)


def compute_phi_fixed_point(
    target: jnp.ndarray, lagrange_multiplier: jnp.ndarray, iterations: int
) -> tuple[jnp.ndarray, int, float]:
  """Computes fixed point of phi.

  Defined in Lemma 3.4 of https://arxiv.org/abs/2202.08312.

  Args:
    target: Rank-2 array (IE, matrix) playing the role of S*S in the definition
      of phi. Assumed to be Hermitian and positive-definite.
    lagrange_multiplier: Initialization of the lagrange multipliers.
    iterations: The number of iterations toward the fixed point of phi.

  Returns:
   Updated langrage_multiplier vector.
  """

  if len(target.shape) != 2:
    raise ValueError(
        'Expected `target` argument to be a rank-2 ndarray (IE, a '
        f'matrix); found an array of rank {len(target.shape)}'
    )
  if target.shape[0] != target.shape[1]:
    raise ValueError(
        'Expected target to be a square matrix; found matrix of '
        f'shape {target.shape}'
    )
  if not jnp.all(jnp.isfinite(target)):
    raise ValueError('Cannot compute fixed-point for matrix with nan entries.')

  v = lagrange_multiplier
  target = target.astype(v.dtype)

  for _ in range(iterations):
    diag = jnp.diag(v)
    diag_sqrt = diag**0.5
    v = jnp.diag(
        diagonalize_and_take_jax_matrix_sqrt(diag_sqrt @ target @ diag_sqrt)
    )
    jnp.all(v > 0)

  return v  # pytype: disable=bad-return-type  # jnp-array
