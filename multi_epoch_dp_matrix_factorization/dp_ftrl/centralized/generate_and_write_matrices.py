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
"""Generates and writes matrix factorizations to disk for experiments."""

import collections
from collections.abc import Mapping
import enum
import math
import os
from typing import Optional

from absl import app
from absl import flags
from absl import logging
from jax import numpy as jnp
import numpy as np
import scipy
import tensorflow as tf

from multi_epoch_dp_matrix_factorization import matrix_io
from multi_epoch_dp_matrix_factorization.fft import generate_noise
from multi_epoch_dp_matrix_factorization.multiple_participations import contrib_matrix_builders

MatrixMechanismType = tuple[np.ndarray, np.ndarray]
LossType = Mapping[str, float]
MatrixMechanismWithLossType = tuple[np.ndarray, np.ndarray, LossType]


class MechanismType(enum.Enum):
  OPTIMAL = 'optimal'
  ONLINE_TREE = 'online_tree'
  FULL_TREE = 'full_tree'
  OPTIMAL_FFT = 'optimal_fft'
  FFT = 'fft'


_MECHANISM = flags.DEFINE_enum_class(
    'mechanism',
    MechanismType.OPTIMAL,
    MechanismType,
    'Mechanism type to generate matrix for.',
)
_BASE_MECHANISM_PATH = flags.DEFINE_string(
    'base_mechanism_path',
    None,
    required=True,
    help=(
        'Path to base matrix used. If using FFT Mechanisms, this is unused.'
    ),
)
_TARGET_STEPS = flags.DEFINE_integer(
    'target_steps',
    None,
    required=True,
    help='Number of total steps to factorize for.',
)
_TARGET_PARTICIPATIONS = flags.DEFINE_integer(
    'target_participations',
    None,
    required=True,
    help='Number of participations to factorize for.',
)
_NUM_STAMPS = flags.DEFINE_integer(
    'num_stamps', 1, 'Number of stamps. Only used for FFT mechanisms.'
)
_OUT_MECHANISM_FOLDER = flags.DEFINE_string(
    'out_mechanism_folder',
    None,
    'The folder to save the mechanism to.',
    required=True,
)
_OUT_MECHANISM_NAME = flags.DEFINE_string(
    'out_mechanism_name',
    None,
    (
        'Name of the mechanism file. '
        'Defaults to the input file name appended with the target geometry.'
    ),
)


def _load_from_path_and_throwaway_lr(path):
  b, c, _ = matrix_io.load_w_h_and_maybe_lr(path)
  return b.numpy(), c.numpy()


def _compute_pinv(matrix: np.ndarray) -> np.ndarray:
  """Computs pseudo-inverse in Jax to leverage accelerators."""
  if matrix.shape[0] == matrix.shape[1]:
    # Assume it's invertible. This assumption is not good in general,
    # but since we're always interested in factorizing full-rank square
    # matrices, this should be true if the matrix is square.
    inverted_jnp_array = jnp.linalg.inv(jnp.array(matrix))
    return np.array(inverted_jnp_array)
  else:
    return np.array(jnp.linalg.pinv(jnp.array(matrix)))


def _sensitivity_for_c(c: np.ndarray, contrib_matrix: np.ndarray):
  """Sensitivity for the encoder C matrix given `contrib_matrix`."""
  x_matrix = c.T @ c
  return np.max(np.diag(contrib_matrix.T @ x_matrix @ contrib_matrix)) ** 0.5


def _stamp_matrices(
    b: np.ndarray,
    c: np.ndarray,
    num_repetitions: int,
    repeat_b: bool = False,
    workload: Optional[np.ndarray] = None,
) -> MatrixMechanismType:
  """Restarts matrices via Kronaker product, optionally taking inverse for B."""
  repeated_c = np.kron(np.eye(num_repetitions, dtype=c.dtype), c)
  if repeat_b:
    repeated_b = np.kron(np.eye(num_repetitions, dtype=b.dtype), b)
  else:
    if workload is None:
      raise ValueError(
          'If not repeating b, and instead taking an inverse, '
          'the workload must be specified.'
      )
    repeated_b = workload @ _compute_pinv(repeated_c)
  return repeated_b, repeated_c


def _make_full_tree(full_c: np.ndarray, num_steps: int) -> MatrixMechanismType:
  """Makes matrices for Full Honaker methods."""
  next_pwr_of_two = int(2 ** math.ceil(math.log2(num_steps)))

  num_leaves = next_pwr_of_two  # Alias for ease of reasoning
  num_tree_elements = 2 * num_leaves - 1

  subselected_c = full_c[:num_tree_elements, :num_steps]
  prefix_sum_matrix = np.tril(np.ones(shape=(num_steps, num_steps)))
  constructed_b = prefix_sum_matrix @ np.linalg.pinv(subselected_c)
  return constructed_b, subselected_c


def _make_online_tree_completed_matrices(
    full_b: np.ndarray, full_c: np.ndarray, num_steps: int
) -> MatrixMechanismType:
  """Makes matrices for Online Honaker with tree completion."""
  next_pwr_of_two = int(2 ** math.ceil(math.log2(num_steps)))
  num_leaves = next_pwr_of_two  # Alias for ease of reasoning
  num_tree_elements = 2 * num_leaves - 1

  # Select the appropriate complete subtree factorization from B and C
  subselected_c = full_c[:num_tree_elements, :next_pwr_of_two]
  subselected_b = full_b[:next_pwr_of_two, :num_tree_elements]

  # Grab the last row of B; this is what will be used in the 'tree completion'
  # step, which finishes the epoch.
  completion_row = subselected_b[-1, :]

  # Compute mask corresponding to nodes in the tree that either contain
  # observed gradients, or don't---so called virtual gradients.
  num_zero_grads = subselected_c.shape[1] - num_steps
  grad_indicator = np.array([1.0] * num_steps + [0.0] * num_zero_grads)
  zero_embedded_indicator = subselected_c @ grad_indicator
  virtual_grad_indicator = zero_embedded_indicator == 0

  # Mask the elements of the last row which correspond to virtual gradients,
  # to prevent adding unnecessary noise.
  zeros = np.zeros_like(completion_row)
  completion_row_zeroed = np.where(
      virtual_grad_indicator, zeros, completion_row
  )
  completion_row_zeroed = np.reshape(completion_row_zeroed, [1, -1])
  b_without_last_row = subselected_b[: num_steps - 1]
  tree_completed_b = np.concatenate(
      [b_without_last_row, completion_row_zeroed], axis=0
  )

  # Further filter down C for good measure
  truncated_c = subselected_c[:, :num_steps]

  # Make sure the factorized matrices generate the prefix sum.
  target_prefix_sum = np.tril(np.ones(shape=(num_steps, num_steps)))
  np.testing.assert_allclose(
      tree_completed_b @ truncated_c, target_prefix_sum, atol=1e-10
  )
  return tree_completed_b, truncated_c


def _restart_tree_matrices(
    b: np.ndarray, c: np.ndarray, num_repetitions: int, online: bool = True
) -> MatrixMechanismType:
  """Restarts Tree matrices, either `online` or not, `num_repetitions` times."""
  new_num_steps = b.shape[0] * num_repetitions
  target_matrix = np.tril(np.ones(shape=(new_num_steps, new_num_steps)))

  if not online:
    repeated_b, repeated_c = _stamp_matrices(
        b, c, num_repetitions, repeat_b=False, workload=target_matrix
    )
  else:
    num_original_steps = b.shape[0]
    num_original_cols = b.shape[1]

    # If online, the diagonal elements are just stamps.
    repeated_b, repeated_c = _stamp_matrices(
        b, c, num_repetitions, repeat_b=True
    )

    # The final b row will be *repeated* along the 0th dimension; this
    # corresponds to 'reusing' the final estimate as a fixed value in the
    # stamped mechanism.
    final_b_row = np.reshape(b[-1, :], [1, -1])
    final_b_row_block = np.concatenate(
        [final_b_row] * num_original_steps, axis=0
    )

    # Stamp out this final row block in the repeated matrix.
    for row_block_idx in range(1, num_repetitions):
      for col_block_idx in range(row_block_idx):
        row_start = row_block_idx * num_original_steps
        row_end = (row_block_idx + 1) * num_original_steps
        col_start = col_block_idx * num_original_cols
        col_end = (col_block_idx + 1) * num_original_cols
        repeated_b[row_start:row_end, col_start:col_end] = final_b_row_block

  # Ensure the matrix generates the target prefix sum.
  np.testing.assert_allclose(repeated_b @ repeated_c, target_matrix, atol=1e-10)
  return repeated_b, repeated_c


def _fft_decoder_matrix(num_steps):
  fft_dim = int(num_steps * 2)
  fft_matrix = scipy.linalg.dft(fft_dim, 'sqrtn')
  fft_matrix = np.matrix(fft_matrix).H
  fft_matrix = fft_matrix @ (
      np.eye(fft_dim) * np.sqrt(generate_noise._get_dft_vector(num_steps))  # pylint: disable=protected-access
  )
  return fft_matrix


def _fft_encoder_matrix(num_steps):
  return generate_noise._generate_c_matrix_by_fft(num_steps)  # pylint: disable=protected-access


def _make_fft_matrices(
    num_steps, num_stamps: int = 1, optimal: bool = False
) -> MatrixMechanismType:
  """Makes FFT matrices optionally with stamping if `num_stamps` >= 1."""
  if num_stamps <= 0:
    raise ValueError(
        f'Stamps must be >=1, got: {num_stamps}. Stamps=1 is no stamping.'
    )
  fft = scipy.linalg.dft(num_steps * 2, scale='sqrtn')
  ifft = np.matrix(fft).H

  c = _fft_encoder_matrix(num_steps)
  c = (ifft @ c).real[:, :num_steps]  # Equivalent reals encoder
  c = np.kron(np.eye(num_stamps), c)

  if optimal:
    target = np.tril(np.ones(shape=(num_steps, num_steps)))
    b = target @ np.linalg.pinv(c)

  else:
    b = _fft_decoder_matrix(num_steps)  # generated base decoder

    b = (b @ fft).real[:num_steps, :]  # Get reals equivalent
    constant_b = np.tile(b[-1], len(b)).reshape(b.shape)

    # Stamp and fix the output of the last row for each block.
    b = np.kron(np.eye(num_stamps), b)
    already_added = np.eye(num_stamps)
    below_diag = np.tril(np.ones_like(already_added)) - already_added
    repeated_constants = np.kron(below_diag, constant_b)
    b += repeated_constants
  return b, c


def _normalize_sensitivity(
    b: np.ndarray, c: np.ndarray, num_steps: int, num_epochs: int
) -> MatrixMechanismWithLossType:
  """Normalizes sensitivity of matrices for multiple participations.

  In particular, performs the following:
    * Computes sensitivity for num_epochs epochs and n rounds. Note that this
      involves potentially 'truncating' the mechanism, i.e., computing
      sensitivity only over the first some slice, if the repeated mechanism
      results in more iterations than will be used (because the number of steps
      in the existing mechanism does not evenly divide the intended target).
    * Normalizes the resulting factorization to sensitivity 1

  Args:
    b: B decoder matrix. Also named w/W elsewhere.
    c: C encoder matrix. Also named h/H elsewhere.
    num_steps: number of steps desired to run for.
    num_epochs: number of epochs desired to run for.

  Returns:
    A tuple with the resulting normalized b,c matrices and the loss data.
    The loss data is itself a mapping with the keys:
      c_sens2: The squared sensitivity after normalization.
      b_var2: The squared variance after normalization.
      loss2: The squared loss after normalization.
      sens_orgi: The original sensitivity of the C matrix, before normalization.
  """
  all_positive = np.all(c.T @ c >= -1e-9)
  # Calculate the encoder's sensitivity depending on its participation pattern
  # and the constraints present.
  if all_positive:
    all_positive_contrib_matrix = (
        contrib_matrix_builders.epoch_participation_matrix_all_positive(
            n=num_steps, num_epochs=num_epochs
        )
    )
    sens = _sensitivity_for_c(c, contrib_matrix=all_positive_contrib_matrix)
  else:
    logging.info('Using general spectral norm bound.')
    sens = generate_noise.get_spectral_norm_sensitivity(
        c, num_epochs=num_epochs, num_steps=num_steps
    )

  normalized_c = c / sens
  normalized_b = b * sens

  # Compute some statistics about these matrices to pass back to our callers.
  b_var2 = np.linalg.norm(normalized_b[:num_steps, :]) ** 2

  # Notice that the below should be 1 up to numerical precision.
  if all_positive:
    c_sens2 = (
        _sensitivity_for_c(
            normalized_c, contrib_matrix=all_positive_contrib_matrix
        )
        ** 2
    )
  else:
    c_sens2 = (
        generate_noise.get_spectral_norm_sensitivity(
            normalized_c, num_epochs=num_epochs, num_steps=num_steps
        )
        ** 2
    )

  loss_data = collections.OrderedDict(
      c_sens2=c_sens2, b_var2=b_var2, loss2=b_var2 * c_sens2, sens_orig=sens
  )
  return normalized_b, normalized_c, loss_data


def main(*_) -> None:
  mechanism = _MECHANISM.value
  num_steps = _TARGET_STEPS.value
  num_parts = _TARGET_PARTICIPATIONS.value
  target_matrix = np.tril(np.ones(shape=(num_steps, num_steps)))

  if mechanism is MechanismType.OPTIMAL:
    b, c = _load_from_path_and_throwaway_lr(_BASE_MECHANISM_PATH.value)
    num_stamps = math.ceil(num_steps / c.shape[1])
    stamped_b, stamped_c = _stamp_matrices(
        b, c, num_stamps, repeat_b=False, workload=target_matrix
    )
    stamped_c = stamped_c[:, :num_steps]  # Ensure `num_steps` long.

    b, c, loss_data = _normalize_sensitivity(
        stamped_b, stamped_c, num_steps, num_parts
    )

  elif mechanism is MechanismType.ONLINE_TREE:
    b, c = _load_from_path_and_throwaway_lr(_BASE_MECHANISM_PATH.value)
    b, c = _make_online_tree_completed_matrices(b, c, num_steps)
    b, c = _restart_tree_matrices(b, c, num_steps, online=True)

    # This will bypass restarts because of the above methods.
    b, c, loss_data = _normalize_sensitivity(b, c, num_steps, num_parts)

  elif mechanism is MechanismType.FULL_TREE:
    b, c = _load_from_path_and_throwaway_lr(_BASE_MECHANISM_PATH.value)
    b, c = _make_full_tree(c, num_steps)
    b, c = _restart_tree_matrices(b, c, num_steps, online=True)

    # This will bypass restarts because of the above methods.
    b, c, loss_data = _normalize_sensitivity(b, c, num_steps, num_parts)

  elif mechanism is MechanismType.FFT:
    num_stamps = _NUM_STAMPS.value
    b, c = _make_fft_matrices(num_steps, num_stamps)
    b, c, loss_data = _normalize_sensitivity(b, c, num_steps, num_parts)

  elif mechanism is MechanismType.OPTIMAL_FFT:
    num_stamps = _NUM_STAMPS.value
    b, c = _make_fft_matrices(num_steps, num_stamps, optimal=True)
    b, c, loss_data = _normalize_sensitivity(b, c, num_steps, num_parts)

  logging.info('Generated mechanism: %s with loss: %f.', mechanism, loss_data)

  if _OUT_MECHANISM_NAME.value is not None:
    mechanism_name = _OUT_MECHANISM_NAME.value
  else:
    mechanism_name = (
        f'{mechanism.value}_steps={num_steps}_parts={num_parts}'
        f'_stamps={num_stamps}'
    )

  outfile = os.path.join(_OUT_MECHANISM_FOLDER.value, mechanism_name)
  matrix_io.verify_and_write(
      tf.convert_to_tensor(b),
      tf.convert_to_tensor(c),
      tf.constant(target_matrix),
      outfile,
  )


if __name__ == '__main__':
  app.run(main)
