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
"""Generates cumulative noise b_i in fourier domain for a specified DPFTRL run.

Let N be the number of mini-batch steps, E the number of epochs, m the number of
data points, b the mini-batch size, L the clipping norm of each individual
gradient (computed per datum).
Also, let epsilon and delta be our privacy parameters for (epsilon, delta)-DP.

With these specified up-front by a practictioner looking to train a
differentially private (DP) model, this package will generate the required
noises b_i for all i in [N] such that adding this noise in cumulative training
will result in an (epsilon, delta)-DP model.

Take SGD which performs the update rule
  theta_i = theta_{i-1} - eta * g_i,
with eta some learning rate and g_i the gradient for minibatch (step) i. We will
term this residual training and augmenting this rule to:
  theta_i = theta_{i-1} - eta * (g_i + b_i - b_{i-1})
will guarantee DP when our noises are added. In cumulative training, the rule:
  theta_i = theta_0 - eta * (theta_{i-1} + g_i + b_i)
will equivalently guarantee DP.

By specifying parameters for a training run upfront, this package generates all
b_i required and stores them in their own files noise_i.npy in the specified
location. Then, these noises can be loaded and added at training time.

Note, clipping of individual gradients must still be performed.
We now describe some important variables used in this generation.

Though our analysis will deal with bounds in the minibatch gradient sums, the
actual contribution of each data is precisely L.

We will perform analysis using zCDP. It suffices to recall that satisfying
rho-zCDP requires adding noise to a gradient sum g_i with variance:
  sigma^2 = sensitivity^2 / (2 * rho).
where we must now calculate this sensitivity.

The sensitivity is the worst-case change if we remove any one data point
(here and in FL, "data point" is synonomous with "user") at a Hamming distance
of 1.
"""
from collections.abc import Sequence
import functools
import warnings

import dp_accounting
import numpy as np

# RDP analysis is over a given order of the Renyi divergence. Bound all these.
_DEFAULT_ORDERS = tuple(np.linspace(1, 32, 100))


@functools.lru_cache(maxsize=1)  # Should always be called on same input arg.
def _get_dft_vector(num_total_steps: int) -> np.ndarray:
  """Gets the (vector) diagonal of the dft representing releasing all sums."""
  # Faster than doing this in numpy first.
  release_all_prefix_sums = np.concatenate(
      [[1] * num_total_steps, [0] * num_total_steps]
  )
  return np.fft.fft(release_all_prefix_sums, norm="backward")  # Unnormalized.


def get_all_noise(
    num_params: int,
    num_total_steps: int,
    noise_scale: float,
    generator_seed: int,
) -> np.ndarray:
  """Gets dft noise across `num_total_steps` at `noise_scale` for `num_params`.

  This method generates noise in the Fourier domain then takes an inverse FFT.

  Args:
    num_params: Int representing the total number of parameters in the model.
    num_total_steps: Int representing the total number of DP-SGD steps to be
      performed in training.
    noise_scale: Float representing the standard deviation of noise to be added,
      i.e., the output of `get_noise_scale_factor`.
    generator_seed: A seed for a pseudo-random number generator. Care must be
      taken because the same seed will give the same output noise, i.e.,
      separate calls to this function almost surely desire new seeds.

  Returns:
    An np.ndarray of shape [`num_total_steps`, `num_params`] of np.float64. This
    is the noise across all num_steps as calculated by the fft approximation.
  """
  dft_sqrt = np.sqrt(_get_dft_vector(num_total_steps))
  generator = np.random.default_rng(generator_seed)
  noise_tuples = generator.normal(
      0, noise_scale, [2 * num_total_steps, 2, num_params]
  )
  # The below list comprehension is significantly faster than np.ndarray.view

  complex_noise = [
      noise_tuple[0, :] + 1j * noise_tuple[1, :] for noise_tuple in noise_tuples
  ]
  z = np.fft.ifft(
      np.multiply(complex_noise, dft_sqrt[..., None]), axis=0, norm="ortho"
  )  # normalize by 1/sqrt(vector_size) on fft and ifft.
  return z.real[:num_total_steps, :]


def get_epsilon_privacy_spent(
    rho: float, delta: float, orders: Sequence[float] = _DEFAULT_ORDERS
) -> float:
  """Gets the epsilon budget given a `rho`-zcdp usage and target `delta`."""
  rdp_epsilons = [order * rho for order in orders]  # Not (epsilon, delta)-DP.
  return dp_accounting.rdp.compute_epsilon(orders, rdp_epsilons, delta=delta)[0]


def _compute_midpoint(a: float, b: float) -> float:
  return (a + b) / 2


def binary_search_rho(
    epsilon: float,
    delta: float,
    epsilon_tolerance: float = 1e-4,
    max_steps: int = 100000,
) -> float:
  """Searches for a rho-zCDP roughly within `epsilon_tolerance` of `epsilon`.

  Args:
    epsilon: The target epsilon to guarantee. Returned `rho` will be less than
      and within `epsilon_tolerance` of `epsilon` unless `max_steps` is reached
      first.
    delta: The target delta to guarantee. Should be at least smaller than 1/N
      where N is the number of records/users.
    epsilon_tolerance: The deviation from `epsilon` that we allow for in the
      returned `rho`.
    max_steps: The max steps to binary search rho.

  Returns:
    `rho`-zCDP guaranteeing (`epsilon`, `delta`)-DP.

  Raises:
    RuntimeError: If the desired `epsilon` was not achieved before `max_steps`.
      This may happen due to the underlying zCDP to (`epsilon`,`delta`)-DP
      calculation being highly nonlinear.
  """
  min_rho, max_rho = 0.0, 1e10
  epsilon_at_rho = get_epsilon_privacy_spent(
      _compute_midpoint(min_rho, max_rho), delta
  )

  step_counter = 0
  check_valid_steps = lambda: step_counter < max_steps
  check_in_tolerance = lambda: abs(epsilon - epsilon_at_rho) < epsilon_tolerance
  check_valid_rho = lambda: max_rho > min_rho

  while check_valid_steps() and not check_in_tolerance() and check_valid_rho():
    current_rho = _compute_midpoint(min_rho, max_rho)
    epsilon_at_rho = get_epsilon_privacy_spent(current_rho, delta)
    if epsilon_at_rho > epsilon:
      max_rho = current_rho
    else:
      min_rho = current_rho
    step_counter += 1

  if not check_in_tolerance():
    if check_valid_steps():
      epsilon_at_rho = get_epsilon_privacy_spent(min_rho, delta)
      raise RuntimeError(
          f"Could not guarantee `epsilon_tolerance`={epsilon_tolerance}. "
          f"Returned `rho` has `epsilon`={epsilon_at_rho}. "
          "Either lower the tolerance or changed the desired epislon."
      )
    else:
      warnings.warn(
          f"`max_steps`={max_steps} reached before `epsilon_tolerance`="
          f"{epsilon_tolerance} achieved. Consider increasing `max_steps`."
      )
  return min_rho


def _get_noise_scale_factor(sensitivity: float, rho: float) -> float:
  """Gets the noise scale for Gaussian IID noise to satisfy `rho`-zCDP.

  To satisfy `rho`-zCDP, a mechanism must output sensitivity^2/(2 * `rho`)
  variance. Thus, the noise scale is the square root of this.

  Args:
    sensitivity: The sensitivity of the mechanism, i.e., how much changing any
      datum can affect the output of the mechanism.
    rho: zCDP privacy budget parameter.

  Returns:
    Noise scale such that sampling Gaussian noise with this standard deviation
    satisfies `rho`-zCDP.
  """
  return sensitivity / np.sqrt(2.0 * rho)


def upper_bound_max_deviation_with_spectral_norm(
    c_matrix: np.ndarray, num_total_steps: int, num_epochs: int
) -> float:
  """Gets `lambda` the max deviation of the query.

  This methods upper bounds the true max deviation of a change in any single
  user's contribution using the spectral norm, which is much quicker and more
  space efficient that exhaustively searching the true max deviation. This,
  however, comes at some cost in utility. Note that we assume a fixed
  user contribution pattern for each epoch, i.e., the data/users can only be
  shuffled prior to the start of the training algorithm.

  This bound is achieved by bounding the deviation of submatrices of the
  `c_matrix` based on the worst-case DP user participation pattern
  (participates anywhere else in the epoch).

  Args:
    c_matrix: The square root of the discrete Fourier transform (DFT) vector
      multiplied with the fast Fourier transform (FFT) matrix.
    num_total_steps: Total number of DP-SGD steps to be performed in training.
    num_epochs: The total number of epochs in DP-SGD training. Defines the
      number of times any single datum will be seen.

  Returns:
    `lambda` the max deviation under a change in any single user datum.
  """
  steps_per_epoch = num_total_steps // num_epochs

  def per_epoch_position_contribution(epoch_position):
    participation_vector = np.arange(
        epoch_position, num_total_steps + epoch_position, steps_per_epoch
    )
    per_epoch_position_stream = c_matrix[:, participation_vector]
    largest_singular_value = np.linalg.norm(per_epoch_position_stream, 2)
    return largest_singular_value

  return np.max(
      [
          per_epoch_position_contribution(epoch_position)
          for epoch_position in range(steps_per_epoch)
      ]
  )


def get_spectral_norm_sensitivity(
    c: np.ndarray, num_steps: int, num_epochs: int
) -> float:
  """Computes spectral norm sensitivity upper bound."""
  if num_epochs > num_steps:
    raise ValueError(
        f"Got `num_epochs`={num_epochs} which must be less than"
        f"`num_steps`={num_steps}."
    )
  if num_steps % num_epochs != 0:
    raise ValueError(
        f"Got `num_steps`={num_steps} not evenly divisible by"
        f"`num_epochs`={num_epochs}.")
  participation_normalization = np.sqrt(num_epochs)
  lambda_ = upper_bound_max_deviation_with_spectral_norm(
      c, num_steps, num_epochs
  )
  return participation_normalization * lambda_


def get_spectral_norm_sensitivity_for_fft(
    c_matrix: np.ndarray, num_epochs: int, kappa: float
) -> float:
  """Computes the upper-bounded sensitivity for DP-FTRL using the spectral norm.

  Linear algebraically, we consider a matrix `c_matrix` (henceforth C). Recall C
  operates on the prefix gradient sums so that a column from C releases a
  particular gradient sum. Consider a submatrix G chosen from C based on the
  participation pattern, i.e., G = C[:, participation_pattern]. The maximum
  scaling of the sensitivity is bounded by the maximum of the spectral norm of
  G * z, where z is in {-1, 1}^`num_epochs`, i.e., max ||Gz||_2. This quantity
  (max ||Gz||_2) is in turn bounded by the max spectral norm for these G
  matricies mutiplied with sqrt(`num_epochs`), where ||z'||_2 <= 1, i.e.,
  sqrt(`num_epochs`) * max ||Gz'||_2 where here ||z'||_2 <= 1.

  Note that we (due to `upper_bound_max_deviation_with_spectral_norm`) assume a
  fixed user contribution pattern for each epoch, i.e., the data/users can only
  be shuffled prior to the start of the training algorithm.

  Args:
    c_matrix: Matrix factorization of the A matrix into B * C, where C is the
      `c_matrix`. Thus, C is the factor applied on the input queries
      (gradients). This is the matrix we are calculating sensitivity for. The
      main C in this package is currently generated by FFT.
    num_epochs: The total number of epochs in DP-SGD training. Defines the
      number of times any single datum will be seen.
    kappa: The (L2) norm bound on the contribution of each datum's gradient to
      the sum.

  Returns:
    The sensitivity of the mechanism.
  """
  dims_not_even = [
      dimension_shape % 2 != 0 for dimension_shape in c_matrix.shape
  ]
  if any(dims_not_even):
    raise ValueError(
        "Expected `c_matrix` to be of shape [2 * num_steps] "
        f"in each dimension. Got shape={c_matrix.shape}."
    )
  if c_matrix.ndim != 2:
    raise ValueError(
        f"Expected a rank-2 matrix for `c_matrix`.Got rank={c_matrix.ndim}"
    )
  if c_matrix.shape[0] != c_matrix.shape[1]:
    raise ValueError(f"Expected square matrix. Got shape={c_matrix.shape}")

  num_steps = len(c_matrix) // 2
  if num_steps % num_epochs != 0:
    raise ValueError(
        "Expected equal integer steps per epoch."
        f"Got: num_steps={num_steps}, num_epochs={num_epochs}."
    )

  # `num_epochs` is the number of times each datum is seen. See the method
  # docstring for an analytical explanation of how these are chosen.
  participation_normalization = np.sqrt(num_epochs)
  lambda_ = upper_bound_max_deviation_with_spectral_norm(
      c_matrix, num_steps, num_epochs
  )

  return lambda_ * participation_normalization * kappa


def _generate_c_matrix_by_fft(num_total_steps: int) -> np.ndarray:
  """Generates `c_matrix` as the fft matrix multiplied with the dft vector.

  The `c_matrix` (henceforth C) represents a linear mapping of the release of
  the cumulative gradients at each step G. Here, this C matrix is calculated
  using the FFT, which achieves small but not optimal L2 error. It is thus
  defined as C = sqrt(Sigma) * F, where Sigma is a zero-matrix with the
  square-root of the dft of the gradient release vector (a half 1s half 0s
  vector whose convolution gives the prefix-sum) on the diagonal and F is the
  normalized FFT matrix.

  Args:
    num_total_steps: Total number of DP-SGD steps to be performed in training.

  Returns:
    C = Sigma * F representing the FFT approximation of the linear mapping onto
      the release of the cumulative gradients at each step, G.
  """
  fft_matrix = np.eye(2 * num_total_steps, dtype=np.complex128)  # F matrix
  fft_matrix = np.fft.fft(fft_matrix, axis=-1, norm="ortho")  # Normalized.
  dft_sqrt = _get_dft_vector(num_total_steps)
  return fft_matrix * dft_sqrt[..., None]  # Multiply across rows.


def generate_noise_for_training_run(
    num_params: int,
    num_training_datums: int,
    num_epochs: int,
    minibatch_size: int,
    l2_norm_clip: float,
    epsilon: float,
    delta: float,
    generator_seed: int,
) -> np.ndarray:
  """Generates the cumulative FFT noises for DPSGD based on training params.

  This method uses the FFT to analyze the L2 error imposed on the prefix
  gradient sums and hence also to compute the noises required for a particular
  (epsilon,delta)-DP guarantee.

  The precomputed noise assumes that noise is added to the minibatch gradient
  sum as follows: sum(grad_i)_j + b_j, where grad_i indicates the clipped
  gradient for datum i and j indicates the j'th minibatch.

  If the mean gradient is used instead, noise should be divided by the
  minibatch size as is common in most DPSGD implementations. Note that this
  returns the cumulative noise, where the residual will be required for native
  DPSGD. This may take significant memory. If needed, `num_training_datums` can
  be reduced and called in parallel however many times needed to generate the
  required number of coordinates of noise.

  Args:
    num_params: Int representing the total number of parameters in the model.
    num_training_datums: Int representing the number of datums in the training
      set. Must be evenly divisible by `minibatch_size`
    num_epochs: Int representing the number of full-passes through the dataset.
    minibatch_size: Int representing how many data samples are in each
      minibatch. Must be a divisor of `num_training_datums`.
    l2_norm_clip: Float representing the clipping value per individual gradient.
      Each individual gradient will be clipped so that its l2 norm is at most
      this value.
    epsilon: Float representing the privacy budget in (epsilon, delta)-DP.
    delta: Float representing the probability of failure in (epsilon, delta)-DP.
    generator_seed: A seed for a pseudo-random number generator. Care must be
      taken because the same seed will give the same output noise, i.e.,
      separate calls to this function almost surely desire new seeds.

  Returns:
    A np.ndarray of shape [num_total_steps, num_params] with the pre-generated
    noise to be added to the (clipped) gradient sum.

  Raises:
    ValueError: If `num_training_datums` is not evenly divisible by
      `minibatch_size`.
  """
  if num_training_datums % minibatch_size != 0:
    raise ValueError(
        "`num_training_datums` was not evenly divisible by `minibatch_size` "
        f"Got `num_training_datums`={num_training_datums}, "
        f"`minibatch_size`={minibatch_size}."
    )

  num_total_steps = num_epochs * (num_training_datums // minibatch_size)
  rho = binary_search_rho(epsilon, delta)

  kappa = l2_norm_clip  # To be explicit about what kappa is set to.
  c_matrix = _generate_c_matrix_by_fft(num_total_steps)

  sensitivity = get_spectral_norm_sensitivity_for_fft(
      c_matrix, num_epochs, kappa
  )
  del c_matrix  # save memory.
  scaling = _get_noise_scale_factor(sensitivity, rho)
  return get_all_noise(num_params, num_total_steps, scaling, generator_seed)


def get_unique_noise_directory(
    epsilon: float,
    delta: float,
    num_epochs: int,
    total_num_steps: int,
    l2_clip: float,
    seed: int,
    restart_period: int,
) -> str:
  """Gets a unique noise directory for parameters that lead to new noises."""
  training_params = [
      epsilon,
      delta,
      num_epochs,
      total_num_steps,
      l2_clip,
      seed,
      restart_period,
  ]
  return "_".join([str(param) for param in training_params])
