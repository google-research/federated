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
"""Tests for generate_noise."""
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from multi_epoch_dp_matrix_factorization.fft import generate_noise


class GenerateNoiseTest(parameterized.TestCase):
  """Tests for noise generation."""

  @parameterized.named_parameters(
      [('scale_0', 0), ('scale_1', 1), ('scale_10', 10)]
  )
  def test_get_all_noise_returns_correct_scale(self, scale):
    """Tests that under unitary matrix, the noise scale should be the same."""
    num_total_steps = 10

    with mock.patch.object(
        generate_noise,
        '_get_dft_vector',
        return_value=np.ones([num_total_steps * 2]),
    ):
      noises = generate_noise.get_all_noise(
          num_params=1000,
          num_total_steps=num_total_steps,
          noise_scale=scale,
          generator_seed=2,
      )

    np.testing.assert_allclose(
        np.std(noises),
        scale,
        rtol=0.01,
        err_msg='Noise standard deviation was too far from chosen scale.',
    )

  @parameterized.named_parameters([
      ('steps_1_params_1', 1, 1),
      ('steps_10_params_1', 10, 1),
      ('steps_1_params_10', 1, 10),
      ('steps_10_params_10', 10, 10),
  ])
  def test_get_all_noise_shape(self, num_total_steps, num_params):
    noises = generate_noise.get_all_noise(
        num_params=num_params,
        num_total_steps=num_total_steps,
        noise_scale=1.0,
        generator_seed=2,
    )

    self.assertSequenceEqual(
        noises.shape,
        (num_total_steps, num_params),
        'Noise shape was not [`num_total_steps`, `num_params`].',
    )

  def test_get_all_noise_type(self):
    noises = generate_noise.get_all_noise(
        num_params=5, num_total_steps=5, noise_scale=1.0, generator_seed=2
    )

    self.assertEqual(noises.dtype, float, 'Noise was not of dtype float.')

  def test_get_training_run_noise_linear_in_kappa(self):
    l2_norm_clip = 1.0
    minibatch_size = 100
    num_params = 10
    num_epochs = 5
    epsilon = 25.0
    delta = 1e-5
    generator_seed = 2
    num_training_datums = 1000

    noises_sum = generate_noise.generate_noise_for_training_run(
        num_params=num_params,
        num_training_datums=num_training_datums,
        num_epochs=num_epochs,
        minibatch_size=minibatch_size,
        l2_norm_clip=l2_norm_clip,
        epsilon=epsilon,
        delta=delta,
        generator_seed=generator_seed,
    )

    noises_mean = generate_noise.generate_noise_for_training_run(
        num_params=num_params,
        num_training_datums=num_training_datums,
        num_epochs=num_epochs,
        minibatch_size=minibatch_size,
        l2_norm_clip=l2_norm_clip / minibatch_size,
        epsilon=epsilon,
        delta=delta,
        generator_seed=generator_seed,
    )

    self.assertAlmostEqual(np.std(noises_sum / 100), np.std(noises_mean))

  def test_get_training_run_noise_epsilon_asymptotic_large(self):
    noises_sum = generate_noise.generate_noise_for_training_run(
        num_params=5,
        num_training_datums=5,
        num_epochs=1,
        minibatch_size=1,
        l2_norm_clip=1,
        epsilon=1e10,
        delta=1e-5,
        generator_seed=2,
    )
    self.assertAlmostEqual(np.std(noises_sum / 100), 0.0, places=6)

  def test_get_training_run_noise_type(self):
    noises_sum = generate_noise.generate_noise_for_training_run(
        num_params=5,
        num_training_datums=5,
        num_epochs=1,
        minibatch_size=1,
        l2_norm_clip=1,
        epsilon=1e10,
        delta=1e-5,
        generator_seed=2,
    )
    self.assertEqual(noises_sum.dtype, float, 'Noise was not of dtype float.')

  @parameterized.named_parameters(
      [('5,1,5', 5, 1, 5), ('1_5_2', 1, 5, 2), ('10_10_10', 10, 10, 10)]
  )
  def test_get_training_run_noise_shape(
      self, num_epochs, minibatch_size, num_params
  ):
    num_training_datums = 1000
    noises_sum = generate_noise.generate_noise_for_training_run(
        num_params=num_params,
        num_training_datums=num_training_datums,
        num_epochs=num_epochs,
        minibatch_size=minibatch_size,
        l2_norm_clip=1,
        epsilon=1e10,
        delta=1e-5,
        generator_seed=2,
    )

    total_steps = int(num_epochs * (num_training_datums / minibatch_size))
    self.assertSequenceEqual(noises_sum.shape, [total_steps, num_params])


class PrivacyParamsTest(parameterized.TestCase):
  """Tests for computing conversions of privacy parameters."""

  @parameterized.named_parameters([('epsilon_2.5', 2.5), ('epsilon_25', 25)])
  def test_binary_search_rho_returns_original_epsilon(self, epsilon):
    delta = 1e-5
    rho = generate_noise.binary_search_rho(
        epsilon, delta=delta, epsilon_tolerance=1e-5
    )
    self.assertAlmostEqual(
        generate_noise.get_epsilon_privacy_spent(rho, delta), epsilon, places=3
    )

  @parameterized.named_parameters([('digits_3', 3), ('digits_7', 7)])
  def test_binary_search_rho_returns_desired_epsilon_tolerance(self, digits):
    delta = 1e-5
    epsilon = 100.0
    # Because the computation of epsilon is non-linear, we are not guaranteed
    # to always hit the tolerance exactly, so we add a 5% relative slack.

    rho = generate_noise.binary_search_rho(
        epsilon,
        delta=delta,
        epsilon_tolerance=10 ** (-(digits * 1.05)),
        max_steps=10**digits,
    )
    self.assertAlmostEqual(
        generate_noise.get_epsilon_privacy_spent(rho, delta),
        epsilon,
        places=digits - 1,
    )

  @parameterized.named_parameters(
      [('delta_1e-3', 1e-3), ('delta_1e-5', 1e-5), ('delta_1e-7', 1e-7)]
  )
  def test_binary_search_rho_lower_delta_increases_rho(self, delta):
    epsilon = 5.0
    rho = generate_noise.binary_search_rho(epsilon=epsilon, delta=delta)

    self.assertGreater(rho, generate_noise.binary_search_rho(rho, delta * 10))

  def test_binary_search_rho_warns_max_steps(self):
    delta = 1e-5
    epsilon = 100.0
    # Because the computation of epsilon is non-linear, we are not guaranteed
    # to always hit the tolerance exactly, so we add a 20% relative slack.

    with self.assertWarnsRegex(
        Warning, '.*[cC]onsider increasing.*max_steps.*'
    ):
      generate_noise.binary_search_rho(
          epsilon, delta=delta, epsilon_tolerance=1e-10, max_steps=2
      )

  @parameterized.named_parameters([('low_epsilon', 1e-2), ('high_epsilon', 50)])
  def test_binary_search_default_epsilon_in_tolerance(self, epsilon):
    epsilon = 5.0
    delta = 1e-5
    rho = generate_noise.binary_search_rho(epsilon=epsilon, delta=delta)

    self.assertAlmostEqual(
        epsilon, generate_noise.get_epsilon_privacy_spent(rho, delta), places=3
    )

  @parameterized.named_parameters(
      [('rank_1', 1), ('rank_5', 5), ('rank_10', 10)]
  )
  def test_spectral_norm_sensitivity_identity_independent_rank(self, rank):
    matrix = np.eye(rank)
    sensitivity = generate_noise.get_spectral_norm_sensitivity(matrix, rank, 1)

    # max is always any column, which has a single non-zero entry.
    self.assertEqual(sensitivity, 1.0)

  @parameterized.named_parameters([('num_epochs_2', 2), ('num_epochs_6', 6)])
  def test_spectral_norm_sensitivity_identity_sqrt_particitipations(
      self, num_epochs
  ):
    rank = 6
    matrix = np.eye(rank)
    sensitivity = generate_noise.get_spectral_norm_sensitivity(
        matrix, rank, num_epochs
    )

    # sensitivity grows by the sqrt of the number of participations (epochs).
    self.assertEqual(sensitivity, np.sqrt(num_epochs))

  def test_spectral_norm_raises_too_many_epochs(self):
    rank = 5
    matrix = np.eye(rank)

    epochs = 10
    with self.assertRaisesRegex(
        ValueError,
        f'.*num_epochs.*{epochs}.*less than.*num_steps.*{rank}.*',
    ):
      generate_noise.get_spectral_norm_sensitivity(matrix, rank, epochs)

  def test_spectral_norm_raises_uneven_epochs(self):
    rank = 5
    matrix = np.eye(rank)

    epochs = 2
    with self.assertRaisesRegex(
        ValueError,
        f'.*num_steps.*{rank}.*not evenly divisible.*num_epochs.*{epochs}.*',
    ):
      generate_noise.get_spectral_norm_sensitivity(matrix, rank, epochs)


if __name__ == '__main__':
  absltest.main()
