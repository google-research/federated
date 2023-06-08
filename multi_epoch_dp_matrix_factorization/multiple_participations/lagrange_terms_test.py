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
"""Tests for lagrange_terms."""

from absl.testing import absltest
from jax import numpy as jnp
import numpy as np

from multi_epoch_dp_matrix_factorization.multiple_participations import lagrange_terms


class LagrangeTermsTest(absltest.TestCase):

  def test_lagrange_terms(self):
    n = 8
    contrib_matrix = jnp.eye(n)
    lagrange_multiplier = jnp.arange(1, n + 1)
    lt = lagrange_terms.LagrangeTerms(
        lagrange_multiplier=lagrange_multiplier, contrib_matrix=contrib_matrix
    )
    lt.assert_valid()
    np.testing.assert_allclose(lt.u_total(), jnp.diag(jnp.arange(1, n + 1)))
    self.assertEqual(lt.num_iters, n)

    # Summarizing as a single U-matrix should not change u_total
    lt2 = lagrange_terms.summarize(lt)
    np.testing.assert_allclose(lt.u_total(), lt2.u_total())

  def test_lagrange_terms_vector_and_summary(self):
    n = 8
    lagrange_multiplier = jnp.arange(1, n + 1)
    lt3 = lagrange_terms.LagrangeTerms(
        lagrange_multiplier=lagrange_multiplier,
        contrib_matrix=jnp.eye(n),
        u_matrices=jnp.ones(shape=(1, n, n)),
        u_multipliers=jnp.array([1.3]),
    )
    np.testing.assert_allclose(
        lt3.u_total(),
        jnp.diag(jnp.arange(1, n + 1)) + 1.3 * jnp.ones(shape=(n, n)),
    )
    np.testing.assert_allclose(lt3.multiplier_sum(), 1.3 + n * (n + 1) / 2)
    self.assertEqual(lt3.num_iters, n)

    # Again, summarize shouldn't change u_total.
    lt4 = lagrange_terms.summarize(lt3)
    np.testing.assert_allclose(lt3.u_total(), lt4.u_total())
    np.testing.assert_allclose(lt3.multiplier_sum(), lt4.multiplier_sum())

    # Test non-negative multipliers reflected in u_total.
    lt5 = lagrange_terms.LagrangeTerms(
        lagrange_multiplier=lagrange_multiplier,
        contrib_matrix=jnp.eye(n),
        u_matrices=jnp.ones(shape=(1, n, n)),
        u_multipliers=jnp.array([1.3]),
        nonneg_multiplier=jnp.ones(shape=(n, n)),
    )
    np.testing.assert_allclose(
        lt4.u_total() - jnp.ones(shape=(n, n)), lt5.u_total()
    )


if __name__ == '__main__':
  absltest.main()
