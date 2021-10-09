# Copyright 2021, Google LLC.
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
"""Tests for minimal random coding."""

from absl.testing import absltest
import numpy as np
from rcc_dp import get_parameters
from rcc_dp import miracle


class MiracleTest(absltest.TestCase):

  def test_encoder_decoder(self):
    """Test whether the candidate generated at the decoder is correct."""
    eps_space = [1, 2, 3]
    number_candidates_space = [2**3, 2**4]
    d = 100
    n = 100
    budget = 0.99
    for eps in eps_space:
      for number_candidates in number_candidates_space:
        x = np.random.normal(0, 1, (d, n))
        x /= np.linalg.norm(x, axis=0)
        c1, c2, _, gamma = get_parameters.get_parameters_unbiased_miracle(
            eps, d, number_candidates, budget)
        for i in range(n):
          k, z, _ = miracle.encoder(0, x[:, i], number_candidates, c1, c2,
                                    gamma)
          z_k = miracle.decoder(0, k, d, number_candidates)
          # Test if the corresponding candidates are equal
          self.assertListEqual(list(z[:, k]), list(z_k))

if __name__ == "__main__":
  absltest.main()
