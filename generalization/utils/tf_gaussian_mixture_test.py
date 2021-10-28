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
"""Tests for tf_gaussian_mixture.py."""

from absl.testing import parameterized
import numpy as np
import sklearn.mixture
import tensorflow as tf

from generalization.utils import tf_gaussian_mixture

# pylint: disable=invalid-name


class ModelDeltaProcessTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('random', 'random'),
                                  ('kmeans (full batch)', 'kmeans'))
  def test_tf_gmm_equivalent_to_sklearn_gmm(self, init_params):
    """Test equivalency of jmixture.GaussianMixture vs sklearn.mixture.GaussianMixture."""

    X = np.array(np.random.standard_normal((5, 4)))
    n_components = 3
    random_state = 1

    gm = sklearn.mixture.GaussianMixture(
        n_components=n_components,
        init_params=init_params,
        random_state=random_state)

    tfgm = tf_gaussian_mixture.GaussianMixture(
        n_components=n_components,
        init_params=init_params,
        random_state=random_state)

    gm.fit(X)
    tfgm.fit(X)

    self.assertAllClose(gm.means_, tfgm.means_, rtol=1e-4)
    self.assertAllClose(gm.covariances_, tfgm.covariances_, rtol=1e-4)
    self.assertAllClose(gm.predict(X), tfgm.predict(X), rtol=1e-4)


if __name__ == '__main__':
  tf.test.main()
