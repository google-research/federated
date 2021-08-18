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
"""Tests for non_iid_histograms."""
from absl.testing import absltest
import numpy as np
import numpy.testing as npt

from analytics.utils import non_iid_histograms


class NonIidHistogramsTest(absltest.TestCase):

  def test_generate_non_iid_poisson_counts(self):
    num_users = 10000
    avg_count = 2
    rng = np.random.RandomState(2021)

    iid_counts = non_iid_histograms.generate_non_iid_poisson_counts(
        num_users, iid_param=0, avg_count=avg_count, rng=rng)
    assert abs(np.var(iid_counts) - avg_count) <= 0.1

    non_iid_counts_dir_4 = non_iid_histograms.generate_non_iid_poisson_counts(
        num_users, iid_param=4, avg_count=avg_count, rng=rng)

    non_iid_counts_dir_10 = non_iid_histograms.generate_non_iid_poisson_counts(
        num_users, iid_param=10, avg_count=avg_count, rng=rng)
    assert np.var(iid_counts) < np.var(non_iid_counts_dir_4) and np.var(
        non_iid_counts_dir_4) < np.var(non_iid_counts_dir_10)

  def test_generate_non_iid_distributions_dirichlet(self):
    rng = np.random.RandomState(2021)
    ref_distribution = np.array([0.1, 0.2, 0.3, 0.4])
    num_users = 100
    distrs_iid = non_iid_histograms.generate_non_iid_distributions_dirichlet(
        num_users, ref_distribution, 0, rng)
    npt.assert_array_equal(distrs_iid, np.tile(ref_distribution,
                                               (num_users, 1)))

    distrs_non_iid_small = non_iid_histograms.generate_non_iid_distributions_dirichlet(
        num_users, ref_distribution, 0.2, rng)
    non_iid_small_var = np.var(distrs_non_iid_small, axis=0)
    assert (non_iid_small_var > 0).all()
    distrs_non_iid_large = non_iid_histograms.generate_non_iid_distributions_dirichlet(
        num_users, ref_distribution, 10, rng)
    non_iid_large_var = np.var(distrs_non_iid_large, axis=0)
    assert (non_iid_small_var < non_iid_large_var).all()

  def test_generate_histograms(self):
    rng = np.random.RandomState(2021)
    ref_distribution = np.array([0.1, 0.2, 0.3, 0.4])
    num_users = 100
    avg_count = 50

    hist_count_iid = non_iid_histograms.generate_histograms(
        num_users, 0, avg_count, ref_distribution, 1, rng)
    iid_count_var = np.var(np.sum(hist_count_iid, axis=1))
    assert iid_count_var < 1.1 * avg_count

    hist_count_non_iid = non_iid_histograms.generate_histograms(
        num_users, 10, avg_count, ref_distribution, 1, rng)
    non_iid_count_var = np.var(np.sum(hist_count_non_iid, axis=1))
    assert iid_count_var < non_iid_count_var

    hist_distr_iid = non_iid_histograms.generate_histograms(
        num_users, 0, avg_count, ref_distribution, 0, rng)
    emp_distr_iid = hist_distr_iid / np.sum(
        hist_distr_iid, axis=1)[:, np.newaxis]
    iid_distr_var = np.var(emp_distr_iid, axis=0)
    assert (iid_distr_var < 0.01).all()

    hist_distr_non_iid = non_iid_histograms.generate_histograms(
        num_users, 0, avg_count, ref_distribution, 10, rng)
    emp_distr_non_iid = hist_distr_non_iid / np.sum(
        hist_distr_non_iid, axis=1)[:, np.newaxis]
    non_iid_distr_var = np.var(emp_distr_non_iid, axis=0)
    assert (iid_distr_var < non_iid_distr_var).all()


if __name__ == '__main__':
  absltest.main()
