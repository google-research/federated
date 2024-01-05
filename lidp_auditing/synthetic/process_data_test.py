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
import numpy as np
import tensorflow as tf

from lidp_auditing.synthetic import generate_data
from lidp_auditing.synthetic import process_data


class ProcessDataTest(tf.test.TestCase):

  def test_process(self):
    n, k, dim, t = 100, 10, 25, 8
    beta, delta = 0.1, 1e-10
    # Generate data to test
    outs = generate_data.generate_data_numba(n, k, dim, seed=0)
    # Generate statistics
    stat_tpr = outs[0] + outs[2]
    stat_fpr = outs[1] + outs[3]
    # Test
    thresholds = _get_candidate_thresholds(stat_tpr, stat_fpr, t)
    out_vectorized = process_data.get_eps_for_all_thresholds_vectorized(
        stat_tpr, stat_fpr, thresholds, beta, delta
    )
    out_loop = process_data.get_eps_for_all_thresholds_loop(
        stat_tpr, stat_fpr, thresholds, beta, delta
    )
    self.assertEqual(out_vectorized.shape[1], t)
    self.assertEqual(out_loop.shape[1], t)
    self.assertAllClose(out_vectorized, out_loop)


def _get_candidate_thresholds(stat_tpr, stat_fpr, num):
  """Get a few candidate thresholds within the range of the statistics."""
  m1 = min(stat_tpr.min(), stat_fpr.min())
  m2 = max(stat_tpr.max(), stat_fpr.max())
  return np.linspace(0.75 * m1 + 0.25 * m2, 0.25 * m1 + 0.75 * m2, num)


if __name__ == "__main__":
  tf.test.main()
