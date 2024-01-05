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

from lidp_auditing.confidence_estimators import asymptotic
from lidp_auditing.confidence_estimators import asymptotic_vectorized


class EstimatorTest(tf.test.TestCase):

  def test_estimator(self):
    n = 1000
    k = 10
    xs_list = [np.ones((n, k)) * 0.25, np.ones((n, k)) * 0.75]
    # Non-vectorized confidence estimators
    loop_out = [
        asymptotic.get_asymptotic_confidence_intervals(xs, beta=0.05)
        for xs in xs_list
    ]
    # Vectorized confidence estimators
    vectorized_out = asymptotic_vectorized.get_asymptotic_confidence_intervals(
        np.stack(xs_list).transpose(1, 2, 0), beta=0.05  # (n, k, batch)
    )
    for i, lo in enumerate(loop_out):
      out1_left, out1_right = lo["left"], lo["right"]
      out2_left, out2_right = vectorized_out[0][i], vectorized_out[1][i]
      self.assertAllClose(out1_left, out2_left)
      self.assertAllClose(out1_right, out2_right)


if __name__ == "__main__":
  tf.test.main()
