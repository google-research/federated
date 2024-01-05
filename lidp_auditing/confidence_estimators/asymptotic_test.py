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


class EstimatorTest(tf.test.TestCase):

  def test_estimator(self):
    n = 1000
    k = 10
    xs = np.ones((n, k)) * 0.5
    out = asymptotic.get_asymptotic_confidence_intervals(xs, beta=0.05)
    mean = xs.mean()
    self.assertAllLessEqual(out["left"].to_numpy(), mean)
    self.assertAllGreaterEqual(out["right"].to_numpy(), mean)


if __name__ == "__main__":
  tf.test.main()
