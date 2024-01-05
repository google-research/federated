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
import tensorflow as tf

from lidp_auditing.synthetic import generate_data


class GenerateDataTest(tf.test.TestCase):

  def test_generate(self):
    n = 100
    k = 10
    dim = 25
    outs = generate_data.generate_data_numba(n, k, dim, seed=0)
    for out in outs:
      self.assertEqual(out.shape[0], n)
      self.assertEqual(out.shape[1], k)


if __name__ == "__main__":
  tf.test.main()
