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
"""Tests for secret_sharer."""

import string

from absl import flags
from absl.testing import absltest

from differential_privacy import run_stackoverflow_with_secrets

FLAGS = flags.FLAGS


class RunStackoverflowWithSecretsTest(absltest.TestCase):

  def test_permute_and_batch(self):

    def to_histogram(l):
      hist = {}
      for i in l:
        c = hist.get(i, 0)
        hist[i] = c + 1
      return hist

    num_values = 8
    batch_size = 3
    for seed in range(10):
      values = list(string.ascii_lowercase)[:num_values]
      permute_and_batch = run_stackoverflow_with_secrets.PermuteAndBatch(
          values, seed, batch_size)

      data_so_far = []
      for i in range(10):
        batch = permute_and_batch(i)
        self.assertLen(batch, batch_size)
        data_so_far.extend(batch)
        hist = to_histogram(data_so_far)
        hist_hist = to_histogram(hist.values())
        data_len = len(data_so_far)
        if data_len < num_values or data_len % num_values == 0:
          self.assertLen(hist_hist, 1)
        else:
          self.assertLen(hist_hist, 2)
        hist_count = sum(k * v for k, v in hist_hist.items())
        self.assertEqual(hist_count, data_len)

        if i > 0:
          recomputed_last_batch = permute_and_batch(i - 1)
          self.assertSequenceEqual(recomputed_last_batch, last_batch)
        last_batch = batch


if __name__ == '__main__':
  absltest.main()
