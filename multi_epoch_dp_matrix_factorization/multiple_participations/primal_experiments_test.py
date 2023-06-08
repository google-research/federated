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
"""Tests for primal_experiments."""

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
import numpy as np

from multi_epoch_dp_matrix_factorization import matrix_io
from multi_epoch_dp_matrix_factorization.multiple_participations import primal_experiments

FLAGS = flags.FLAGS


class PrimalExperimentsTest(absltest.TestCase):

  @flagsaver.flagsaver(
      workload='prefix',
      iterations=16,
      num_epochs=4,
      bands=None,
      toeplitz=False,
      dual=False,
  )
  def test_primal_runs_without_error(self):
    primal_experiments.main(['foo'])

    n = FLAGS.iterations
    key = 'mech=4epochs,target=prefix'
    path = matrix_io.get_matrix_path(n, key)
    decoder, encoder, _ = matrix_io.load_w_h_and_maybe_lr(path)
    self.assertEqual(decoder.dtype, np.float64)
    self.assertEqual(encoder.dtype, np.float64)
    self.assertEqual(decoder.shape, (n, n))
    self.assertEqual(encoder.shape, (n, n))

  @flagsaver.flagsaver(
      workload='prefix',
      iterations=16,
      num_epochs=4,
      bands=None,
      toeplitz=False,
      dual=True,
  )
  def test_dual_runs_without_error(self):
    primal_experiments.main(['foo'])

    n = FLAGS.iterations
    key = 'mech=4epochs_dual,target=prefix'
    path = matrix_io.get_matrix_path(n, key)
    decoder, encoder, _ = matrix_io.load_w_h_and_maybe_lr(path)
    self.assertEqual(decoder.dtype, np.float64)
    self.assertEqual(encoder.dtype, np.float64)
    self.assertEqual(decoder.shape, (n, n))
    self.assertEqual(encoder.shape, (n, n))


if __name__ == '__main__':
  absltest.main()
