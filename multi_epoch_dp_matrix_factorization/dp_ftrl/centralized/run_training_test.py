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
"""Tests for run_training."""
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
import numpy as np
import tensorflow_datasets as tfds

from multi_epoch_dp_matrix_factorization.dp_ftrl.centralized import grad_processor_builders
from multi_epoch_dp_matrix_factorization.dp_ftrl.centralized import run_training

FLAGS = flags.FLAGS


def _make_cifar10_dataset():
  # Returns mock data for the purpose of testing.
  train = dict(
      image=np.zeros(shape=[100, 32, 32, 3]),
      label=np.ones(shape=[100], dtype=np.int64),
  )
  test = dict(
      image=np.zeros(shape=[100, 32, 32, 3]),
      label=np.ones(shape=[100], dtype=np.int64),
  )
  return {'train': train, 'test': test}


class RunTrainingTest(absltest.TestCase):

  @flagsaver.flagsaver(
      dataset='cifar10',
      mechanism=grad_processor_builders.GradProcessorSpec.NO_PRIVACY,
      batch_size=5,
      epochs=1,
  )
  def test_nonprivate_training_runs(self):
    mock_ds_load = mock.MagicMock()
    mock_ds_load.return_value = _make_cifar10_dataset()

    with mock.patch.object(tfds, 'as_numpy', mock_ds_load), mock.patch.object(
        tfds, 'load'
    ):
      # We patch laod as well to prevent a call out to the network.
      tempdir = self.create_tempdir('root').full_path
      FLAGS.root_output_dir = tempdir
      run_training.main(['binary_test'])


if __name__ == '__main__':
  absltest.main()
