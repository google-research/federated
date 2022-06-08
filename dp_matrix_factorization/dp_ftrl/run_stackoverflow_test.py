# Copyright 2022, Google LLC.
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
"""Tests for run_stackoverflow."""

import os

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
import numpy as np
import tensorflow as tf

from dp_matrix_factorization import matrix_constructors
from dp_matrix_factorization.dp_ftrl import aggregator_builder
from dp_matrix_factorization.dp_ftrl import run_stackoverflow

FLAGS = flags.FLAGS

TOTAL_ROUNDS = 1

MOMENTUM = 0.95
MATRIX_NAME = f'lr_momentum_0p{100*MOMENTUM:.0f}'


class RunStackoverflowTest(tf.test.TestCase):

  @flagsaver.flagsaver(
      vocab_size=100,
      embedding_size=8,
      latent_size=8,
      num_validation_examples=10,
      total_epochs=1,
      total_rounds=TOTAL_ROUNDS,
      clients_per_round=2,
      aggregator_method='lr_momentum_matrix',
      lr_momentum_matrix_name=MATRIX_NAME,
      noise_multiplier=0.0,
      use_synthetic_data=True,
      server_momentum=MOMENTUM)
  def test_run_a_few_rounds(self):
    matrix_root_dir = self.create_tempdir('matrix_root').full_path
    matrix_subdir = f'{matrix_root_dir}/{MATRIX_NAME}/size={TOTAL_ROUNDS}'

    def _write(filename, value):
      tf.io.write_file(
          os.path.join(matrix_subdir, filename), tf.io.serialize_tensor(value))

    learning_rates = np.linspace(1.0, 0.1, num=TOTAL_ROUNDS)
    w_matrix = matrix_constructors.momentum_sgd_matrix(TOTAL_ROUNDS, MOMENTUM,
                                                       learning_rates)
    h_matrix = np.eye(TOTAL_ROUNDS)
    _write(aggregator_builder._LR_VECTOR_STRING, learning_rates)
    _write(aggregator_builder._W_MATRIX_STRING, w_matrix)
    _write(aggregator_builder._H_MATRIX_STRING, h_matrix)

    FLAGS.matrix_root_path = matrix_root_dir
    FLAGS.root_output_dir = self.create_tempdir('temp_output').full_path

    run_stackoverflow.train_and_eval()


if __name__ == '__main__':
  absltest.main()
