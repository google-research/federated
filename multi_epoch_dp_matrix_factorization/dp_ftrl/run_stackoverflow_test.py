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
"""Tests for run_stackoverflow."""

import os

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver
import numpy as np
import pandas as pd
import tensorflow as tf

from multi_epoch_dp_matrix_factorization import matrix_constructors
from multi_epoch_dp_matrix_factorization import matrix_io
from multi_epoch_dp_matrix_factorization.dp_ftrl import run_stackoverflow

FLAGS = flags.FLAGS

MOMENTUM = 0.95
MATRIX_NAME = f'lr_momentum_0p{100*MOMENTUM:.0f}'

COMMON_FLAGS = dict(
    vocab_size=100,
    embedding_size=8,
    latent_size=8,
    num_validation_examples=10,
    noise_multiplier=0.0,
    use_synthetic_data=True,
    run_name='run_name',
)


class RunStackoverflowTest(tf.test.TestCase):

  @flagsaver.flagsaver(
      **COMMON_FLAGS,
      total_epochs=1,
      total_rounds=1,
      clients_per_round=2,
      aggregator_method='lr_momentum_matrix',
      lr_momentum_matrix_name=MATRIX_NAME,
      server_momentum=MOMENTUM,
  )
  def test_run_a_few_rounds(self):
    matrix_root_dir = self.create_tempdir('matrix_root').full_path
    matrix_subdir = f'{matrix_root_dir}/{MATRIX_NAME}/size={FLAGS.total_rounds}'

    def _write(filename, value):
      tf.io.write_file(
          os.path.join(matrix_subdir, filename), tf.io.serialize_tensor(value)
      )

    learning_rates = np.linspace(1.0, 0.1, num=FLAGS.total_rounds)
    w_matrix = matrix_constructors.momentum_sgd_matrix(
        FLAGS.total_rounds, MOMENTUM, learning_rates
    )
    h_matrix = np.eye(FLAGS.total_rounds)
    _write(matrix_io.LR_VECTOR_STRING, learning_rates)
    _write(matrix_io.W_MATRIX_STRING, w_matrix)
    _write(matrix_io.H_MATRIX_STRING, h_matrix)

    FLAGS.matrix_root_path = matrix_root_dir
    FLAGS.root_output_dir = self.create_tempdir('temp_output').full_path

    run_stackoverflow.train_and_eval()

  @flagsaver.flagsaver(
      **COMMON_FLAGS,
      aggregator_method='dp_sgd',
      clients_per_round=1,
      reshuffle_each_epoch=False,
      server_momentum=MOMENTUM,
      server_optimizer_lr_cooldown=True,
      total_epochs=3,
      total_rounds=7,
      zero_large_updates=False,
  )
  def test_fewer_rounds_than_epochs_cooldown_and_dont_zero(self):
    FLAGS.root_output_dir = self.create_tempdir('temp_output').full_path
    run_stackoverflow.train_and_eval()
    hparams = pd.read_csv(
        os.path.join(
            FLAGS.root_output_dir, 'results', FLAGS.run_name, 'hparams.csv'
        ),
        header=0,
        index_col=0,
    ).transpose()
    self.assertEqual(int(hparams['total_epochs'].item()), 3)
    self.assertEqual(hparams['aggregator_method'].item(), 'dp_sgd')

    metrics = pd.read_csv(
        os.path.join(
            FLAGS.root_output_dir,
            'results',
            FLAGS.run_name,
            'experiment.metrics.csv',
        )
    )

    # The test data has 3 clients, so at 1 client_per_round,
    # an epoch is 3 rounds, with total_epochs=3, we can have up to
    # 9 rounds: [(0, 1, 2), (3, 4, 5), (6, 7 ,8)].
    # However, we test that if we ask for 7 total rounds,
    # we get that (with test metrics after round 7).
    self.assertEqual(metrics.round_num.max(), FLAGS.total_rounds)
    # See comment in training_loops, training metrics are
    # actually associated with final_round_num - 1 as rounds
    # are zero-indexed.
    last_train_metrics = metrics[metrics.round_num == FLAGS.total_rounds - 1]
    # Note epoch counts are also zero-indexed, so 2 is the 3rd epoch.
    self.assertEqual(int(last_train_metrics['train/epoch'].item()), 2)
    test_metrics = metrics[metrics.round_num == FLAGS.total_rounds]
    self.assertGreater(float(test_metrics['test/evaluate_secs'].item()), 0)
    self.assertTrue(
        np.isfinite(float(test_metrics['test/accuracy_no_oov_or_eos'].item()))
    )

  @flagsaver.flagsaver(
      **COMMON_FLAGS,
      total_epochs=6,
      total_rounds=2048,
      clients_per_round=2,
      reshuffle_each_epoch=False,
      aggregator_method='lr_momentum_matrix',
      lr_momentum_matrix_name=MATRIX_NAME,
      server_momentum=MOMENTUM,
  )
  def test_raises_bad_sensivity(self):
    matrix_root_dir = self.create_tempdir('matrix_root').full_path
    matrix_subdir = f'{matrix_root_dir}/{MATRIX_NAME}/size={FLAGS.total_rounds}'

    def _write(filename, value):
      tf.io.write_file(
          os.path.join(matrix_subdir, filename), tf.io.serialize_tensor(value)
      )

    learning_rates = np.linspace(1.0, 0.1, num=FLAGS.total_rounds)
    w_matrix = matrix_constructors.momentum_sgd_matrix(
        FLAGS.total_rounds, MOMENTUM, learning_rates
    )
    h_matrix = np.eye(FLAGS.total_rounds)
    _write(matrix_io.LR_VECTOR_STRING, learning_rates)
    _write(matrix_io.W_MATRIX_STRING, w_matrix)
    _write(matrix_io.H_MATRIX_STRING, h_matrix)

    FLAGS.matrix_root_path = matrix_root_dir
    FLAGS.root_output_dir = self.create_tempdir('temp_output').full_path
    # With 6 epochs, the sensitivity of the identity matrix will be
    # sqrt(6) = 2.44...
    with self.assertRaisesRegex(
        ValueError, r'Expected sensitivity <= 1.0, but calculated 2\.44'
    ):
      run_stackoverflow.train_and_eval()


if __name__ == '__main__':
  absltest.main()
