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

import collections
import os
import os.path

import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff

from generalization.utils import metric_utils


def _create_scalar_metrics():
  return collections.OrderedDict([
      ('a', {
          'b': 1.0,
          'c': 2.0,
      }),
      ('val_a', {
          'b': 1.0,
          'c': 2.0,
      }),
  ])


def _create_nonscalar_metrics():
  return collections.OrderedDict([
      ('a', {
          'b': tf.ones([1]),
          'c': tf.zeros([2, 2]),
      }),
      ('val_a', {
          'b': tf.ones([1]),
          'c': tf.zeros([2, 2]),
      }),
  ])


def _create_scalar_metrics_with_extra_column():
  metrics = _create_scalar_metrics()
  metrics['a']['d'] = 3.0
  return metrics


class AtomicCSVLoggerCallbackTest(tf.test.TestCase):

  def test_initializes(self):
    tmpdir = self.get_temp_dir()
    logger = metric_utils.AtomicCSVLoggerCallback(tmpdir)
    self.assertIsInstance(logger, tf.keras.callbacks.Callback)

  def test_writes_dict_as_csv(self):
    tmpdir = self.get_temp_dir()
    logger = metric_utils.AtomicCSVLoggerCallback(tmpdir)
    logger.on_epoch_end(epoch=0, logs={'value': 0, 'value_1': 'a'})
    logger.on_epoch_end(epoch=1, logs={'value': 2, 'value_1': 'b'})
    logger.on_epoch_end(epoch=2, logs={'value': 3, 'value_1': 'c'})
    logger.on_epoch_end(epoch=1, logs={'value': 4, 'value_1': 'd'})
    read_logs = pd.read_csv(
        os.path.join(tmpdir, 'experiment.metrics.csv'),
        index_col=0,
        header=0,
        engine='c')
    self.assertNotEmpty(read_logs)
    pd.testing.assert_frame_equal(
        read_logs, pd.DataFrame({
            'value': [0, 4],
            'value_1': ['a', 'd'],
        }))


class WriteHparamsTest(tf.test.TestCase):

  def test_write_hparams_writes_to_correct_csv(self):
    root_output_dir = self.get_temp_dir()
    experiment_name = 'test'
    hparams = {'a': 1, 'b': 'foo'}
    hparam_file = os.path.join(root_output_dir, 'results', experiment_name,
                               'hparams.csv')
    metric_utils.write_hparams(hparams, root_output_dir, experiment_name)
    self.assertTrue(tf.io.gfile.exists(hparam_file))

  def test_write_hparams_writes_to_correct_tensorboard_dir(self):
    root_output_dir = self.get_temp_dir()
    experiment_name = 'test'
    hparams = {'a': 1, 'b': 'foo'}
    metric_utils.write_hparams(hparams, root_output_dir, experiment_name)

    summary_dir = os.path.join(root_output_dir, 'logdir', experiment_name)
    self.assertTrue(tf.io.gfile.exists(summary_dir))
    self.assertLen(tf.io.gfile.listdir(summary_dir), 1)


class ConfigureManagersTest(tf.test.TestCase):

  def test_checkpoint_manager_saves_to_correct_dir(self):
    root_output_dir = self.get_temp_dir()
    experiment_name = 'test'
    program_state_manager, _ = metric_utils.configure_default_managers(
        root_output_dir, experiment_name)
    self.assertIsInstance(program_state_manager,
                          tff.program.FileProgramStateManager)

  def test_logging_manager_exists(self):
    root_output_dir = self.get_temp_dir()
    experiment_name = 'test'
    _, metrics_managers = metric_utils.configure_default_managers(
        root_output_dir, experiment_name)
    logging_manager = metrics_managers[0]
    self.assertIsInstance(logging_manager, tff.program.LoggingReleaseManager)

  def test_csv_manager_saves_to_correct_dir(self):
    root_output_dir = self.get_temp_dir()
    experiment_name = 'test'
    _, metrics_managers = metric_utils.configure_default_managers(
        root_output_dir, experiment_name)
    csv_manager = metrics_managers[1]
    self.assertIsInstance(csv_manager, tff.program.CSVFileReleaseManager)

    expected_metrics_file = os.path.join(root_output_dir, 'results',
                                         experiment_name,
                                         'experiment.metrics.csv')
    self.assertEqual(csv_manager._file_path, expected_metrics_file)

  def test_default_writer_saves_to_correct_dir(self):
    root_output_dir = self.get_temp_dir()
    experiment_name = 'test'
    _, metrics_managers = metric_utils.configure_default_managers(
        root_output_dir, experiment_name)
    default_writer_manager = metrics_managers[2]
    self.assertIsInstance(default_writer_manager,
                          tff.program.TensorBoardReleaseManager)


if __name__ == '__main__':
  tf.test.main()
