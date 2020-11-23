# Copyright 2019, Google LLC.
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
import csv
import os

import tensorflow as tf

from utils import metrics_manager


def _create_dummy_metrics():
  return collections.OrderedDict([
      ('a', {
          'b': 1.0,
          'c': 2.0,
      }),
  ])


def _create_dummy_metrics_with_extra_column():
  metrics = _create_dummy_metrics()
  metrics['a']['d'] = 3.0
  return metrics


class ScalarMetricsManagerTest(tf.test.TestCase):

  def test_metrics_are_appended(self):
    metrics_mngr = metrics_manager.ScalarMetricsManager(self.get_temp_dir())
    _, metrics = metrics_mngr.get_metrics()
    self.assertEmpty(metrics)

    metrics_mngr.update_metrics(0, _create_dummy_metrics())
    _, metrics = metrics_mngr.get_metrics()
    self.assertLen(metrics, 1)

    metrics_mngr.update_metrics(1, _create_dummy_metrics())
    _, metrics = metrics_mngr.get_metrics()
    self.assertLen(metrics, 2)

  def test_update_metrics_returns_flat_dict(self):
    metrics_mngr = metrics_manager.ScalarMetricsManager(self.get_temp_dir())
    input_data_dict = _create_dummy_metrics()
    appended_data_dict = metrics_mngr.update_metrics(0, input_data_dict)
    self.assertEqual({
        'a/b': 1.0,
        'a/c': 2.0,
        'round_num': 0.0
    }, appended_data_dict)

  def test_column_names(self):
    metrics_mngr = metrics_manager.ScalarMetricsManager(self.get_temp_dir())
    metrics_mngr.update_metrics(0, _create_dummy_metrics())
    fieldnames, _ = metrics_mngr.get_metrics()
    self.assertCountEqual(['a/b', 'a/c', 'round_num'], fieldnames)

  def test_update_metrics_adds_column_if_previously_unseen_metric_added(self):
    metrics_mngr = metrics_manager.ScalarMetricsManager(self.get_temp_dir())
    metrics_mngr.update_metrics(0, _create_dummy_metrics())
    fieldnames, metrics = metrics_mngr.get_metrics()
    self.assertCountEqual(fieldnames, ['round_num', 'a/b', 'a/c'])
    self.assertNotIn('a/d', metrics[0].keys())

    metrics_mngr.update_metrics(1, _create_dummy_metrics_with_extra_column())
    fieldnames, metrics = metrics_mngr.get_metrics()
    self.assertCountEqual(fieldnames, ['round_num', 'a/b', 'a/c', 'a/d'])
    self.assertEqual(metrics[0]['a/d'], '')

  def test_update_metrics_adds_empty_str_if_previous_column_not_provided(self):
    metrics_mngr = metrics_manager.ScalarMetricsManager(self.get_temp_dir())
    metrics_mngr.update_metrics(0, _create_dummy_metrics_with_extra_column())
    metrics_mngr.update_metrics(1, _create_dummy_metrics())
    _, metrics = metrics_mngr.get_metrics()
    self.assertEqual(metrics[1]['a/d'], '')

  def test_csvfile_is_saved(self):
    temp_dir = self.get_temp_dir()
    metrics_manager.ScalarMetricsManager(temp_dir, prefix='foo')
    self.assertEqual(set(os.listdir(temp_dir)), set(['foo.metrics.csv']))

  def test_reload_of_csvfile(self):
    temp_dir = self.get_temp_dir()
    metrics_mngr = metrics_manager.ScalarMetricsManager(temp_dir, prefix='bar')
    metrics_mngr.update_metrics(0, _create_dummy_metrics())
    metrics_mngr.update_metrics(5, _create_dummy_metrics())

    new_metrics_mngr = metrics_manager.ScalarMetricsManager(
        temp_dir, prefix='bar')
    fieldnames, metrics = new_metrics_mngr.get_metrics()
    self.assertCountEqual(fieldnames, ['round_num', 'a/b', 'a/c'])
    self.assertLen(metrics, 2, 'There should be 2 rows (for rounds 0 and 5).')
    self.assertEqual(5, metrics[-1]['round_num'],
                     'Last metrics are for round 5.')

    self.assertEqual(set(os.listdir(temp_dir)), set(['bar.metrics.csv']))

  def test_update_metrics_raises_value_error_if_round_num_is_negative(self):
    metrics_mngr = metrics_manager.ScalarMetricsManager(self.get_temp_dir())

    with self.assertRaises(ValueError):
      metrics_mngr.update_metrics(-1, _create_dummy_metrics())

  def test_update_metrics_raises_value_error_if_round_num_is_out_of_order(self):
    metrics_mngr = metrics_manager.ScalarMetricsManager(self.get_temp_dir())

    metrics_mngr.update_metrics(1, _create_dummy_metrics())

    with self.assertRaises(ValueError):
      metrics_mngr.update_metrics(0, _create_dummy_metrics())

  def test_clear_rounds_after_raises_runtime_error_if_no_metrics(self):
    metrics_mngr = metrics_manager.ScalarMetricsManager(self.get_temp_dir())

    # Clear is allowed with no metrics if no rounds have yet completed.
    metrics_mngr.clear_rounds_after(last_valid_round_num=0)

    with self.assertRaises(RuntimeError):
      # Raise exception with no metrics if no rounds have yet completed.
      metrics_mngr.clear_rounds_after(last_valid_round_num=1)

  def test_clear_rounds_after_raises_value_error_if_round_num_is_negative(self):
    metrics_mngr = metrics_manager.ScalarMetricsManager(self.get_temp_dir())
    metrics_mngr.update_metrics(0, _create_dummy_metrics())

    with self.assertRaises(ValueError):
      metrics_mngr.clear_rounds_after(last_valid_round_num=-1)

  def test_rows_are_cleared_and_last_round_num_is_reset(self):
    metrics_mngr = metrics_manager.ScalarMetricsManager(self.get_temp_dir())

    metrics_mngr.update_metrics(0, _create_dummy_metrics())
    metrics_mngr.update_metrics(5, _create_dummy_metrics())
    metrics_mngr.update_metrics(10, _create_dummy_metrics())
    _, metrics = metrics_mngr.get_metrics()
    self.assertLen(metrics, 3,
                   'There should be 3 rows (for rounds 0, 5, and 10).')

    metrics_mngr.clear_rounds_after(last_valid_round_num=7)

    _, metrics = metrics_mngr.get_metrics()
    self.assertLen(
        metrics, 2,
        'After clearing all rounds after last_valid_round_num=7, should be 2 '
        'rows of metrics (for rounds 0 and 5).')
    self.assertEqual(5, metrics[-1]['round_num'],
                     'Last metrics retained are for round 5.')

    # The internal state of the manager knows the last round number is 7, so it
    # raises an exception if a user attempts to add new metrics at round 7, ...
    with self.assertRaises(ValueError):
      metrics_mngr.update_metrics(7, _create_dummy_metrics())

    # ... but allows a user to add new metrics at a round number greater than 7.
    metrics_mngr.update_metrics(8, _create_dummy_metrics())  # (No exception.)

  def test_rows_are_cleared_is_reflected_in_saved_file(self):
    temp_dir = self.get_temp_dir()
    metrics_mngr = metrics_manager.ScalarMetricsManager(temp_dir, prefix='foo')

    metrics_mngr.update_metrics(0, _create_dummy_metrics())
    metrics_mngr.update_metrics(5, _create_dummy_metrics())
    metrics_mngr.update_metrics(10, _create_dummy_metrics())

    filename = os.path.join(temp_dir, 'foo.metrics.csv')
    with tf.io.gfile.GFile(filename, 'r') as csvfile:
      num_lines_before = len(csvfile.readlines())

    # The CSV file should have 4 lines, one for the fieldnames, and 3 for each
    # call to `update_metrics`.
    self.assertEqual(num_lines_before, 4)

    metrics_mngr.clear_rounds_after(last_valid_round_num=7)

    with tf.io.gfile.GFile(filename, 'r') as csvfile:
      num_lines_after = len(csvfile.readlines())

    # The CSV file should have 3 lines, one for the fieldnames, and 2 for the
    # calls to `update_metrics` with round_nums less <= 7.
    self.assertEqual(num_lines_after, 3)

  def test_constructor_raises_value_error_if_csvfile_is_invalid(self):
    metrics_missing_round_num = _create_dummy_metrics()
    temp_dir = self.get_temp_dir()
    # This csvfile is 'invalid' in that it was not originally created by an
    # instance of ScalarMetricsManager, and is missing a column for
    # round_num.
    invalid_csvfile = os.path.join(temp_dir, 'foo.metrics.csv')
    with tf.io.gfile.GFile(invalid_csvfile, 'w') as csvfile:
      writer = csv.DictWriter(
          csvfile, fieldnames=metrics_missing_round_num.keys())
      writer.writeheader()
      writer.writerow(metrics_missing_round_num)

    with self.assertRaises(ValueError):
      metrics_manager.ScalarMetricsManager(temp_dir, prefix='foo')


if __name__ == '__main__':
  tf.test.main()
