# Copyright 2018, Google LLC.
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
import contextlib
import os
from unittest import mock

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import pandas as pd
import tensorflow as tf

from utils import utils_impl

FLAGS = flags.FLAGS


@contextlib.contextmanager
def flag_sandbox(flag_value_dict):

  def _set_flags(flag_dict):
    for name, value in flag_dict.items():
      FLAGS[name].value = value

  # Store the current values and override with the new.
  preserved_value_dict = {
      name: FLAGS[name].value for name in flag_value_dict.keys()
  }
  _set_flags(flag_value_dict)
  yield

  # Restore the saved values.
  for name in preserved_value_dict.keys():
    FLAGS[name].unparse()
  _set_flags(preserved_value_dict)


class UtilsTest(tf.test.TestCase, parameterized.TestCase):

  def test_atomic_write_raises_on_pandas_series_input(self):
    output_file = os.path.join(absltest.get_default_test_tmpdir(), 'foo.csv')
    with self.assertRaisesRegex(
        ValueError, 'dataframe must be an instance of `pandas.DataFrame`'):
      utils_impl.atomic_write_to_csv(pd.Series(dict(a=1)), output_file)

  def test_atomic_write_raises_on_dict_input(self):
    output_file = os.path.join(absltest.get_default_test_tmpdir(), 'foo.csv')
    with self.assertRaisesRegex(
        ValueError, 'dataframe must be an instance of `pandas.DataFrame`'):
      utils_impl.atomic_write_to_csv(dict(a=1), output_file)

  @parameterized.named_parameters(('unzipped', 'foo.csv'),
                                  ('zipped', 'baz.csv.bz2'))
  def test_atomic_write(self, name):
    dataframe = pd.DataFrame(dict(a=[1, 2], b=[4.0, 5.0]))
    output_file = os.path.join(absltest.get_default_test_tmpdir(), name)
    utils_impl.atomic_write_to_csv(dataframe, output_file)
    dataframe2 = pd.read_csv(output_file, index_col=0)
    pd.testing.assert_frame_equal(dataframe, dataframe2)

    # Overwriting
    dataframe3 = pd.DataFrame(dict(a=[1, 2, 3], b=[4.0, 5.0, 6.0]))
    utils_impl.atomic_write_to_csv(dataframe3, output_file)
    dataframe4 = pd.read_csv(output_file, index_col=0)
    pd.testing.assert_frame_equal(dataframe3, dataframe4)

  @parameterized.named_parameters(('unzipped', 'foo.csv'),
                                  ('zipped', 'baz.csv.bz2'))
  def test_atomic_write_series_with_scalar_data(self, name):
    series_data = dict(a=1, b=4.0)
    output_file = os.path.join(absltest.get_default_test_tmpdir(), name)
    utils_impl.atomic_write_series_to_csv(series_data, output_file)
    dataframe = pd.read_csv(output_file, index_col=0)
    pd.testing.assert_frame_equal(
        pd.DataFrame(pd.Series(series_data), columns=['0']), dataframe)

  @parameterized.named_parameters(('unzipped', 'foo.csv'),
                                  ('zipped', 'baz.csv.bz2'))
  def test_atomic_write_series_with_non_scalar_data(self, name):
    series_data = dict(a=[1, 2], b=[3.0, 4.0])
    output_file = os.path.join(absltest.get_default_test_tmpdir(), name)
    utils_impl.atomic_write_series_to_csv(series_data, output_file)
    dataframe = pd.read_csv(output_file, index_col=0)

    series_data_as_string = dict(a='[1, 2]', b='[3.0, 4.0]')
    expected_df = pd.DataFrame(pd.Series(series_data_as_string), columns=['0'])
    pd.testing.assert_frame_equal(expected_df, dataframe)

  @parameterized.named_parameters(('unzipped', 'foo.csv'),
                                  ('zipped', 'baz.csv.bz2'))
  def test_atomic_read(self, name):
    dataframe = pd.DataFrame(dict(a=[1, 2], b=[4.0, 5.0]))
    csv_file = os.path.join(absltest.get_default_test_tmpdir(), name)
    utils_impl.atomic_write_to_csv(dataframe, csv_file)

    dataframe2 = utils_impl.atomic_read_from_csv(csv_file)
    pd.testing.assert_frame_equal(dataframe, dataframe2)

  def test_iter_grid(self):
    grid = dict(a=[], b=[])
    self.assertCountEqual(list(utils_impl.iter_grid(grid)), [])

    grid = dict(a=[1])
    self.assertCountEqual(list(utils_impl.iter_grid(grid)), [dict(a=1)])

    grid = dict(a=[1, 2])
    self.assertCountEqual(
        list(utils_impl.iter_grid(grid)), [dict(a=1), dict(a=2)])

    grid = dict(a=[1, 2], b='b', c=[3.0, 4.0])
    self.assertCountEqual(
        list(utils_impl.iter_grid(grid)), [
            dict(a=1, b='b', c=3.0),
            dict(a=1, b='b', c=4.0),
            dict(a=2, b='b', c=3.0),
            dict(a=2, b='b', c=4.0)
        ])

  def test_record_new_flags(self):
    with utils_impl.record_new_flags() as hparam_flags:
      flags.DEFINE_string('exp_name', 'name', 'Unique name for the experiment.')
      flags.DEFINE_float('learning_rate', 0.1, 'Optimizer learning rate.')

    self.assertCountEqual(hparam_flags, ['exp_name', 'learning_rate'])

  def test_convert_flag_names_to_odict(self):
    with utils_impl.record_new_flags() as hparam_flags:
      flags.DEFINE_integer('flag1', 1, 'This is the first flag.')
      flags.DEFINE_float('flag2', 2.0, 'This is the second flag.')

    hparam_odict = utils_impl.lookup_flag_values(hparam_flags)
    expected_odict = collections.OrderedDict(flag1=1, flag2=2.0)

    self.assertEqual(hparam_odict, expected_odict)

  def test_convert_undefined_flag_names(self):
    with self.assertRaisesRegex(ValueError, '"bad_flag" is not a defined flag'):
      utils_impl.lookup_flag_values(['bad_flag'])

  def test_convert_nonstr_flag(self):
    with self.assertRaisesRegex(ValueError, 'All flag names must be strings'):
      utils_impl.lookup_flag_values([300])

  @mock.patch.object(utils_impl, 'multiprocessing')
  def test_launch_experiment(self, mock_multiprocessing):
    pool = mock_multiprocessing.Pool(processes=10)

    grid_dict = [
        collections.OrderedDict([('a_long', 1), ('b', 4.0)]),
        collections.OrderedDict([('a_long', 1), ('b', 5.0)])
    ]

    utils_impl.launch_experiment(
        'bazel run //research/emnist:run_experiment --',
        grid_dict,
        '/tmp_dir',
        short_names={'a_long': 'a'})
    expected = [
        'bazel run //research/emnist:run_experiment -- --a_long=1 --b=4.0 '
        '--root_output_dir=/tmp_dir --exp_name=0-a=1,b=4.0',
        'bazel run //research/emnist:run_experiment -- --a_long=1 --b=5.0 '
        '--root_output_dir=/tmp_dir --exp_name=1-a=1,b=5.0'
    ]
    result = pool.apply_async.call_args_list
    result = [args[0][1][0] for args in result]
    self.assertCountEqual(result, expected)


if __name__ == '__main__':
  tf.test.main()
