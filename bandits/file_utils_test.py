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
"""Tests for file_utils."""
import os

from absl.testing import absltest
from absl.testing import parameterized
import pandas as pd
import tensorflow as tf

from bandits import file_utils


class UtilsTest(tf.test.TestCase, parameterized.TestCase):

  def test_atomic_write_raises_on_pandas_series_input(self):
    output_file = os.path.join(absltest.get_default_test_tmpdir(), 'foo.csv')
    with self.assertRaisesRegex(
        ValueError, 'dataframe must be an instance of `pandas.DataFrame`'
    ):
      file_utils.atomic_write_to_csv(pd.Series(dict(a=1)), output_file)

  def test_atomic_write_raises_on_dict_input(self):
    output_file = os.path.join(absltest.get_default_test_tmpdir(), 'foo.csv')
    with self.assertRaisesRegex(
        ValueError, 'dataframe must be an instance of `pandas.DataFrame`'
    ):
      file_utils.atomic_write_to_csv(dict(a=1), output_file)

  @parameterized.named_parameters(
      ('unzipped', 'foo.csv'), ('zipped', 'baz.csv.bz2')
  )
  def test_atomic_write(self, name):
    dataframe = pd.DataFrame(dict(a=[1, 2], b=[4.0, 5.0]))
    output_file = os.path.join(absltest.get_default_test_tmpdir(), name)
    file_utils.atomic_write_to_csv(dataframe, output_file)
    dataframe2 = pd.read_csv(output_file, index_col=0)
    pd.testing.assert_frame_equal(dataframe, dataframe2)

    # Overwriting
    dataframe3 = pd.DataFrame(dict(a=[1, 2, 3], b=[4.0, 5.0, 6.0]))
    file_utils.atomic_write_to_csv(dataframe3, output_file)
    dataframe4 = pd.read_csv(output_file, index_col=0)
    pd.testing.assert_frame_equal(dataframe3, dataframe4)

  @parameterized.named_parameters(
      ('unzipped', 'foo.csv'), ('zipped', 'baz.csv.bz2')
  )
  def test_atomic_write_series_with_scalar_data(self, name):
    series_data = dict(a=1, b=4.0)
    output_file = os.path.join(absltest.get_default_test_tmpdir(), name)
    file_utils.atomic_write_series_to_csv(series_data, output_file)
    dataframe = pd.read_csv(output_file, index_col=0)
    pd.testing.assert_frame_equal(
        pd.DataFrame(pd.Series(series_data), columns=['0']), dataframe
    )

  @parameterized.named_parameters(
      ('unzipped', 'foo.csv'), ('zipped', 'baz.csv.bz2')
  )
  def test_atomic_write_series_with_non_scalar_data(self, name):
    series_data = dict(a=[1, 2], b=[3.0, 4.0])
    output_file = os.path.join(absltest.get_default_test_tmpdir(), name)
    file_utils.atomic_write_series_to_csv(series_data, output_file)
    dataframe = pd.read_csv(output_file, index_col=0)

    series_data_as_string = dict(a='[1, 2]', b='[3.0, 4.0]')
    expected_df = pd.DataFrame(pd.Series(series_data_as_string), columns=['0'])
    pd.testing.assert_frame_equal(expected_df, dataframe)


if __name__ == '__main__':
  tf.test.main()
