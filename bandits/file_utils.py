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
"""File utils."""
import os.path
import shutil
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf


def atomic_write_to_csv(
    dataframe: pd.DataFrame, output_file: str, overwrite: bool = True
) -> None:
  """Writes `dataframe` to `output_file` as a (possibly zipped) CSV file.

  Args:
    dataframe: A `pandas.Dataframe`.
    output_file: The final output file to write. The output will be compressed
      depending on the filename, see documentation for
      `pandas.DateFrame.to_csv(compression='infer')`.
    overwrite: Whether to overwrite `output_file` if it exists.

  Raises:
    ValueError: If `dataframe` is not an instance of `pandas.DataFrame`.
  """
  if not isinstance(dataframe, pd.DataFrame):
    raise ValueError(
        'dataframe must be an instance of `pandas.DataFrame`, received a `{}`'
        .format(type(dataframe))
    )
  # Exporting via to_hdf() is an appealing option, because we could perhaps
  # maintain more type information, and also write both hyperparameters and
  # results to the same HDF5 file. However, to_hdf() call uses pickle under the
  # hood, and there seems to be no way to tell it to use pickle protocol=2, it
  # defaults to 4. This means the results cannot be read from Python 2. We
  # currently still want Python 2 support, so sticking with CSVs for now.

  # At least when writing a zip, .to_csv() is not happy taking a gfile,
  # so we need a temp file on the local filesystem.
  tmp_dir = tempfile.mkdtemp(prefix='atomic_write_to_csv_tmp')
  # We put the output_file name last so we preserve the extension to allow
  # inference of the desired compression format. Note that files with .zip
  # extension (but not .bz2, .gzip, or .xv) have unexpected internal filenames
  # due to https://github.com/pandas-dev/pandas/issues/26023, not
  # because of something we are doing here.
  tmp_name = os.path.join(tmp_dir, os.path.basename(output_file))
  assert not tf.io.gfile.exists(tmp_name), 'file [{!s}] exists'.format(tmp_name)
  dataframe.to_csv(tmp_name, header=True)

  # Now, copy to a temp gfile next to the final target, allowing for
  # an atomic move.
  tmp_gfile_name = os.path.join(
      os.path.dirname(output_file),
      '{}.tmp{}'.format(
          os.path.basename(output_file), np.random.randint(0, 2**31)
      ),
  )
  tf.io.gfile.copy(src=tmp_name, dst=tmp_gfile_name, overwrite=overwrite)

  # Finally, do an atomic rename and clean up:
  tf.io.gfile.rename(tmp_gfile_name, output_file, overwrite=overwrite)
  shutil.rmtree(tmp_dir)


def atomic_write_series_to_csv(
    series_data: Any, output_file: str, overwrite: bool = True
) -> None:
  """Writes series data to `output_file` as a (possibly zipped) CSV file.

  The series data will be written to a CSV with two columns, an unlabeled
  column with the indices of `series_data` (the keys if it is a `dict`), and a
  column with label `0` containing the associated values in `series_data`. Note
  that if `series_data` has non-scalar values, these will be written via their
  string representation.

  Args:
    series_data: A structure that can be converted to a `pandas.Series`,
      typically an array-like, iterable, dictionary, or scalar value. For more
      details, see documentation for `pandas.Series`.
    output_file: The final output file to write. The output will be compressed
      depending on the filename, see documentation for
      `pandas.DateFrame.to_csv(compression='infer')`.
    overwrite: Whether to overwrite `output_file` if it exists.
  """
  dataframe = pd.DataFrame(pd.Series(series_data))
  atomic_write_to_csv(dataframe, output_file, overwrite)
