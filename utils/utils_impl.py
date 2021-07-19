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
"""Utilities supporting experiments."""

import collections
import contextlib
import functools
import itertools
import multiprocessing
import os.path
import shutil
import subprocess
import tempfile
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Union

from absl import flags
import numpy as np
import pandas as pd
import tensorflow as tf


def iter_grid(
    grid_dict: Mapping[str, Sequence[Union[int, float, str]]]
) -> Iterator[Dict[str, Union[int, float, str]]]:
  """Iterates over all combinations of values in the provied dict-of-lists.

  >>> list(iter_grid({'a': [1, 2], 'b': [4.0, 5.0, 6.0]))
  [OrderedDict([('a', 1), ('b', 4.0)]),
   OrderedDict([('a', 1), ('b', 5.0)]),
   OrderedDict([('a', 1), ('b', 6.0)]),
   OrderedDict([('a', 2), ('b', 4.0)]),
   OrderedDict([('a', 2), ('b', 5.0)]),
   OrderedDict([('a', 2), ('b', 6.0)])]

  Args:
    grid_dict: A dictionary of iterables.

  Yields:
    A sequence of dictionaries with keys from grid, and values corresponding
    to all combinations of items in the corresponding iterables.
  """
  names_to_lists = collections.OrderedDict(sorted(grid_dict.items()))
  names = names_to_lists.keys()
  for values in itertools.product(*names_to_lists.values()):
    yield collections.OrderedDict(zip(names, values))


def atomic_write_to_csv(dataframe: pd.DataFrame,
                        output_file: str,
                        overwrite: bool = True) -> None:
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
        .format(type(dataframe)))
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
      os.path.dirname(output_file), '{}.tmp{}'.format(
          os.path.basename(output_file),
          np.random.randint(0, 2**63, dtype=np.int64)))
  tf.io.gfile.copy(src=tmp_name, dst=tmp_gfile_name, overwrite=overwrite)

  # Finally, do an atomic rename and clean up:
  tf.io.gfile.rename(tmp_gfile_name, output_file, overwrite=overwrite)
  shutil.rmtree(tmp_dir)


def atomic_write_series_to_csv(series_data: Any,
                               output_file: str,
                               overwrite: bool = True) -> None:
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


def atomic_read_from_csv(csv_file):
  """Reads a `pandas.DataFrame` from the (possibly zipped) `csv_file`.

  Format note: The CSV is expected to have an index column.

  Args:
    csv_file: A (possibly zipped) CSV file.

  Returns:
    A `pandas.Dataframe`.
  """
  if csv_file.endswith('.bz2'):
    file_open_mode = 'rb'
    compression = 'bz2'
  else:
    file_open_mode = 'r'
    compression = None
  return pd.read_csv(
      tf.io.gfile.GFile(csv_file, mode=file_open_mode),
      compression=compression,
      engine='c',
      index_col=0)


_all_hparam_flags = []


@contextlib.contextmanager
def record_hparam_flags():
  """A context manager that adds all flags created in its scope to a global list of flags, and yields all flags created in its scope.

  This is useful for defining hyperparameter flags of an experiment, especially
  when the flags are partitioned across a number of modules. The total list of
  flags defined across modules can then be accessed via get_hparam_flags().

  Example usage:
  ```python
  with record_hparam_flags() as optimizer_hparam_flags:
      flags.DEFINE_string('optimizer', 'sgd', 'Optimizer for training.')
  with record_hparam_flags() as evaluation_hparam_flags:
      flags.DEFINE_string('eval_metric', 'accuracy', 'Metric for evaluation.')
  experiment_hparam_flags = get_hparam_flags().
  ```

  Check `research/optimization/emnist/run_emnist.py` for more usage details.

  Yields:
    A list of all newly created flags.
  """
  old_flags = set(iter(flags.FLAGS))
  new_flags = []
  yield new_flags
  new_flags.extend([f for f in flags.FLAGS if f not in old_flags])
  _all_hparam_flags.extend(new_flags)


def get_hparam_flags():
  """Returns a list of flags defined within the scope of record_hparam_flags."""
  return _all_hparam_flags


@contextlib.contextmanager
def record_new_flags() -> Iterator[List[str]]:
  """A context manager that returns all flags created in its scope.

  This is useful to define all of the flags which should be considered
  hyperparameters of the training run, without needing to repeat them.

  Example usage:
  ```python
  with record_new_flags() as hparam_flags:
      flags.DEFINE_string('exp_name', 'name', 'Unique name for the experiment.')
  ```

  Check `research/emnist/run_experiment.py` for more details about the usage.

  Yields:
    A list of all newly created flags.
  """
  old_flags = set(iter(flags.FLAGS))
  new_flags = []
  yield new_flags
  new_flags.extend([f for f in flags.FLAGS if f not in old_flags])


def lookup_flag_values(flag_list: Iterable[str]) -> collections.OrderedDict:
  """Returns a dictionary of (flag_name, flag_value) pairs for an iterable of flag names."""
  flag_odict = collections.OrderedDict()
  for flag_name in flag_list:
    if not isinstance(flag_name, str):
      raise ValueError(
          'All flag names must be strings. Flag {} was of type {}.'.format(
              flag_name, type(flag_name)))

    if flag_name not in flags.FLAGS:
      raise ValueError('"{}" is not a defined flag.'.format(flag_name))
    flag_odict[flag_name] = flags.FLAGS[flag_name].value

  return flag_odict


def hparams_to_str(wid: int,
                   param_dict: Mapping[str, str],
                   short_names: Optional[Mapping[str, str]] = None) -> str:
  """Convenience method which flattens the hparams to a string.

  Used as mapping function for the WorkUnitCustomiser.

  Args:
    wid: Work unit id, int type.
    param_dict: A dict of parameters.
    short_names: A dict of mappings of parameter names.

  Returns:
    The hparam string.
  """
  if not param_dict:
    return str(wid)

  if not short_names:
    short_names = {}

  name = [
      '{}={}'.format(short_names.get(k, k), str(v))
      for k, v in sorted(param_dict.items())
  ]
  hparams_str = '{}-{}'.format(str(wid), ','.join(name))

  # Escape some special characters
  replace_str = {
      '\n': ',',
      ':': '=',
      '\'': '',
      '"': '',
  }
  for c, new_c in replace_str.items():
    hparams_str = hparams_str.replace(c, new_c)
  for c in ('\\', '/', '[', ']', '(', ')', '{', '}', '%'):
    hparams_str = hparams_str.replace(c, '-')
  if len(hparams_str) > 170:
    raise ValueError(
        'hparams_str string is too long ({}). You can input a short_name dict '
        'to map the long parameter name to a short name. For example, '
        ' launch_experiment(executable, grid_iter, '
        ' {{server_learning_rate: s_lr}}) \n'
        'Received: {}'.format(len(hparams_str), hparams_str))
  return hparams_str


def launch_experiment(executable: str,
                      grid_iter: Iterable[Mapping[str, Union[int, float, str]]],
                      root_output_dir: str = '/tmp/exp',
                      short_names: Optional[Mapping[str, str]] = None,
                      max_workers: int = 1):
  """Launch experiments of grid search in parallel or sequentially.

  Example usage:
  ```python
  grid_iter = iter_grid({'a': [1, 2], 'b': [4.0, 5.0]))
  launch_experiment('run_exp.py', grid_iter)
  ```

  Args:
    executable: An executable which takes flags --root_output_dir
      and --exp_name, e.g., `bazel run //research/emnist:run_experiment --`.
    grid_iter: A sequence of dictionaries with keys from grid, and values
      corresponding to all combinations of items in the corresponding iterables.
    root_output_dir: The directory where all outputs are stored.
    short_names: Short name mapping for the parameter name used if parameter
      string length is too long.
    max_workers: The max number of commands to run in parallel.
  """
  command_list = []
  for idx, param_dict in enumerate(grid_iter):
    param_list = [
        '--{}={}'.format(key, str(value))
        for key, value in sorted(param_dict.items())
    ]

    short_names = short_names or {}
    param_str = hparams_to_str(idx, param_dict, short_names)

    param_list.append('--root_output_dir={}'.format(root_output_dir))
    param_list.append('--exp_name={}'.format(param_str))
    command = '{} {}'.format(executable, ' '.join(param_list))
    command_list.append(command)

  pool = multiprocessing.Pool(processes=max_workers)
  executor = functools.partial(subprocess.call, shell=True)
  for command in command_list:
    pool.apply_async(executor, (command,))
  pool.close()
  pool.join()
