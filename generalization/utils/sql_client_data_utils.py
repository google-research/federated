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
"""Utilities for constructing and parsing sqlite3-backed ClientData."""

import collections
import os
import tempfile
from typing import Callable, List, Mapping

from absl import logging

import pandas as pd
import sqlite3
import tensorflow as tf
import tensorflow_federated as tff

from generalization.utils import logging_utils


# The following three feature builders are borrowed from
# https://www.tensorflow.org/tutorials/load_data/tfrecord
def _bytes_feature(tensor) -> tf.train.Feature:
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tensor.numpy()]))


def _float_feature(tensor) -> tf.train.Feature:
  """Returns a float_list from a float / double."""
  return tf.train.Feature(
      float_list=tf.train.FloatList(value=tensor.numpy().flatten()))


def _int64_feature(tensor) -> tf.train.Feature:
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(
      int64_list=tf.train.Int64List(value=tensor.numpy().flatten()))


def build_serializer(
    element_spec: Mapping[str, tf.TensorSpec]
) -> Callable[[Mapping[str, tf.Tensor]], bytes]:
  """Build a serializer based on the element_spec of a dataset."""
  feature_fn = {}
  for key, tensor_spec in element_spec.items():
    if tensor_spec.dtype is tf.string:
      feature_fn[key] = _bytes_feature
    elif tensor_spec.dtype.is_floating:
      feature_fn[key] = _float_feature
    elif tensor_spec.dtype.is_integer:
      feature_fn[key] = _int64_feature
    else:
      raise ValueError(f'unsupported dtype {tensor_spec.dtype}')

  def serializer(element: Mapping[str, tf.Tensor]) -> bytes:

    feature = {key: feature_fn[key](tensor) for key, tensor in element.items()}

    return tf.train.Example(features=tf.train.Features(
        feature=feature)).SerializeToString()

  return serializer


def build_parser(
    element_spec: Mapping[str, tf.TensorSpec]
) -> Callable[[bytes], Mapping[str, tf.Tensor]]:
  """Build a parser based on the element_spec of a dataset."""
  parse_spec = {}
  for key, tensor_spec in element_spec.items():
    if tensor_spec.dtype is tf.string:
      parser_dtype = tf.string
    elif tensor_spec.dtype.is_floating:
      parser_dtype = tf.float32
    elif tensor_spec.dtype.is_integer:
      parser_dtype = tf.int64
    else:
      raise ValueError(f'unsupported dtype {tensor_spec.dtype}')

    parse_spec[key] = tf.io.FixedLenFeature(
        shape=tensor_spec.shape, dtype=parser_dtype)

  def parser(tensor_proto: bytes) -> Mapping[str, tf.Tensor]:
    parsed_features = tf.io.parse_example(tensor_proto, parse_spec)

    result = collections.OrderedDict()

    for key, tensor_spec in element_spec.items():
      result[key] = tf.cast(parsed_features[key], tensor_spec.dtype)

    return result

  return parser


def save_to_sql_client_data(
    client_ids: List[str],
    dataset_fn: Callable[[str], tf.data.Dataset],
    database_filepath: str,
    allow_overwrite: bool = False,
) -> None:
  """Serialize a federated dataset into a SQL database compatible with `SqlClientData`.

  Requirement: All the clients must share the same dataset.element_spec.
    Otherwise TypeError will be raised.

  Limitations: At this time the shared element_spec must be of type
    `Mapping[str, TensorSpec]`. Otherwise `TypeError` will be raised.

  Args:
    client_ids: A list of string identifiers for clients in this dataset.
    dataset_fn: A callable that accepts a `str` as an argument and returns the
      `tf.data.Dataset` instance.
    database_filepath: A `str` filepath for the SQL database.
    allow_overwrite: Whether to allow overwriting if file already exists at
      dataset_filepath.

  Raises:
    FileExistsError: if file exists at `dataset_filepath` and `allow_overwrite`
      is False.
    TypeError: if the element_spec of local datasets are not identical across
    clients, or if the element_spec of datasets are not of type
      `Mapping[str, TensorSpec]`.

  """

  if tf.io.gfile.exists(database_filepath) and not allow_overwrite:
    raise FileExistsError(f'File already exists at {database_filepath}')

  tmp_database_filepath = tempfile.mkstemp()[1]
  logging.info('Building local SQL database at %s.', tmp_database_filepath)
  example_client_id = client_ids[0]
  example_dataset = dataset_fn(example_client_id)
  example_element_spec = example_dataset.element_spec

  if not isinstance(example_element_spec, Mapping):
    raise TypeError('The element_spec of the local dataset must be a Mapping, '
                    f'found {example_element_spec} instead')
  for key, val in example_element_spec.items():
    if not isinstance(val, tf.TensorSpec):
      raise TypeError(
          'The element_spec of the local dataset must be a Mapping[str, TensorSpec], '
          f'and must not be nested, found {key}:{val} instead.')

  serializer = build_serializer(example_element_spec)
  parser = build_parser(example_element_spec)

  with sqlite3.connect(tmp_database_filepath) as con:
    test_setup_queries = [
        """CREATE TABLE examples (
           split_name TEXT NOT NULL,
           client_id TEXT NOT NULL,
           serialized_example_proto BLOB NOT NULL);""",
        # The `client_metadata` table is required, though not documented.
        """CREATE TABLE client_metadata (
           client_id TEXT NOT NULL,
           split_name TEXT NOT NULL,
           num_examples INTEGER NOT NULL);""",
    ]
    for q in test_setup_queries:
      con.execute(q)

    logging.info('Starting writing to SQL database at scratch path {%s}.',
                 tmp_database_filepath)

    logger = logging_utils.ProgressLogger(
        name='writing SQL Database',
        every=(len(client_ids) + 9) // 10,
        total=len(client_ids),
    )
    for client_id in client_ids:
      local_ds = dataset_fn(client_id)

      if local_ds.element_spec != example_element_spec:
        raise TypeError(f"""
            All the clients must share the same dataset element type.
            The local dataset of client '{client_id}' has element type
            {local_ds.element_spec}, which is different from client
            '{example_client_id}' which has element type {example_element_spec}.
            """)

      num_elem = 0
      for elem in local_ds:
        num_elem += 1
        con.execute(
            'INSERT INTO examples '
            '(split_name, client_id, serialized_example_proto) '
            'VALUES (?, ?, ?);', ('N/A', client_id, serializer(elem)))

      con.execute(
          'INSERT INTO client_metadata (client_id, split_name, num_examples) '
          'VALUES (?, ?, ?);', (client_id, 'N/A', num_elem))

      logger.increment(1)
    del logger

  if tf.io.gfile.exists(database_filepath):
    tf.io.gfile.remove(database_filepath)
  tf.io.gfile.makedirs(os.path.dirname(database_filepath))
  tf.io.gfile.copy(tmp_database_filepath, database_filepath)
  tf.io.gfile.remove(tmp_database_filepath)
  logging.info('SQL database saved at %s', database_filepath)


def save_to_sql_client_data_from_mapping(
    cid_to_ds_mapping: Mapping[str, tf.data.Dataset],
    database_filepath: str,
    allow_overwrite: bool = False,
) -> None:
  """Serialize a federated dataset into a `SqlClientData` from a mapping."""
  client_ids = list(cid_to_ds_mapping.keys())
  dataset_fn = cid_to_ds_mapping.get
  save_to_sql_client_data(client_ids, dataset_fn, database_filepath,
                          allow_overwrite)


def save_to_sql_client_data_from_client_data(
    cd: tff.simulation.datasets.ClientData,
    database_filepath: str,
    allow_overwrite: bool = False,
) -> None:
  """Serialize a federated dataset into a `SqlClientData` from an existing ClientData."""
  client_ids = cd.client_ids
  dataset_fn = cd.create_tf_dataset_for_client
  save_to_sql_client_data(client_ids, dataset_fn, database_filepath,
                          allow_overwrite)


def load_parsed_sql_client_data(
    database_filepath: str, element_spec: Mapping[str, tf.TensorSpec]
) -> tff.simulation.datasets.ClientData:
  """Load a SqlClientData from file and parse with the given element_spec.

  Args:
    database_filepath: A `str` filepath of the SQL database. This function will
      first fetch the SQL database to a local temporary directory if
      `database_filepath` is a remote directory.
    element_spec: The `element_spec` of the local dataset. This is used to parse
      the serialized SqlClientData.

  Returns:
    A parsed ClientData instance backed by SqlClientData.

  Raises:
    FileNotFoundError: if database_filepath does not exist.
  """
  parser = build_parser(element_spec)

  def dataset_parser(ds: tf.data.Dataset) -> tf.data.Dataset:
    return ds.map(parser, num_parallel_calls=tf.data.AUTOTUNE)

  if not tf.io.gfile.exists(database_filepath):
    raise FileNotFoundError(f'No such file or directory: {database_filepath}')
  elif not os.path.exists(database_filepath):
    logging.info('Starting fetching SQL database to local.')
    tmp_dir = tempfile.mkdtemp()
    tmp_database_filepath = tf.io.gfile.join(
        tmp_dir, os.path.basename(database_filepath))
    tf.io.gfile.copy(database_filepath, tmp_database_filepath, overwrite=True)
    database_filepath = tmp_database_filepath
    logging.info('Finished fetching SQL database to local.')

  return tff.simulation.datasets.SqlClientData(database_filepath).preprocess(
      dataset_parser)


def load_sql_client_data_metadata(database_filepath: str) -> pd.DataFrame:
  """Load the metadata from a SqlClientData database.

  Args:
    database_filepath: A `str` filepath of the SQL database.
  This function will first fetch the SQL database to a local temporary directory
    if `database_filepath` is a remote directory.

  Returns:
    A pandas DataFrame containing the metadata.

  Raises:
    FileNotFoundError: if database_filepath does not exist.
  """

  if not tf.io.gfile.exists(database_filepath):
    raise FileNotFoundError(f'No such file or directory: {database_filepath}')
  elif not os.path.exists(database_filepath):
    logging.info('Starting fetching SQL database to local.')
    tmp_dir = tempfile.mkdtemp()
    tmp_database_filepath = tf.io.gfile.join(
        tmp_dir, os.path.basename(database_filepath))
    tf.io.gfile.copy(database_filepath, tmp_database_filepath, overwrite=True)
    database_filepath = tmp_database_filepath
    logging.info('Finished fetching SQL database to local.')

  con = sqlite3.connect(database_filepath)
  return pd.read_sql_query('SELECT * from client_metadata', con)
