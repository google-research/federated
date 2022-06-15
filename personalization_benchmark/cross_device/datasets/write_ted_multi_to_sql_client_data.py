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
"""Creates a federated version of the TedMultiTranslate dataset."""

import collections
from collections.abc import Sequence
import os
from typing import List, OrderedDict

from absl import app
from absl import flags
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_federated as tff

_TED_MULTI_DATA_DIRECTORY = flags.DEFINE_string(
    'ted_multi_data_directory',
    default=None,
    help='Which directory to write the processed dataset. The train/validation/'
    'test splits will be saved under `{ted_multi_data_directory}/{split}`.')

flags.register_validator(
    flag_name='ted_multi_data_directory',
    checker=lambda name: isinstance(name, str) and name,
    message='--ted_multi_data_directory must be a non-empty string.')


def _convert_list_ordered_dicts_to_tf_data(
    list_ordered_dicts: List[OrderedDict[str, tf.Tensor]]) -> tf.data.Dataset:
  dict_of_lists = collections.OrderedDict()
  for key in list_ordered_dicts[0]:
    dict_of_lists[key] = [d[key] for d in list_ordered_dicts]
  return tf.data.Dataset.from_tensor_slices(dict_of_lists)


def save_to_sql_client_data(split: str, database_filepath: str) -> None:
  """Processes and saves a split of ted_multi_translate to disk."""
  ted_multi_ds = tfds.load('ted_multi_translate')[split]
  client_dataset_by_id = collections.defaultdict(list)
  for example in ted_multi_ds:
    talk_name = example['talk_name']
    languages = example['translations']['language']
    translations = example['translations']['translation']
    for (language, translation) in zip(languages, translations):
      # Need to convert bytes to str via `decode('utf-8')`; otherwise, it won't
      # work with `tff.simulation.datasets.save_to_sql_client_data`.
      client_id = tf.strings.join([language, talk_name],
                                  separator='-').numpy().decode('utf-8')
      # Features are intentionally sorted lexicographically by key for
      # consistency across datasets.
      example = collections.OrderedDict(
          sorted([
              ('language', language),
              ('talk_name', talk_name),
              ('translation', translation),
          ]))
      client_dataset_by_id[client_id].append(example)

  def dataset_fn(client_id: str) -> tf.data.Dataset:
    return _convert_list_ordered_dicts_to_tf_data(
        client_dataset_by_id[client_id])

  tff.simulation.datasets.save_to_sql_client_data(
      client_ids=list(client_dataset_by_id.keys()),
      dataset_fn=dataset_fn,
      database_filepath=database_filepath)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  for split in ['train', 'validation', 'test']:
    database_filepath = os.path.join(_TED_MULTI_DATA_DIRECTORY.value, split)
    save_to_sql_client_data(split, database_filepath)


if __name__ == '__main__':
  app.run(main)
