# Copyright 2020, Google LLC.
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
"""Pre-processing utils for StackOverflow data for reconstruction experiments."""

import collections
from typing import List

import tensorflow as tf
import tensorflow_federated as tff

from utils.datasets import stackoverflow_word_prediction


def _creation_date_string_to_integer(dates: tf.Tensor) -> tf.Tensor:
  """Converts ISO date string to integer that can be sorted.

  Returned integers retain the property that sorting the integers results in
  dates sorted in chronological order, so these integers can be used to sort
  examples by date. Ignores fractional seconds if provided.
  Assumes standard time offset.

  For example:
    2009-06-15T13:45:30 -> 20090615134530
    2009-06-15T13:45:30.345Z -> 20090615134530

  Args:
    dates: A tf.string tensor of dates in simplified ISO 8601 format. The data
      produced by `tff.simulation.datasets.stackoverflow.load_data` conforms to
      this format.

  Returns:
    A tf.int64 tensor of integers representing dates.
  """
  year = tf.strings.to_number(tf.strings.substr(dates, 0, 4), out_type=tf.int64)
  month = tf.strings.to_number(
      tf.strings.substr(dates, 5, 2), out_type=tf.int64)
  day = tf.strings.to_number(tf.strings.substr(dates, 8, 2), out_type=tf.int64)
  hour = tf.strings.to_number(
      tf.strings.substr(dates, 11, 2), out_type=tf.int64)
  minute = tf.strings.to_number(
      tf.strings.substr(dates, 14, 2), out_type=tf.int64)
  second = tf.strings.to_number(
      tf.strings.substr(dates, 17, 2), out_type=tf.int64)

  timestamp = 0
  timestamp = (timestamp + year) * 100
  timestamp = (timestamp + month) * 100
  timestamp = (timestamp + day) * 100
  timestamp = (timestamp + hour) * 100
  timestamp = (timestamp + minute) * 100
  timestamp = timestamp + second
  return timestamp


def _sort_examples_by_date(
    examples: collections.OrderedDict) -> collections.OrderedDict:
  """Sorts a batch of dataset elements by increasing creation date.

  Sorting is stable, so original ordering is consistently retained for ties.

  Args:
    examples: A batch of examples.

  Returns:
    Output batch, sorted by creation date.
  """
  date_integers = _creation_date_string_to_integer(examples['creation_date'])
  sorted_indices = tf.argsort(date_integers, stable=True)
  new_examples = collections.OrderedDict()
  for key in examples:
    new_examples[key] = tf.gather(examples[key], sorted_indices)
  return new_examples


def create_preprocess_fn(vocab: List[str],
                         num_oov_buckets: int,
                         client_batch_size: int,
                         max_sequence_length: int,
                         max_elements_per_client: int,
                         feature_dtypes: collections.OrderedDict,
                         sort_by_date: bool = True) -> tff.Computation:
  """Creates a preprocessing functions for Stack Overflow next-word-prediction.

  This function returns a `tff.Computation` which takes a dataset and returns a
  dataset, suitable for mapping over a set of unprocessed client datasets.

  Args:
    vocab: Vocabulary which defines the embedding.
    num_oov_buckets: The number of out of vocabulary buckets. Tokens that are
      not present in the `vocab` are hashed into one of these buckets.
    client_batch_size: Integer representing batch size to use on the clients.
    max_sequence_length: Integer determining shape of padded batches. Sequences
      will be padded up to this length, and sentences longer than this will be
      truncated to this length.
    max_elements_per_client: Integer controlling the maximum number of elements
      to take per client. This is intended primarily to contend with the small
      set of clients with tens of thousands of examples.
    feature_dtypes: An OrderedDict mapping from string key names to dtypes for
      each feature.
    sort_by_date: If True, sort elements by increasing "creation_date". If
      False, shuffle elements instead. Sorting or shuffling is applied after
      limiting to `max_elements_per_client`.

  Returns:
    A `tff.Computation` taking as input a `tf.data.Dataset`, and returning a
    `tf.data.Dataset` formed by preprocessing according to the input arguments.
  """
  if client_batch_size <= 0:
    raise ValueError('client_batch_size must be a positive integer. You have '
                     'passed {}.'.format(client_batch_size))
  elif max_sequence_length <= 0:
    raise ValueError('max_sequence_length must be a positive integer. You have '
                     'passed {}.'.format(max_sequence_length))
  elif max_elements_per_client <= 0:
    raise ValueError(
        'max_elements_per_client must be a positive integer. You have '
        'passed {}.'.format(max_elements_per_client))
  if num_oov_buckets <= 0:
    raise ValueError('num_oov_buckets must be a positive integer. You have '
                     'passed {}.'.format(num_oov_buckets))

  @tff.tf_computation(tff.SequenceType(feature_dtypes))
  def preprocess_fn(dataset):
    to_ids = stackoverflow_word_prediction.build_to_ids_fn(
        vocab=vocab,
        max_sequence_length=max_sequence_length,
        num_oov_buckets=num_oov_buckets)
    dataset = dataset.take(max_elements_per_client)
    if sort_by_date:
      dataset = dataset.batch(max_elements_per_client).map(
          _sort_examples_by_date).unbatch()
    else:
      dataset = dataset.shuffle(max_elements_per_client)
    dataset = dataset.map(
        to_ids, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return stackoverflow_word_prediction.batch_and_split(
        dataset, max_sequence_length, client_batch_size)

  return preprocess_fn
