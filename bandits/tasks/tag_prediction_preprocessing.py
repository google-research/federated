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
"""Preprocessing library for Stack Overflow tag prediction tasks."""
from collections.abc import Callable

import tensorflow as tf
import tensorflow_federated as tff

DEFAULT_SEQUENCE_LENGTH = 20
DEFAULT_SHUFFLE_BUFFER_SIZE = 1000
DEFAULT_TAG_VOCAB_SIZE = 500
DEFAULT_WORD_VOCAB_SIZE = 10000
LABEL_DIST_SHIFT_RATIO = 0.4


def build_to_ids_fn(
    word_vocab: list[str],
    tag_vocab: list[str],
    label_distribution_shift: bool = False,
) -> Callable[[tf.Tensor], tf.Tensor]:
  """Constructs a function mapping examples to sequences of token indices."""
  word_vocab_size = len(word_vocab)
  word_table_values = tf.range(word_vocab_size, dtype=tf.int64)
  word_table = tf.lookup.StaticVocabularyTable(
      tf.lookup.KeyValueTensorInitializer(word_vocab, word_table_values),
      num_oov_buckets=1,
  )

  tag_vocab_size = len(tag_vocab)
  tag_table_values = tf.range(tag_vocab_size, dtype=tf.int64)
  tag_table = tf.lookup.StaticVocabularyTable(
      tf.lookup.KeyValueTensorInitializer(tag_vocab, tag_table_values),
      num_oov_buckets=1,
  )

  def normal_tag_to_id(tags):
    tags = tf.strings.split(tags, sep='|')
    tags = tag_table.lookup(tags)
    tags = tf.one_hot(tags, tag_vocab_size + 1)
    return tf.reduce_sum(tags, axis=0)[:tag_vocab_size]

  if label_distribution_shift:
    top_tag_size = int(tag_vocab_size * (1 - LABEL_DIST_SHIFT_RATIO))

    def tag_to_id(tags):
      tags = normal_tag_to_id(tags)
      zero_tags = tf.zeros([tag_vocab_size - top_tag_size])
      new_tags = tf.concat([tags[:top_tag_size], zero_tags], axis=0)
      return new_tags

  else:
    tag_to_id = normal_tag_to_id

  def to_ids(example):
    """Converts a Stack Overflow example to a bag-of-words/tags format."""
    sentence = tf.strings.join(
        [example['tokens'], example['title']], separator=' '
    )
    words = tf.strings.split(sentence)
    tokens = word_table.lookup(words)
    token_sums = tf.reduce_sum(tf.one_hot(tokens, word_vocab_size), axis=0)
    num_tokens = tf.reduce_sum(token_sums)
    tokens = tf.math.divide_no_nan(token_sums, num_tokens)

    tags = tag_to_id(example['tags'])
    return (tokens, tags)

  return to_ids


def create_preprocess_fn(
    preprocess_spec: tff.simulation.baselines.ClientSpec,
    word_vocab: list[str],
    tag_vocab: list[str],
    num_parallel_calls: int = tf.data.experimental.AUTOTUNE,
    label_distribution_shift: bool = False,
) -> Callable[[tf.data.Dataset], tf.data.Dataset]:
  """Creates a preprocessing function for Stack Overflow tag prediction data.

  This function creates a `tff.Computation` which takes a dataset, and returns
  a preprocessed dataset. This preprocessing shuffles the dataset, repeats it
  some number of times, takes a maximum number of examples, and then maps the
  elements to tuples of the form (tokens, tags), where tokens are bag-of-words
  vectors, and tags are binary vectors indicating that a given tag is associated
  with the example.

  Args:
    preprocess_spec: A `tff.simulation.baselines.ClientSpec` containing
      information on how to preprocess clients.
    word_vocab: A list of strings representing the in-vocabulary words.
    tag_vocab: A list of tokens representing the in-vocabulary tags.
    num_parallel_calls: An integer representing the number of parallel calls
      used when performing `tf.data.Dataset.map`.
    label_distribution_shift: Label distribution shift by masking 40% of the
      selected tags. For example, if we select top 50 tags, the 0/1 multi-label
      value for tag 30-49 will always be zero if
      `label_distribution_shift=True`.

  Returns:
    A callable taking as input a `tf.data.Dataset`, and returning a
    `tf.data.Dataset` formed by preprocessing according to the input arguments.
  """
  if not word_vocab:
    raise ValueError('word_vocab must be non-empty.')
  if not tag_vocab:
    raise ValueError('tag_vocab must be non-empty.')

  shuffle_buffer_size = preprocess_spec.shuffle_buffer_size
  if shuffle_buffer_size is None:
    shuffle_buffer_size = DEFAULT_SHUFFLE_BUFFER_SIZE

  def preprocess_fn(dataset):
    if shuffle_buffer_size > 1:
      dataset = dataset.shuffle(shuffle_buffer_size)
    if preprocess_spec.num_epochs > 1:
      dataset = dataset.repeat(preprocess_spec.num_epochs)
    if preprocess_spec.max_elements is not None:
      dataset = dataset.take(preprocess_spec.max_elements)
    to_ids = build_to_ids_fn(
        word_vocab, tag_vocab, label_distribution_shift=label_distribution_shift
    )
    dataset = dataset.map(to_ids, num_parallel_calls=num_parallel_calls)
    return dataset.batch(preprocess_spec.batch_size)

  return preprocess_fn
