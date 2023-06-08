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
"""Data loader for Stack Overflow next-word-prediction tasks."""

from collections.abc import Callable, Mapping

import attr
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

EVAL_BATCH_SIZE = 100


@attr.s(eq=False, frozen=True)
class SpecialTokens(object):
  """Structure for Special tokens.

  Attributes:
    pad: int - Special token for padding.
    oov: list - Special tokens for out of vocabulary tokens.
    bos: int - Special token for beginning of sentence.
    eos: int - Special token for end of sentence.
  """

  pad = attr.ib()
  oov = attr.ib()
  bos = attr.ib()
  eos = attr.ib()


def create_vocab(vocab_size: int) -> list[str]:
  """Creates vocab from `vocab_size` most common words in Stackoverflow."""
  return list(
      tff.simulation.datasets.stackoverflow.load_word_counts(
          vocab_size=vocab_size
      ).keys()
  )


def split_input_target(chunk: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
  """Generate input and target data.

  The task of language model is to predict the next word.

  Args:
    chunk: A Tensor of text data.

  Returns:
    A tuple of input and target data.
  """
  input_text = tf.map_fn(lambda x: x[:-1], chunk)
  target_text = tf.map_fn(lambda x: x[1:], chunk)
  return (input_text, target_text)


def build_to_ids_fn(
    vocab: list[str], max_sequence_length: int, num_oov_buckets: int = 1
) -> Callable[[Mapping[str, tf.Tensor]], tf.Tensor]:
  """Constructs function mapping examples to sequences of token indices."""
  special_tokens = get_special_tokens(len(vocab), num_oov_buckets)
  bos = special_tokens.bos
  eos = special_tokens.eos

  table_values = np.arange(len(vocab), dtype=np.int64)
  table = tf.lookup.StaticVocabularyTable(
      tf.lookup.KeyValueTensorInitializer(vocab, table_values),
      num_oov_buckets=num_oov_buckets,
  )

  def to_ids(example: Mapping[str, tf.Tensor]) -> tf.Tensor:
    sentence = tf.reshape(example['tokens'], shape=[1])
    words = tf.strings.split(sentence, sep=' ').values
    # NOTE: It would be worth checking that this truncation isn't
    # throwing away too much data, making the problem easier. It seems likely
    # better to just wrap long examples around.
    truncated_words = words[:max_sequence_length]
    tokens = table.lookup(truncated_words) + 1
    tokens = tf.cond(
        tf.less(tf.size(tokens), max_sequence_length),
        lambda: tf.concat([tokens, [eos]], 0),
        lambda: tokens,
    )

    return tf.concat([[bos], tokens], 0)

  return to_ids


def batch_and_split(
    dataset: tf.data.Dataset, max_sequence_length: int, batch_size: int
) -> tf.data.Dataset:
  return dataset.padded_batch(
      batch_size, padded_shapes=[max_sequence_length + 1]
  ).map(split_input_target, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def get_special_tokens(
    vocab_size: int, num_oov_buckets: int = 1
) -> SpecialTokens:
  """Gets tokens dataset preprocessing code will add to Stackoverflow."""
  return SpecialTokens(
      pad=0,
      oov=[vocab_size + 1 + n for n in range(num_oov_buckets)],
      bos=vocab_size + num_oov_buckets + 1,
      eos=vocab_size + num_oov_buckets + 2,
  )


def create_preprocess_fn(
    vocab: list[str],
    num_oov_buckets: int,
    client_batch_size: int,
    client_epochs_per_round: int,
    max_sequence_length: int,
    max_elements_per_client: int,
    max_shuffle_buffer_size: int = 10000,
) -> Callable[[tf.data.Dataset], tf.data.Dataset]:
  """Creates a preprocessing functions for Stack Overflow next-word-prediction.

  This function returns a `tff.Computation`-compatible function
  which takes a dataset and returns a dataset, suitable for mapping
  over a set of unprocessed client datasets.

  Args:
    vocab: Vocabulary which defines the embedding.
    num_oov_buckets: The number of out of vocabulary buckets. Tokens that are
      not present in the `vocab` are hashed into one of these buckets.
    client_batch_size: Integer representing batch size to use on the clients.
    client_epochs_per_round: Number of epochs for which to repeat train client
      dataset. Must be a positive integer.
    max_sequence_length: Integer determining shape of padded batches. Sequences
      will be padded up to this length, and sentences longer than this will be
      truncated to this length.
    max_elements_per_client: Integer controlling the maximum number of elements
      to take per client. If -1, keeps all elements for each client. This is
      applied before repeating `client_epochs_per_round`, and is intended
      primarily to contend with the small set of clients with tens of thousands
      of examples.
    max_shuffle_buffer_size: Maximum shuffle buffer size.

  Returns:
    A callable taking as input a `tf.data.Dataset`, and returning a
    `tf.data.Dataset` formed by preprocessing according to the input arguments.
  """
  if client_batch_size <= 0:
    raise ValueError(
        'client_batch_size must be a positive integer. You have '
        'passed {}.'.format(client_batch_size)
    )
  elif client_epochs_per_round <= 0:
    raise ValueError(
        'client_epochs_per_round must be a positive integer. '
        'You have passed {}.'.format(client_epochs_per_round)
    )
  elif max_sequence_length <= 0:
    raise ValueError(
        'max_sequence_length must be a positive integer. You have '
        'passed {}.'.format(max_sequence_length)
    )
  elif max_elements_per_client == 0 or max_elements_per_client < -1:
    raise ValueError(
        'max_elements_per_client must be a positive integer or -1. You have '
        'passed {}.'.format(max_elements_per_client)
    )
  if num_oov_buckets <= 0:
    raise ValueError(
        'num_oov_buckets must be a positive integer. You have '
        'passed {}.'.format(num_oov_buckets)
    )

  if (
      max_elements_per_client == -1
      or max_elements_per_client > max_shuffle_buffer_size
  ):
    shuffle_buffer_size = max_shuffle_buffer_size
  else:
    shuffle_buffer_size = max_elements_per_client

  def preprocess_fn(dataset):
    to_ids = build_to_ids_fn(
        vocab=vocab,
        max_sequence_length=max_sequence_length,
        num_oov_buckets=num_oov_buckets,
    )
    dataset = (
        dataset.take(max_elements_per_client)
        .shuffle(shuffle_buffer_size)
        .repeat(client_epochs_per_round)
        .map(to_ids, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    )
    return batch_and_split(dataset, max_sequence_length, client_batch_size)

  return preprocess_fn
