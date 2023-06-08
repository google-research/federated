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
import collections

import numpy as np
import tensorflow as tf

from multi_epoch_dp_matrix_factorization.dp_ftrl.datasets import stackoverflow_word_prediction

TEST_DATA = collections.OrderedDict(
    creation_date=(['unused date']),
    score=([tf.constant(0, dtype=tf.int64)]),
    tags=(['unused test tag']),
    title=(['unused title']),
    tokens=(['one must imagine']),
    type=(['unused type']),
)

TEST_SEED = 0xBAD5EED


class ConvertToTokensTest(tf.test.TestCase):

  def test_split_input_target(self):
    tokens = tf.constant([[0, 1, 2, 3, 4]], dtype=tf.int64)
    expected_input = [[0, 1, 2, 3]]
    expected_target = [[1, 2, 3, 4]]
    split = stackoverflow_word_prediction.split_input_target(tokens)
    self.assertAllEqual(self.evaluate(split[0]), expected_input)
    self.assertAllEqual(self.evaluate(split[1]), expected_target)

  def test_build_to_ids_fn_truncates(self):
    vocab = ['A', 'B', 'C']
    max_seq_len = 1
    bos = stackoverflow_word_prediction.get_special_tokens(len(vocab)).bos
    to_ids_fn = stackoverflow_word_prediction.build_to_ids_fn(
        vocab, max_seq_len
    )
    data = {'tokens': 'A B C'}
    processed = to_ids_fn(data)
    self.assertAllEqual(self.evaluate(processed), [bos, 1])

  def test_build_to_ids_fn_embeds_all_vocab(self):
    vocab = ['A', 'B', 'C']
    max_seq_len = 5
    special_tokens = stackoverflow_word_prediction.get_special_tokens(
        len(vocab)
    )
    bos = special_tokens.bos
    eos = special_tokens.eos
    to_ids_fn = stackoverflow_word_prediction.build_to_ids_fn(
        vocab, max_seq_len
    )
    data = {'tokens': 'A B C'}
    processed = to_ids_fn(data)
    self.assertAllEqual(self.evaluate(processed), [bos, 1, 2, 3, eos])

  def test_pad_token_correct(self):
    vocab = ['A', 'B', 'C']
    max_seq_len = 5
    to_ids_fn = stackoverflow_word_prediction.build_to_ids_fn(
        vocab, max_seq_len
    )
    special_tokens = stackoverflow_word_prediction.get_special_tokens(
        len(vocab)
    )
    pad, bos, eos = special_tokens.pad, special_tokens.bos, special_tokens.eos
    data = {'tokens': 'A B C'}
    processed = to_ids_fn(data)
    batched_ds = tf.data.Dataset.from_tensor_slices([processed]).padded_batch(
        1, padded_shapes=[6]
    )
    sample_elem = next(iter(batched_ds))
    self.assertAllEqual(self.evaluate(sample_elem), [[bos, 1, 2, 3, eos, pad]])

  def test_oov_token_correct(self):
    vocab = ['A', 'B', 'C']
    max_seq_len = 5
    num_oov_buckets = 2
    to_ids_fn = stackoverflow_word_prediction.build_to_ids_fn(
        vocab, max_seq_len, num_oov_buckets=num_oov_buckets
    )
    oov_tokens = stackoverflow_word_prediction.get_special_tokens(
        len(vocab), num_oov_buckets=num_oov_buckets
    ).oov
    data = {'tokens': 'A B D'}
    processed = to_ids_fn(data)
    self.assertLen(oov_tokens, num_oov_buckets)
    self.assertIn(self.evaluate(processed)[3], oov_tokens)


class BatchAndSplitTest(tf.test.TestCase):

  def test_batch_and_split_fn_returns_dataset_with_correct_type_spec(self):
    token = tf.constant([[0, 1, 2, 3, 4]], dtype=tf.int64)
    ds = tf.data.Dataset.from_tensor_slices(token)
    padded_and_batched = stackoverflow_word_prediction.batch_and_split(
        ds, max_sequence_length=6, batch_size=1
    )
    self.assertIsInstance(padded_and_batched, tf.data.Dataset)
    self.assertEqual(
        padded_and_batched.element_spec,
        (
            tf.TensorSpec([None, 6], dtype=tf.int64),
            tf.TensorSpec([None, 6], dtype=tf.int64),
        ),
    )

  def test_batch_and_split_fn_returns_dataset_yielding_expected_elements(self):
    token = tf.constant([[0, 1, 2, 3, 4]], dtype=tf.int64)
    ds = tf.data.Dataset.from_tensor_slices(token)
    padded_and_batched = stackoverflow_word_prediction.batch_and_split(
        ds, max_sequence_length=6, batch_size=1
    )
    num_elems = 0
    for elem in padded_and_batched:
      self.assertAllEqual(
          self.evaluate(elem[0]), np.array([[0, 1, 2, 3, 4, 0]], np.int64)
      )
      self.assertAllEqual(
          self.evaluate(elem[1]), np.array([[1, 2, 3, 4, 0, 0]], np.int64)
      )
      num_elems += 1
    self.assertEqual(num_elems, 1)


class DatasetPreprocessFnTest(tf.test.TestCase):

  def test_preprocess_fn_return_dataset_element_spec(self):
    ds = tf.data.Dataset.from_tensor_slices(TEST_DATA)
    preprocess_fn = stackoverflow_word_prediction.create_preprocess_fn(
        client_batch_size=32,
        client_epochs_per_round=1,
        max_sequence_length=10,
        max_elements_per_client=100,
        vocab=['one', 'must'],
        num_oov_buckets=1,
    )
    preprocessed_ds = preprocess_fn(ds)
    self.assertEqual(
        preprocessed_ds.element_spec,
        (
            tf.TensorSpec(shape=[None, 10], dtype=tf.int64),
            tf.TensorSpec(shape=[None, 10], dtype=tf.int64),
        ),
    )

  def test_preprocess_fn_return_dataset_element_spec_oov_buckets(self):
    ds = tf.data.Dataset.from_tensor_slices(TEST_DATA)
    preprocess_fn = stackoverflow_word_prediction.create_preprocess_fn(
        client_batch_size=32,
        client_epochs_per_round=1,
        max_sequence_length=10,
        max_elements_per_client=100,
        vocab=['one', 'must'],
        num_oov_buckets=10,
    )
    preprocessed_ds = preprocess_fn(ds)
    self.assertEqual(
        preprocessed_ds.element_spec,
        (
            tf.TensorSpec(shape=[None, 10], dtype=tf.int64),
            tf.TensorSpec(shape=[None, 10], dtype=tf.int64),
        ),
    )

  def test_preprocess_fn_returns_correct_sequence(self):
    ds = tf.data.Dataset.from_tensor_slices(TEST_DATA)
    preprocess_fn = stackoverflow_word_prediction.create_preprocess_fn(
        client_batch_size=32,
        client_epochs_per_round=1,
        max_sequence_length=6,
        max_elements_per_client=100,
        vocab=['one', 'must'],
        num_oov_buckets=1,
    )

    preprocessed_ds = preprocess_fn(ds)
    element = next(iter(preprocessed_ds))

    # BOS is len(vocab)+2, EOS is len(vocab)+3, pad is 0, OOV is len(vocab)+1
    self.assertAllEqual(
        self.evaluate(element[0]), np.array([[4, 1, 2, 3, 5, 0]])
    )

  def test_preprocess_fn_returns_correct_sequence_oov_buckets(self):
    ds = tf.data.Dataset.from_tensor_slices(TEST_DATA)
    preprocess_fn = stackoverflow_word_prediction.create_preprocess_fn(
        client_batch_size=32,
        client_epochs_per_round=1,
        max_sequence_length=6,
        max_elements_per_client=100,
        vocab=['one', 'must'],
        num_oov_buckets=3,
    )
    preprocessed_ds = preprocess_fn(ds)
    element = next(iter(preprocessed_ds))
    # BOS is len(vocab)+3+1
    self.assertEqual(self.evaluate(element[0])[0][0], 6)
    self.assertEqual(self.evaluate(element[0])[0][1], 1)
    self.assertEqual(self.evaluate(element[0])[0][2], 2)
    # OOV is [len(vocab)+1, len(vocab)+2, len(vocab)+3]
    self.assertIn(self.evaluate(element[0])[0][3], [3, 4, 5])
    # EOS is len(vocab)+3+2
    self.assertEqual(self.evaluate(element[0])[0][4], 7)
    # pad is 0
    self.assertEqual(self.evaluate(element[0])[0][5], 0)


STACKOVERFLOW_MODULE = 'tensorflow_federated.simulation.datasets.stackoverflow'

if __name__ == '__main__':
  tf.test.main()
