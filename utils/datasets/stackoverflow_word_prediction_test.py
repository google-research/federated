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
import collections
import hashlib
import string
from unittest import mock

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from utils.datasets import stackoverflow_word_prediction

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
        vocab, max_seq_len)
    data = {'tokens': 'A B C'}
    processed = to_ids_fn(data)
    self.assertAllEqual(self.evaluate(processed), [bos, 1])

  def test_build_to_ids_fn_embeds_all_vocab(self):
    vocab = ['A', 'B', 'C']
    max_seq_len = 5
    special_tokens = stackoverflow_word_prediction.get_special_tokens(
        len(vocab))
    bos = special_tokens.bos
    eos = special_tokens.eos
    to_ids_fn = stackoverflow_word_prediction.build_to_ids_fn(
        vocab, max_seq_len)
    data = {'tokens': 'A B C'}
    processed = to_ids_fn(data)
    self.assertAllEqual(self.evaluate(processed), [bos, 1, 2, 3, eos])

  def test_pad_token_correct(self):
    vocab = ['A', 'B', 'C']
    max_seq_len = 5
    to_ids_fn = stackoverflow_word_prediction.build_to_ids_fn(
        vocab, max_seq_len)
    special_tokens = stackoverflow_word_prediction.get_special_tokens(
        len(vocab))
    pad, bos, eos = special_tokens.pad, special_tokens.bos, special_tokens.eos
    data = {'tokens': 'A B C'}
    processed = to_ids_fn(data)
    batched_ds = tf.data.Dataset.from_tensor_slices([processed]).padded_batch(
        1, padded_shapes=[6])
    sample_elem = next(iter(batched_ds))
    self.assertAllEqual(self.evaluate(sample_elem), [[bos, 1, 2, 3, eos, pad]])

  def test_oov_token_correct(self):
    vocab = ['A', 'B', 'C']
    max_seq_len = 5
    num_oov_buckets = 2
    to_ids_fn = stackoverflow_word_prediction.build_to_ids_fn(
        vocab, max_seq_len, num_oov_buckets=num_oov_buckets)
    oov_tokens = stackoverflow_word_prediction.get_special_tokens(
        len(vocab), num_oov_buckets=num_oov_buckets).oov
    data = {'tokens': 'A B D'}
    processed = to_ids_fn(data)
    self.assertLen(oov_tokens, num_oov_buckets)
    self.assertIn(self.evaluate(processed)[3], oov_tokens)


class BatchAndSplitTest(tf.test.TestCase):

  def test_batch_and_split_fn_returns_dataset_with_correct_type_spec(self):
    token = tf.constant([[0, 1, 2, 3, 4]], dtype=tf.int64)
    ds = tf.data.Dataset.from_tensor_slices(token)
    padded_and_batched = stackoverflow_word_prediction.batch_and_split(
        ds, max_sequence_length=6, batch_size=1)
    self.assertIsInstance(padded_and_batched, tf.data.Dataset)
    self.assertEqual(padded_and_batched.element_spec, (tf.TensorSpec(
        [None, 6], dtype=tf.int64), tf.TensorSpec([None, 6], dtype=tf.int64)))

  def test_batch_and_split_fn_returns_dataset_yielding_expected_elements(self):
    token = tf.constant([[0, 1, 2, 3, 4]], dtype=tf.int64)
    ds = tf.data.Dataset.from_tensor_slices(token)
    padded_and_batched = stackoverflow_word_prediction.batch_and_split(
        ds, max_sequence_length=6, batch_size=1)
    num_elems = 0
    for elem in padded_and_batched:
      self.assertAllEqual(
          self.evaluate(elem[0]), np.array([[0, 1, 2, 3, 4, 0]], np.int64))
      self.assertAllEqual(
          self.evaluate(elem[1]), np.array([[1, 2, 3, 4, 0, 0]], np.int64))
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
        num_oov_buckets=1)
    preprocessed_ds = preprocess_fn(ds)
    self.assertEqual(preprocessed_ds.element_spec,
                     (tf.TensorSpec(shape=[None, 10], dtype=tf.int64),
                      tf.TensorSpec(shape=[None, 10], dtype=tf.int64)))

  def test_preprocess_fn_return_dataset_element_spec_oov_buckets(self):
    ds = tf.data.Dataset.from_tensor_slices(TEST_DATA)
    preprocess_fn = stackoverflow_word_prediction.create_preprocess_fn(
        client_batch_size=32,
        client_epochs_per_round=1,
        max_sequence_length=10,
        max_elements_per_client=100,
        vocab=['one', 'must'],
        num_oov_buckets=10)
    preprocessed_ds = preprocess_fn(ds)
    self.assertEqual(preprocessed_ds.element_spec,
                     (tf.TensorSpec(shape=[None, 10], dtype=tf.int64),
                      tf.TensorSpec(shape=[None, 10], dtype=tf.int64)))

  def test_preprocess_fn_returns_correct_sequence(self):
    ds = tf.data.Dataset.from_tensor_slices(TEST_DATA)
    preprocess_fn = stackoverflow_word_prediction.create_preprocess_fn(
        client_batch_size=32,
        client_epochs_per_round=1,
        max_sequence_length=6,
        max_elements_per_client=100,
        vocab=['one', 'must'],
        num_oov_buckets=1)

    preprocessed_ds = preprocess_fn(ds)
    element = next(iter(preprocessed_ds))

    # BOS is len(vocab)+2, EOS is len(vocab)+3, pad is 0, OOV is len(vocab)+1
    self.assertAllEqual(
        self.evaluate(element[0]), np.array([[4, 1, 2, 3, 5, 0]]))

  def test_preprocess_fn_returns_correct_sequence_oov_buckets(self):
    ds = tf.data.Dataset.from_tensor_slices(TEST_DATA)
    preprocess_fn = stackoverflow_word_prediction.create_preprocess_fn(
        client_batch_size=32,
        client_epochs_per_round=1,
        max_sequence_length=6,
        max_elements_per_client=100,
        vocab=['one', 'must'],
        num_oov_buckets=3)
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


class FederatedDatasetTest(tf.test.TestCase):

  @mock.patch(STACKOVERFLOW_MODULE + '.load_word_counts')
  @mock.patch(STACKOVERFLOW_MODULE + '.load_data')
  def test_preprocess_applied(self, mock_load_data, mock_load_word_counts):
    if tf.config.list_logical_devices('GPU'):
      self.skipTest('skip GPU test')
    # Mock out the actual data loading from disk. Assert that the preprocessing
    # function is applied to the client data, and that only the ClientData
    # objects we desired are used.
    #
    # The correctness of the preprocessing function is tested in other tests.
    mock_train = mock.create_autospec(tff.simulation.datasets.ClientData)
    mock_validation = mock.create_autospec(tff.simulation.datasets.ClientData)
    mock_test = mock.create_autospec(tff.simulation.datasets.ClientData)
    mock_load_data.return_value = (mock_train, mock_validation, mock_test)
    # Return a factor word dictionary.
    mock_load_word_counts.return_value = collections.OrderedDict(a=1)

    _, _ = stackoverflow_word_prediction.get_federated_datasets(
        vocab_size=1000,
        train_client_batch_size=10,
        test_client_batch_size=100,
        train_client_epochs_per_round=1,
        test_client_epochs_per_round=1,
        max_sequence_length=20,
        max_elements_per_train_client=128,
        max_elements_per_test_client=-1,
        num_oov_buckets=1)

    # Assert the validation ClientData isn't used.
    mock_load_data.assert_called_once()
    self.assertEmpty(mock_validation.mock_calls)

    # Assert the training and testing data are preprocessed.
    self.assertEqual(mock_train.mock_calls,
                     mock.call.preprocess(mock.ANY).call_list())
    self.assertEqual(mock_test.mock_calls,
                     mock.call.preprocess(mock.ANY).call_list())

    # Assert the word counts were loaded once to apply to each dataset.
    mock_load_word_counts.assert_called_once()

  @mock.patch(STACKOVERFLOW_MODULE + '.load_word_counts')
  @mock.patch(STACKOVERFLOW_MODULE + '.load_data')
  def test_raises_no_repeat_and_no_take(self, mock_load_data,
                                        mock_load_word_counts):
    if tf.config.list_logical_devices('GPU'):
      self.skipTest('skip GPU test')
    mock_load_data.return_value = (mock.Mock(), mock.Mock(), mock.Mock())
    with self.assertRaisesRegex(
        ValueError, 'client_epochs_per_round must be a positive integer.'):
      stackoverflow_word_prediction.get_federated_datasets(
          vocab_size=100,
          train_client_batch_size=10,
          train_client_epochs_per_round=-1,
          max_sequence_length=20,
          max_elements_per_train_client=128,
          num_oov_buckets=1)


class CentralizedDatasetTest(tf.test.TestCase):

  @mock.patch(STACKOVERFLOW_MODULE + '.load_word_counts')
  @mock.patch(STACKOVERFLOW_MODULE + '.load_data')
  def test_preprocess_applied(self, mock_load_data, mock_load_word_counts):
    if tf.config.list_logical_devices('GPU'):
      self.skipTest('skip GPU test')
    # Mock out the actual data loading from disk. Assert that the preprocessing
    # function is applied to the client data, and that only the ClientData
    # objects we desired are used.
    #
    # The correctness of the preprocessing function is tested in other tests.
    sample_ds = tf.data.Dataset.from_tensor_slices(TEST_DATA)

    mock_train = mock.create_autospec(tff.simulation.datasets.ClientData)
    mock_train.create_tf_dataset_from_all_clients = mock.Mock(
        return_value=sample_ds)

    mock_validation = mock.create_autospec(tff.simulation.datasets.ClientData)

    mock_test = mock.create_autospec(tff.simulation.datasets.ClientData)
    mock_test.create_tf_dataset_from_all_clients = mock.Mock(
        return_value=sample_ds)

    mock_load_data.return_value = (mock_train, mock_validation, mock_test)
    # Return a factor word dictionary.
    mock_load_word_counts.return_value = collections.OrderedDict(a=1)

    _, _, _ = stackoverflow_word_prediction.get_centralized_datasets(
        vocab_size=1000,
        train_batch_size=10,
        validation_batch_size=50,
        test_batch_size=100,
        num_validation_examples=10000,
        max_sequence_length=20,
        num_oov_buckets=1)

    # Assert the validation ClientData isn't used.
    mock_load_data.assert_called_once()
    self.assertEmpty(mock_validation.mock_calls)

    # Assert the validation ClientData isn't used, and the train and test
    # are amalgamated into datasets single datasets over all clients.
    mock_load_data.assert_called_once()
    self.assertEmpty(mock_validation.mock_calls)
    self.assertEqual(mock_train.mock_calls,
                     mock.call.create_tf_dataset_from_all_clients().call_list())
    self.assertEqual(mock_test.mock_calls,
                     mock.call.create_tf_dataset_from_all_clients().call_list())

    # Assert the word counts were loaded once to apply to each dataset.
    mock_load_word_counts.assert_called_once()


class SecretInsertionTest(tf.test.TestCase):

  def test_secret_inserting_transform_fn(self):
    # The built-in synthetic data is too small to ensure all random events
    # occur. So we make a larger synthetic dataset with all random components.

    num_clients = 1000
    num_examples = 20

    np.random.seed(TEST_SEED)

    def random_string():
      return ''.join(np.random.choice(list(string.ascii_uppercase), 5))

    words = ['a' for _ in range(num_examples)]

    def client_data():
      return collections.OrderedDict(
          # We hash on date to seed client randomness.
          creation_date=[random_string() for _ in range(num_examples)],
          score=words,
          tags=words,
          title=words,
          tokens=words,
          type=words)

    data = tff.test.FromTensorSlicesClientData(
        {random_string(): client_data() for _ in range(num_clients)})

    secrets = dict(b=(500, 0.5), c=(30, 1.0), d=(1, 1.0))

    transform_fn = stackoverflow_word_prediction.secret_inserting_transform_fn(
        data.client_ids, secrets, TEST_SEED + 1)
    transformed = tff.simulation.datasets.TransformingClientData(
        data, transform_fn)

    secret_count = {secret: 0 for secret in secrets}
    client_with_secret_count = {secret: 0 for secret in secrets}

    for client_id in transformed.client_ids:
      client_data = transformed.create_tf_dataset_for_client(client_id)
      has_secret = None
      for example in client_data.enumerate():
        tokens = example[1]['tokens'].numpy().decode('utf-8')
        if tokens in secrets:
          secret_count[tokens] += 1
          has_secret = tokens
      if has_secret:
        client_with_secret_count[has_secret] += 1

    # All selected clients should have at least one insertion, hence should
    # be counted. The probability that there exists a client selected for secret
    # b that doesn't have a single insertion is 1-(1-0.5^20)^500 ~ 5e-4.
    self.assertAllEqual(500, client_with_secret_count['b'])
    self.assertAllEqual(30, client_with_secret_count['c'])
    self.assertAllEqual(1, client_with_secret_count['d'])

    # Count of secret b is Bin(10000, 0.5) with mean 5000, stddev 50.
    self.assertAllClose(5000, secret_count['b'], atol=150)
    self.assertAllEqual(30 * num_examples, secret_count['c'])
    self.assertAllEqual(num_examples, secret_count['d'])

  @mock.patch(STACKOVERFLOW_MODULE + '.load_word_counts')
  def test_make_random_secrets(self, mock_load_word_counts):
    mock_load_word_counts.return_value = collections.OrderedDict(
        zip(string.ascii_uppercase, range(26)))

    secrets = stackoverflow_word_prediction.make_random_secrets(
        10, 10, 5, TEST_SEED)
    self.assertLen(secrets, 10)

    allowed_tokens = set(string.ascii_uppercase[:10])
    longhash = hashlib.md5()
    for secret in secrets:
      tokens = secret.split(' ')
      self.assertLen(tokens, 5)
      self.assertAllInSet(tokens, allowed_tokens)
      longhash.update(secret.encode())

    # Check that there is no accidental non-determinism.
    self.assertEqual(
        2353504054,
        int.from_bytes(longhash.digest(), byteorder='big') % (2**32))


if __name__ == '__main__':
  tf.test.main()
