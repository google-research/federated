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
import collections
from unittest import mock

import tensorflow as tf
import tensorflow_federated as tff

from utils.datasets import stackoverflow_tag_prediction


TEST_DATA = collections.OrderedDict(
    creation_date=(['unused date']),
    score=([tf.constant(0, dtype=tf.int64)]),
    tags=(['unused test tag']),
    title=(['unused title']),
    tokens=(['one must imagine']),
    type=(['unused type']),
)


class PreprocessFnTest(tf.test.TestCase):

  def test_word_tokens_to_ids_without_oov(self):
    word_vocab = ['A', 'B', 'C']
    tag_vocab = ['D', 'E', 'F']
    to_ids_fn = stackoverflow_tag_prediction.build_to_ids_fn(
        word_vocab, tag_vocab)
    data = {'tokens': 'A B C', 'title': '', 'tags': ''}
    processed = to_ids_fn(data)
    self.assertAllClose(self.evaluate(processed[0]), [1 / 3, 1 / 3, 1 / 3])

  def test_word_tokens_to_ids_with_oov(self):
    word_vocab = ['A', 'B']
    tag_vocab = ['D', 'E', 'F']
    to_ids_fn = stackoverflow_tag_prediction.build_to_ids_fn(
        word_vocab, tag_vocab)
    data = {'tokens': 'A B C', 'title': '', 'tags': ''}
    processed = to_ids_fn(data)
    self.assertAllClose(self.evaluate(processed[0]), [1 / 3, 1 / 3])

  def test_tag_tokens_to_ids_without_oov(self):
    word_vocab = ['A', 'B', 'C']
    tag_vocab = ['D', 'E', 'F']
    to_ids_fn = stackoverflow_tag_prediction.build_to_ids_fn(
        word_vocab, tag_vocab)
    data = {'tokens': '', 'title': '', 'tags': 'D|E|F'}
    processed = to_ids_fn(data)
    self.assertAllClose(self.evaluate(processed[1]), [1, 1, 1])

  def test_tag_tokens_to_ids_with_oov(self):
    word_vocab = ['A', 'B', 'C']
    tag_vocab = ['D', 'E']
    to_ids_fn = stackoverflow_tag_prediction.build_to_ids_fn(
        word_vocab, tag_vocab)
    data = {'tokens': '', 'title': '', 'tags': 'D|E|F'}
    processed = to_ids_fn(data)
    self.assertAllClose(self.evaluate(processed[1]), [1, 1])

  def test_join_word_tokens_with_title(self):
    word_vocab = ['A', 'B', 'C']
    tag_vocab = ['D', 'E', 'F']
    to_ids_fn = stackoverflow_tag_prediction.build_to_ids_fn(
        word_vocab, tag_vocab)
    data = {'tokens': 'A B C', 'title': 'A B', 'tags': ''}
    processed = to_ids_fn(data)
    self.assertAllClose(self.evaluate(processed[0]), [2 / 5, 2 / 5, 1 / 5])


STACKOVERFLOW_MODULE = 'tensorflow_federated.simulation.datasets.stackoverflow'


class FederatedDatasetTest(tf.test.TestCase):

  @mock.patch(STACKOVERFLOW_MODULE + '.load_tag_counts')
  @mock.patch(STACKOVERFLOW_MODULE + '.load_word_counts')
  @mock.patch(STACKOVERFLOW_MODULE + '.load_data')
  def test_preprocess_applied(self, mock_load_data, mock_load_word_counts,
                              mock_load_tag_counts):
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
    mock_load_tag_counts.return_value = collections.OrderedDict(a=1)

    _, _ = stackoverflow_tag_prediction.get_federated_datasets(
        word_vocab_size=1000,
        tag_vocab_size=500,
        train_client_batch_size=10,
        test_client_batch_size=100,
        train_client_epochs_per_round=1,
        test_client_epochs_per_round=1,
        max_elements_per_train_client=128,
        max_elements_per_test_client=-1)

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

    # Assert the tag counts were loaded once to apply to each dataset.
    mock_load_tag_counts.assert_called_once()


class CentralizedDatasetTest(tf.test.TestCase):

  @mock.patch(STACKOVERFLOW_MODULE + '.load_tag_counts')
  @mock.patch(STACKOVERFLOW_MODULE + '.load_word_counts')
  @mock.patch(STACKOVERFLOW_MODULE + '.load_data')
  def test_preprocess_applied(self, mock_load_data, mock_load_word_counts,
                              mock_load_tag_counts):
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
    mock_load_word_counts.return_value = collections.OrderedDict(a=1)
    mock_load_tag_counts.return_value = collections.OrderedDict(a=1)

    _, _, _ = stackoverflow_tag_prediction.get_centralized_datasets(
        word_vocab_size=1000,
        tag_vocab_size=500,
        train_batch_size=10,
        validation_batch_size=50,
        test_batch_size=100,
        num_validation_examples=10000)

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
    mock_load_tag_counts.assert_called_once()


if __name__ == '__main__':
  tf.test.main()
