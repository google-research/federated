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

import tensorflow as tf

from utils.datasets import stackoverflow_tag_prediction


def _compute_length_of_dataset(ds):
  return ds.reduce(0, lambda x, _: x + 1)


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


class FederatedDatasetTest(tf.test.TestCase):

  def test_federated_dataset_structure(self):
    stackoverflow_train, stackoverflow_test = stackoverflow_tag_prediction.get_federated_datasets(
        word_vocab_size=100,
        tag_vocab_size=5,
        train_client_batch_size=2,
        test_client_batch_size=3)
    self.assertEqual(len(stackoverflow_train.client_ids), 342477)
    self.assertEqual(len(stackoverflow_test.client_ids), 204088)
    sample_train_ds = stackoverflow_train.create_tf_dataset_for_client(
        stackoverflow_train.client_ids[0])
    sample_test_ds = stackoverflow_test.create_tf_dataset_for_client(
        stackoverflow_test.client_ids[0])

    train_batch = next(iter(sample_train_ds))
    test_batch = next(iter(sample_test_ds))
    self.assertEqual(train_batch[0].shape.as_list(), [2, 100])
    self.assertEqual(train_batch[1].shape.as_list(), [2, 5])
    self.assertEqual(test_batch[0].shape.as_list(), [3, 100])
    self.assertEqual(test_batch[1].shape.as_list(), [3, 5])

  def test_raises_negative_train_epochs(self):
    if tf.config.list_logical_devices('GPU'):
      self.skipTest('skip GPU test')
    with self.assertRaisesRegex(
        ValueError, 'client_epochs_per_round must be a positive integer.'):
      stackoverflow_tag_prediction.get_federated_datasets(
          word_vocab_size=1000,
          tag_vocab_size=500,
          train_client_epochs_per_round=-1)

  def test_raises_negative_test_epochs(self):
    if tf.config.list_logical_devices('GPU'):
      self.skipTest('skip GPU test')
    with self.assertRaisesRegex(
        ValueError, 'client_epochs_per_round must be a positive integer.'):
      stackoverflow_tag_prediction.get_federated_datasets(
          word_vocab_size=1000,
          tag_vocab_size=500,
          test_client_epochs_per_round=-1)

  def test_nonpositive_shuffle_buffer_size_in_federated_datasets(self):
    stackoverflow_train, _ = stackoverflow_tag_prediction.get_federated_datasets(
        train_client_batch_size=3, train_shuffle_buffer_size=-1)

    sample_train_ds = stackoverflow_train.create_tf_dataset_for_client(
        stackoverflow_train.client_ids[0])

    train_iter1 = iter(sample_train_ds)
    train_iter2 = iter(sample_train_ds)
    for _ in range(5):
      batch1 = next(train_iter1)
      batch2 = next(train_iter2)
      self.assertAllClose(batch1, batch2)


class CentralizedDatasetTest(tf.test.TestCase):

  def test_centralized_dataset_structure(self):
    global_train, global_val, global_test = stackoverflow_tag_prediction.get_centralized_datasets(
        word_vocab_size=100,
        tag_vocab_size=5,
        train_batch_size=2,
        validation_batch_size=3,
        test_batch_size=5,
        num_validation_examples=10000)

    train_batch = next(iter(global_train))
    val_batch = next(iter(global_val))
    test_batch = next(iter(global_test))
    self.assertEqual(train_batch[0].shape.as_list(), [2, 100])
    self.assertEqual(train_batch[1].shape.as_list(), [2, 5])
    self.assertEqual(val_batch[0].shape.as_list(), [3, 100])
    self.assertEqual(val_batch[1].shape.as_list(), [3, 5])
    self.assertEqual(test_batch[0].shape.as_list(), [5, 100])
    self.assertEqual(test_batch[1].shape.as_list(), [5, 5])

  def test_nonpositive_shuffle_buffer_size_in_centralized_datasets(self):
    stackoverflow_train, _, _ = stackoverflow_tag_prediction.get_centralized_datasets(
        train_batch_size=3, train_shuffle_buffer_size=-1)

    train_iter1 = iter(stackoverflow_train)
    train_iter2 = iter(stackoverflow_train)
    for _ in range(5):
      batch1 = next(train_iter1)
      batch2 = next(train_iter2)
      self.assertAllClose(batch1, batch2)


if __name__ == '__main__':
  tf.test.main()
