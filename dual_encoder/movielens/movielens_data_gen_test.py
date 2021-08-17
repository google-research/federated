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

import collections
import os
import tempfile

from absl.testing import absltest
import pandas as pd
import tensorflow as tf

from dual_encoder.movielens import movielens_data_gen

_TEST_DIR = 'dual_encoder/movielens/testdata'


def _int64_feature(value_list):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value_list))


class MovielensDataGenTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    self.example_1 = tf.train.Example(
        features=tf.train.Features(
            feature={
                'context': _int64_feature([1, 0, 0]),
                'label': _int64_feature([5])
            }))

    self.example_2 = tf.train.Example(
        features=tf.train.Features(
            feature={
                'context': _int64_feature([1, 5, 0]),
                'label': _int64_feature([3])
            }))

    self.example_3 = tf.train.Example(
        features=tf.train.Features(
            feature={
                'context': _int64_feature([1, 5, 3]),
                'label': _int64_feature([3])
            }))

    self.example_4 = tf.train.Example(
        features=tf.train.Features(
            feature={
                'context': _int64_feature([3, 0, 0]),
                'label': _int64_feature([1])
            }))

    self.ratings_df = pd.DataFrame(
        {
            'UserID': [1, 1, 1, 7, 1, 3, 3, 4],
            'MovieID': [1, 3, 5, 2, 3, 1, 3, 2],
            'Rating': [1, 2, 4, 3, 2, 3, 5, 1],
            'Timestamp': [1, 2, 1, 4, 5, 9, 1, 3]
        })
    self.timelines = collections.defaultdict(list)
    self.timelines = {1: [1, 5, 3, 3],
                      7: [2],
                      3: [3, 1],
                      4: [2]
                      }

    # Parameters
    self.tmp_dir = tempfile.mkdtemp()

  def test_read_ratings(self):
    test_ratings_df = movielens_data_gen.read_ratings(_TEST_DIR, self.tmp_dir)
    # The assertListEqual doesn't compare the datatype (e.g. 1 & 1.0 equal).
    self.assertListEqual(list(test_ratings_df.columns),
                         list(self.ratings_df.columns))
    self.assertListEqual(list(test_ratings_df['UserID']),
                         list(self.ratings_df['UserID']))
    self.assertListEqual(list(test_ratings_df['MovieID']),
                         list(self.ratings_df['MovieID']))
    self.assertListEqual(list(test_ratings_df['Rating']),
                         list(self.ratings_df['Rating']))
    self.assertListEqual(list(test_ratings_df['Timestamp']),
                         list(self.ratings_df['Timestamp']))

  def test_split_ratings_df(self):
    test_train_ratings_df, test_val_ratings_df, test_test_ratings_df = (
        movielens_data_gen.split_ratings_df(self.ratings_df, 0.3, 0.3))
    self.assertListEqual(list(test_train_ratings_df['UserID']), [1, 1, 1, 1])
    self.assertListEqual(list(test_val_ratings_df['UserID']), [3, 3])
    self.assertListEqual(list(test_test_ratings_df['UserID']), [7, 4])

  def test_split_ratings_df_empty_val(self):
    test_train_ratings_df, test_val_ratings_df, test_test_ratings_df = (
        movielens_data_gen.split_ratings_df(self.ratings_df, 0.3, 0.0))
    self.assertListEqual(list(test_train_ratings_df['UserID']), [1, 1, 1, 1])
    self.assertEmpty(list(test_val_ratings_df['UserID']))
    self.assertListEqual(list(test_test_ratings_df['UserID']), [7, 3, 3, 4])

  def test_split_ratings_df_empty_test(self):
    test_train_ratings_df, test_val_ratings_df, test_test_ratings_df = (
        movielens_data_gen.split_ratings_df(self.ratings_df, 0.3, 0.7))
    self.assertListEqual(list(test_train_ratings_df['UserID']), [1, 1, 1, 1])
    self.assertListEqual(list(test_val_ratings_df['UserID']), [7, 3, 3, 4])
    self.assertEmpty(list(test_test_ratings_df['UserID']))

  def test_convert_to_timelines(self):
    test_timelines = movielens_data_gen.convert_to_timelines(self.ratings_df)
    self.assertDictEqual(test_timelines, self.timelines)

  def test_generate_examples_from_a_single_timeline(self):
    test_timeline = self.timelines[1]
    test_examples = (
        movielens_data_gen.generate_examples_from_a_single_timeline(
            test_timeline, 3, 0))
    test_parsed_examples = []
    for serialized_example in test_examples:
      example = tf.train.Example()
      example.ParseFromString(serialized_example)
      test_parsed_examples.append(example)
    self.assertLen(test_examples, 3)
    self.assertListEqual(test_parsed_examples,
                         [self.example_1, self.example_2, self.example_3])

  def test_generate_examples_from_timelines(self):
    test_train_examples, test_val_examples, test_test_examples = (
        movielens_data_gen.generate_examples_from_timelines(
            self.timelines, 2, 3, 0, 0.5, 0.3, 2))

    test_parsed_train_examples = []
    test_parsed_val_examples = []
    test_parsed_test_examples = []

    for serialized_example in test_train_examples:
      example = tf.train.Example()
      example.ParseFromString(serialized_example)
      test_parsed_train_examples.append(example)

    for serialized_example in test_val_examples:
      example = tf.train.Example()
      example.ParseFromString(serialized_example)
      test_parsed_val_examples.append(example)

    for serialized_example in test_test_examples:
      example = tf.train.Example()
      example.ParseFromString(serialized_example)
      test_parsed_test_examples.append(example)

    self.assertLen(test_train_examples, 2)
    self.assertLen(test_val_examples, 1)
    self.assertLen(test_test_examples, 1)
    self.assertListEqual(test_parsed_train_examples,
                         [self.example_2, self.example_3])
    self.assertListEqual(test_parsed_val_examples,
                         [self.example_4])
    self.assertListEqual(test_parsed_test_examples,
                         [self.example_1])

  def test_write_tfrecords(self):
    examples = [self.example_1.SerializeToString(),
                self.example_2.SerializeToString()]
    filename = os.path.join(self.tmp_dir, 'test.tfrecord')
    movielens_data_gen.write_tfrecords(examples, filename)
    self.assertTrue(os.path.exists(filename))

  def test_generate_examples_per_user(self):
    test_examples_per_user = (
        movielens_data_gen.generate_examples_per_user(
            self.timelines, 2, 3, 0))
    self.assertListEqual(list(test_examples_per_user.keys()), [1, 3])
    self.assertLen(test_examples_per_user[1], 3)
    self.assertLen(test_examples_per_user[3], 1)

  def test_generate_examples_per_user_max_example_zero(self):
    test_examples_per_user = (
        movielens_data_gen.generate_examples_per_user(
            self.timelines, 2, 3, 0, 0))
    self.assertListEqual(list(test_examples_per_user.keys()), [1, 3])
    self.assertLen(test_examples_per_user[1], 3)
    self.assertLen(test_examples_per_user[3], 1)

  def test_generate_examples_per_user_max_example_nonzero(self):
    test_examples_per_user = (
        movielens_data_gen.generate_examples_per_user(
            self.timelines, 2, 3, 0, 2))
    self.assertListEqual(list(test_examples_per_user.keys()), [1, 3])
    self.assertLen(test_examples_per_user[1], 2)
    self.assertLen(test_examples_per_user[3], 1)

  def test_shuffle_examples_across_users(self):
    examples = {1: [1, 2, 3],
                2: [9, 5, 4],
                3: [7, 6]}
    shuffled_examples = {1: [9, 7, 2],
                         2: [4, 6, 1],
                         3: [5, 3]}
    test_shuffled_examples = (
        movielens_data_gen.shuffle_examples_across_users(examples, seed=1))
    self.assertDictEqual(test_shuffled_examples, shuffled_examples)

  def test_decode_example_with_use_example_weight(self):
    test_timeline = self.timelines[1]
    test_example = (
        movielens_data_gen.generate_examples_from_a_single_timeline(
            test_timeline, 3, 0))[1]
    test_decoded_example = movielens_data_gen.decode_example(test_example)
    self.assertLen(test_decoded_example, 2)
    self.assertListEqual([1, 5, 0], list(test_decoded_example[0]['context']))
    self.assertEqual([3], test_decoded_example[0]['label'])
    self.assertEqual(1.0, test_decoded_example[1])

  def test_decode_example_without_use_example_weight(self):
    test_timeline = self.timelines[1]
    test_example = (
        movielens_data_gen.generate_examples_from_a_single_timeline(
            test_timeline, 3, 0))[1]
    test_decoded_example = movielens_data_gen.decode_example(
        test_example, False)
    print(test_decoded_example)
    self.assertLen(test_decoded_example, 2)
    self.assertListEqual([1, 5, 0], list(test_decoded_example[0]['context']))
    self.assertEqual([3], test_decoded_example[0]['label'])
    self.assertEqual([3], test_decoded_example[1])

  def test_create_tf_datasets(self):
    test_examples_per_user = (
        movielens_data_gen.generate_examples_per_user(
            self.timelines, 2, 3, 0))
    test_dataset_per_user = (
        movielens_data_gen.create_tf_datasets(test_examples_per_user,
                                              batch_size=1,
                                              num_local_epochs=1))

    self.assertLen(test_dataset_per_user, 2)
    self.assertLen(list(test_dataset_per_user[0].as_numpy_iterator()), 3)
    self.assertLen(list(test_dataset_per_user[1].as_numpy_iterator()), 1)

  def test_create_tf_datasets_batch_size(self):
    test_examples_per_user = (
        movielens_data_gen.generate_examples_per_user(
            self.timelines, 2, 3, 0))
    test_dataset_per_user = (
        movielens_data_gen.create_tf_datasets(test_examples_per_user,
                                              batch_size=2,
                                              num_local_epochs=1))

    self.assertLen(test_dataset_per_user, 2)
    self.assertLen(list(test_dataset_per_user[0].as_numpy_iterator()), 2)
    self.assertLen(list(test_dataset_per_user[1].as_numpy_iterator()), 1)

  def test_create_tf_datasets_num_local_epochs(self):
    test_examples_per_user = (
        movielens_data_gen.generate_examples_per_user(
            self.timelines, 2, 3, 0, 0))
    test_dataset_per_user = (
        movielens_data_gen.create_tf_datasets(test_examples_per_user,
                                              batch_size=1,
                                              num_local_epochs=2))

    self.assertLen(test_dataset_per_user, 2)
    self.assertLen(list(test_dataset_per_user[0].as_numpy_iterator()), 6)
    self.assertLen(list(test_dataset_per_user[1].as_numpy_iterator()), 2)

  def test_create_client_datasets(self):
    test_client_datasets = (
        movielens_data_gen.create_client_datasets(
            self.ratings_df,
            min_timeline_len=2,
            max_context_len=3,
            max_examples_per_user=0,
            pad_id=0,
            shuffle_across_users=False,
            batch_size=2,
            num_local_epochs=1))
    self.assertLen(test_client_datasets, 2)
    self.assertLen(list(test_client_datasets[0].as_numpy_iterator()), 2)
    self.assertLen(list(test_client_datasets[1].as_numpy_iterator()), 1)


if __name__ == '__main__':
  absltest.main()
