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
"""Tests for movielens_dataset.py."""

import collections

import numpy as np
import pandas as pd
import tensorflow as tf

from reconstruction.movielens import movielens_dataset


class MovielensDatasetTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()

    self.ratings_df = pd.DataFrame(
        {
            'UserID': [0, 0, 1, 1, 1, 1],
            'MovieID': [0, 1, 2, 3, 2, 2],
            'Rating': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            'Timestamp': [1, 2, 1, 4, 5, 3]
        },
        columns=['UserID', 'MovieID', 'Rating', 'Timestamp'])

  def _verify_server_data_arrays(self, users, movies, ratings):
    """Ensures produced data arrays have expected dtype."""
    self.assertIsInstance(users, np.ndarray)
    self.assertEqual(users.dtype, np.int64)
    self.assertIsInstance(movies, np.ndarray)
    self.assertEqual(movies.dtype, np.int64)
    self.assertIsInstance(ratings, np.ndarray)
    self.assertEqual(ratings.dtype, np.float32)

  def test_split_ratings_df(self):
    train_ratings_df, val_ratings_df, test_ratings_df = movielens_dataset.split_ratings_df(
        self.ratings_df, train_fraction=0.5, val_fraction=0.25)

    self.assertLen(train_ratings_df, 3)
    # 1 example from user 1 goes to train, the other goes to test, since
    # `val_fraction` upper bounds the number of examples provided for each user
    # and remaining examples go to the test set.
    self.assertLen(val_ratings_df, 1)
    self.assertLen(test_ratings_df, 2)
    self.assertListEqual(list(train_ratings_df['UserID']), [0, 1, 1])
    self.assertListEqual(list(train_ratings_df['MovieID']), [0, 2, 2])
    self.assertListEqual(list(train_ratings_df['Rating']), [1.0, 3.0, 6.0])
    self.assertListEqual(list(train_ratings_df['Timestamp']), [1, 1, 3])

    self.assertListEqual(list(val_ratings_df['UserID']), [1])
    self.assertListEqual(list(val_ratings_df['MovieID']), [3])
    self.assertListEqual(list(val_ratings_df['Rating']), [4.0])
    self.assertListEqual(list(val_ratings_df['Timestamp']), [4])

    self.assertListEqual(list(test_ratings_df['UserID']), [0, 1])
    self.assertListEqual(list(test_ratings_df['MovieID']), [1, 2])
    self.assertListEqual(list(test_ratings_df['Rating']), [2.0, 5.0])
    self.assertListEqual(list(test_ratings_df['Timestamp']), [2, 5])

  def test_split_ratings_df_raises_error(self):
    with self.assertRaises(ValueError):
      movielens_dataset.split_ratings_df(
          self.ratings_df, train_fraction=0.5, val_fraction=1.25)

  def test_split_ratings_df_no_val(self):
    train_ratings_df, _, test_ratings_df = movielens_dataset.split_ratings_df(
        self.ratings_df, train_fraction=0.5, val_fraction=0.0)

    self.assertLen(train_ratings_df, 3)
    self.assertLen(test_ratings_df, 3)
    self.assertListEqual(list(train_ratings_df['UserID']), [0, 1, 1])
    self.assertListEqual(list(train_ratings_df['MovieID']), [0, 2, 2])
    self.assertListEqual(list(train_ratings_df['Rating']), [1.0, 3.0, 6.0])
    self.assertListEqual(list(train_ratings_df['Timestamp']), [1, 1, 3])
    self.assertListEqual(list(test_ratings_df['UserID']), [0, 1, 1])
    self.assertListEqual(list(test_ratings_df['MovieID']), [1, 3, 2])
    self.assertListEqual(list(test_ratings_df['Rating']), [2.0, 4.0, 5.0])
    self.assertListEqual(list(test_ratings_df['Timestamp']), [2, 4, 5])

  def test_split_ratings_df_fraction_floor(self):
    """Ensures edge-case behavior is as expected."""
    train_ratings_df, val_ratings_df, test_ratings_df = movielens_dataset.split_ratings_df(
        self.ratings_df, train_fraction=0.5, val_fraction=0.49)

    self.assertLen(train_ratings_df, 3)
    # 1 example from user 1 goes to train, the other goes to test, since
    # `val_fraction` upper bounds the number of examples provided for each user
    # and remaining examples go to the test set, i.e. floor(2 * 0.49) = 0 val
    # examples.
    self.assertLen(val_ratings_df, 1)
    self.assertLen(test_ratings_df, 2)
    self.assertListEqual(list(train_ratings_df['UserID']), [0, 1, 1])
    self.assertListEqual(list(train_ratings_df['MovieID']), [0, 2, 2])
    self.assertListEqual(list(train_ratings_df['Rating']), [1.0, 3.0, 6.0])
    self.assertListEqual(list(train_ratings_df['Timestamp']), [1, 1, 3])

    self.assertListEqual(list(val_ratings_df['UserID']), [1])
    self.assertListEqual(list(val_ratings_df['MovieID']), [3])
    self.assertListEqual(list(val_ratings_df['Rating']), [4.0])
    self.assertListEqual(list(val_ratings_df['Timestamp']), [4])

    self.assertListEqual(list(test_ratings_df['UserID']), [0, 1])
    self.assertListEqual(list(test_ratings_df['MovieID']), [1, 2])
    self.assertListEqual(list(test_ratings_df['Rating']), [2.0, 5.0])
    self.assertListEqual(list(test_ratings_df['Timestamp']), [2, 5])

  def test_get_user_examples(self):
    user_examples = movielens_dataset.get_user_examples(self.ratings_df, 0)

    self.assertCountEqual(user_examples, [(0, 0, 1.0), (0, 1, 2.0)])

  def test_get_user_examples_max_examples(self):
    user_examples = movielens_dataset.get_user_examples(
        self.ratings_df, 0, max_examples_per_user=1)
    # Ensure number of examples is now 1. The exact example may vary due to
    # shuffling.
    self.assertLen(user_examples, 1)

  def test_create_tf_dataset_for_user(self):
    tf_dataset = movielens_dataset.create_tf_dataset_for_user(
        self.ratings_df, 0, personal_model=True, batch_size=1)
    dataset_elements = list(tf_dataset.as_numpy_iterator())
    expected_elements = [
        collections.OrderedDict(
            x=np.array([0], dtype=np.int64),
            y=np.array([1.0], dtype=np.float32)),
        collections.OrderedDict(
            x=np.array([1], dtype=np.int64),
            y=np.array([2.0], dtype=np.float32))
    ]

    self.assertLen(dataset_elements, 2)
    self.assertCountEqual(dataset_elements, expected_elements)

  def test_create_tf_dataset_for_user_local_epochs(self):
    tf_dataset = movielens_dataset.create_tf_dataset_for_user(
        self.ratings_df,
        0,
        personal_model=True,
        batch_size=1,
        num_local_epochs=3)
    dataset_elements = list(tf_dataset.as_numpy_iterator())
    expected_elements = [
        collections.OrderedDict(
            x=np.array([0], dtype=np.int64),
            y=np.array([1.0], dtype=np.float32)),
        collections.OrderedDict(
            x=np.array([1], dtype=np.int64),
            y=np.array([2.0], dtype=np.float32)),
        collections.OrderedDict(
            x=np.array([0], dtype=np.int64),
            y=np.array([1.0], dtype=np.float32)),
        collections.OrderedDict(
            x=np.array([1], dtype=np.int64),
            y=np.array([2.0], dtype=np.float32)),
        collections.OrderedDict(
            x=np.array([0], dtype=np.int64),
            y=np.array([1.0], dtype=np.float32)),
        collections.OrderedDict(
            x=np.array([1], dtype=np.int64),
            y=np.array([2.0], dtype=np.float32)),
    ]

    self.assertLen(dataset_elements, 6)
    self.assertCountEqual(dataset_elements, expected_elements)

  def test_create_tf_dataset_for_user_batch_size(self):
    tf_dataset = movielens_dataset.create_tf_dataset_for_user(
        self.ratings_df, 0, personal_model=True, batch_size=3)
    dataset_elements = list(tf_dataset.as_numpy_iterator())
    expected_elements = [
        collections.OrderedDict(
            x=np.array([0, 1], dtype=np.int64),
            y=np.array([1.0, 2.0], dtype=np.float32)),
    ]

    self.assertLen(dataset_elements, 1)
    self.assertCountEqual(dataset_elements[0], expected_elements[0])

  def test_create_tf_dataset_for_user_non_personal_model(self):
    tf_dataset = movielens_dataset.create_tf_dataset_for_user(
        self.ratings_df, 0, personal_model=False, batch_size=1)
    dataset_elements = list(tf_dataset.as_numpy_iterator())
    expected_elements = [
        collections.OrderedDict(
            x=(np.array([0], dtype=np.int64), np.array([0], dtype=np.int64)),
            y=np.array([1.0], dtype=np.float32)),
        collections.OrderedDict(
            x=(np.array([0], dtype=np.int64), np.array([1], dtype=np.int64)),
            y=np.array([2.0], dtype=np.float32))
    ]

    self.assertLen(dataset_elements, 2)
    self.assertCountEqual(dataset_elements, expected_elements)

  def test_create_tf_dataset_for_user_max_examples_epochs(self):
    tf_dataset = movielens_dataset.create_tf_dataset_for_user(
        self.ratings_df,
        0,
        personal_model=True,
        batch_size=1,
        max_examples_per_user=1,
        num_local_epochs=2)
    dataset_elements = list(tf_dataset.as_numpy_iterator())

    # Ensure each epoch has 1 element. Exact element is random.
    self.assertLen(dataset_elements, 2)

  def test_create_tf_datasets(self):
    tf_datasets = movielens_dataset.create_tf_datasets(
        self.ratings_df,
        personal_model=True,
        batch_size=1,
        max_examples_per_user=None,
        num_local_epochs=2)
    user1_elements = list(tf_datasets[0].as_numpy_iterator())
    user2_elements = list(tf_datasets[1].as_numpy_iterator())

    expected_user1_elements = [
        collections.OrderedDict(
            x=np.array([0], dtype=np.int64),
            y=np.array([1.0], dtype=np.float32)),
        collections.OrderedDict(
            x=np.array([1], dtype=np.int64),
            y=np.array([2.0], dtype=np.float32)),
        collections.OrderedDict(
            x=np.array([0], dtype=np.int64),
            y=np.array([1.0], dtype=np.float32)),
        collections.OrderedDict(
            x=np.array([1], dtype=np.int64),
            y=np.array([2.0], dtype=np.float32)),
    ]
    expected_user2_elements = [
        collections.OrderedDict(
            x=np.array([2], dtype=np.int64),
            y=np.array([3.0], dtype=np.float32)),
        collections.OrderedDict(
            x=np.array([3], dtype=np.int64),
            y=np.array([4.0], dtype=np.float32)),
        collections.OrderedDict(
            x=np.array([2], dtype=np.int64),
            y=np.array([5.0], dtype=np.float32)),
        collections.OrderedDict(
            x=np.array([2], dtype=np.int64),
            y=np.array([6.0], dtype=np.float32)),
        collections.OrderedDict(
            x=np.array([2], dtype=np.int64),
            y=np.array([3.0], dtype=np.float32)),
        collections.OrderedDict(
            x=np.array([3], dtype=np.int64),
            y=np.array([4.0], dtype=np.float32)),
        collections.OrderedDict(
            x=np.array([2], dtype=np.int64),
            y=np.array([5.0], dtype=np.float32)),
        collections.OrderedDict(
            x=np.array([2], dtype=np.int64),
            y=np.array([6.0], dtype=np.float32)),
    ]

    self.assertLen(user1_elements, 4)
    self.assertLen(user2_elements, 8)
    self.assertCountEqual(user1_elements, expected_user1_elements)
    self.assertCountEqual(user2_elements, expected_user2_elements)

  def test_create_tf_datasets_batch_size_num_examples(self):
    tf_datasets = movielens_dataset.create_tf_datasets(
        self.ratings_df,
        personal_model=True,
        batch_size=2,
        max_examples_per_user=1)
    user1_elements = list(tf_datasets[0].as_numpy_iterator())
    user2_elements = list(tf_datasets[1].as_numpy_iterator())

    expected_user1_element = collections.OrderedDict(
        x=np.array([0, 1], dtype=np.int64),
        y=np.array([1.0, 2.0], dtype=np.float32))
    expected_user2_element = collections.OrderedDict(
        x=np.array([2, 3], dtype=np.int64),
        y=np.array([3.0, 4.0], dtype=np.float32))

    self.assertLen(user1_elements, 1)
    self.assertLen(user1_elements, 1)
    self.assertCountEqual(user1_elements[0], expected_user1_element)
    self.assertCountEqual(user2_elements[0], expected_user2_element)

  def test_create_tf_datasets_non_personal_model(self):
    tf_datasets = movielens_dataset.create_tf_datasets(
        self.ratings_df,
        personal_model=False,
        batch_size=2,
        max_examples_per_user=1,
        num_local_epochs=1)
    user1_elements = list(tf_datasets[0].as_numpy_iterator())
    user2_elements = list(tf_datasets[1].as_numpy_iterator())

    expected_user1_element = collections.OrderedDict(
        x=(np.array([0, 0], dtype=np.int64), np.array([0, 1], dtype=np.int64)),
        y=np.array([1.0, 2.0], dtype=np.float32))
    expected_user2_element = collections.OrderedDict(
        x=(np.array([1, 1], dtype=np.int64), np.array([2, 3], dtype=np.int64)),
        y=np.array([3.0, 4.0], dtype=np.float32))

    self.assertLen(user1_elements, 1)
    self.assertLen(user1_elements, 1)
    self.assertCountEqual(user1_elements[0], expected_user1_element)
    self.assertCountEqual(user2_elements[0], expected_user2_element)

  def test_split_tf_datasets(self):
    tf_datasets = [
        tf.data.Dataset.range(10),
        tf.data.Dataset.range(9),
        tf.data.Dataset.range(8),
        tf.data.Dataset.range(7),
    ]

    train_datasets, val_datasets, test_datasets = movielens_dataset.split_tf_datasets(
        tf_datasets, train_fraction=.5, val_fraction=.25)

    self.assertLen(train_datasets, 2)
    self.assertLen(val_datasets, 1)
    self.assertLen(test_datasets, 1)

  def test_split_tf_datasets_empty_val(self):
    tf_datasets = [
        tf.data.Dataset.range(10),
        tf.data.Dataset.range(9),
        tf.data.Dataset.range(8),
        tf.data.Dataset.range(7),
    ]

    train_datasets, val_datasets, test_datasets = movielens_dataset.split_tf_datasets(
        tf_datasets, train_fraction=.5, val_fraction=0.0)

    self.assertLen(train_datasets, 2)
    self.assertEmpty(val_datasets)
    self.assertLen(test_datasets, 2)

  def test_create_merged_np_arrays(self):
    users, movies, ratings = movielens_dataset.create_merged_np_arrays(
        self.ratings_df, max_examples_per_user=None, shuffle_across_users=False)

    self._verify_server_data_arrays(users, movies, ratings)
    self.assertAllEqual(np.shape(users), [6, 1])
    self.assertAllEqual(np.shape(movies), [6, 1])
    self.assertAllEqual(np.shape(ratings), [6, 1])

    zipped_merged_data = zip(users, movies, ratings)

    self.assertCountEqual(zipped_merged_data, [(0, 0, 1.0), (0, 1, 2.0),
                                               (1, 2, 3.0), (1, 3, 4.0),
                                               (1, 2, 5.0), (1, 2, 6.0)])

  def test_create_merged_np_arrays_max_examples_shuffle(self):
    users, movies, ratings = movielens_dataset.create_merged_np_arrays(
        self.ratings_df, max_examples_per_user=2, shuffle_across_users=True)

    self._verify_server_data_arrays(users, movies, ratings)
    self.assertAllEqual(np.shape(users), [4, 1])
    self.assertAllEqual(np.shape(movies), [4, 1])
    self.assertAllEqual(np.shape(ratings), [4, 1])

  def test_create_user_split_np_arrays(self):
    train_arrays, val_arrays, test_arrays = movielens_dataset.create_user_split_np_arrays(
        self.ratings_df,
        max_examples_per_user=None,
        train_fraction=0.5,
        val_fraction=0.25)

    for arrays in [train_arrays, val_arrays, test_arrays]:
      users, movies, ratings = arrays
      self._verify_server_data_arrays(users, movies, ratings)

    zipped_train_data = list(zip(*train_arrays))
    zipped_val_data = list(zip(*val_arrays))
    zipped_test_data = list(zip(*test_arrays))

    # Which user appears in train/test will depend on the random seed used for
    # user shuffling, but we fix that here.
    self.assertCountEqual(zipped_train_data, [(1, 2, 3.0), (1, 3, 4.0),
                                              (1, 2, 5.0), (1, 2, 6.0)])
    self.assertEmpty(zipped_val_data)
    self.assertCountEqual(zipped_test_data, [(0, 0, 1.0), (0, 1, 2.0)])

  def test_create_user_split_np_arrays_val_data(self):
    train_arrays, val_arrays, test_arrays = movielens_dataset.create_user_split_np_arrays(
        self.ratings_df,
        max_examples_per_user=None,
        train_fraction=0.5,
        val_fraction=0.5)

    for arrays in [train_arrays, val_arrays, test_arrays]:
      users, movies, ratings = arrays
      self._verify_server_data_arrays(users, movies, ratings)

    zipped_train_data = list(zip(*train_arrays))
    zipped_val_data = list(zip(*val_arrays))
    zipped_test_data = list(zip(*test_arrays))

    # Which user appears in train/test will depend on the random seed used for
    # user shuffling, but we fix that here.
    self.assertCountEqual(zipped_train_data, [(1, 2, 3.0), (1, 3, 4.0),
                                              (1, 2, 5.0), (1, 2, 6.0)])
    self.assertCountEqual(zipped_val_data, [(0, 0, 1.0), (0, 1, 2.0)])
    self.assertEmpty(list(zipped_test_data))

  def test_create_user_split_np_arrays_max_examples_per_user(self):
    train_arrays, val_arrays, test_arrays = movielens_dataset.create_user_split_np_arrays(
        self.ratings_df,
        max_examples_per_user=2,
        train_fraction=0.5,
        val_fraction=0.5)

    for arrays in [train_arrays, val_arrays, test_arrays]:
      users, movies, ratings = arrays
      self._verify_server_data_arrays(users, movies, ratings)

    zipped_train_data = list(zip(*train_arrays))
    zipped_val_data = list(zip(*val_arrays))
    zipped_test_data = list(zip(*test_arrays))

    self.assertLen(zipped_train_data, 2)
    self.assertLen(zipped_val_data, 2)
    self.assertEmpty(zipped_test_data)


if __name__ == '__main__':
  tf.test.main()
