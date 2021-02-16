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
"""Load and pre-process the MovieLens user-movie ratings dataset.

More information about the dataset can be found here:
https://grouplens.org/datasets/movielens/
"""

import collections
import io
import os
from typing import Any, List, Optional, Tuple
import zipfile

import numpy as np
import pandas as pd
import requests

import tensorflow as tf

# Permalink to stable .zip file where user, rating, and movie DataFrames for
# MovieLens 1M are stored.
MOVIELENS_1M_URL = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"

DEFAULT_DATA_DIRECTORY = "/tmp"

# To ensure data shuffling is reproducible.
NP_RANDOM_SEED = 42

ServerDataArray = Tuple[np.ndarray, np.ndarray, np.ndarray]


def download_and_extract_data(url: str = MOVIELENS_1M_URL,
                              data_directory: str = DEFAULT_DATA_DIRECTORY):
  """Downloads and extracts zip containing MovieLens data to a given directory.

  Args:
    url: Direct path to MovieLens dataset .zip file. See constants above for
      examples.
    data_directory: Local path to extract dataset to.
  """
  r = requests.get(url)
  z = zipfile.ZipFile(io.BytesIO(r.content))
  z.extractall(path=data_directory)


def load_movielens_data(
    data_directory: str = DEFAULT_DATA_DIRECTORY,
    normalize_ratings: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Loads pandas DataFrames for ratings, movies, users from data directory.

  Assumes data is formatted as specified in
  http://files.grouplens.org/datasets/movielens/ml-1m-README.txt. To enable
  easy embedding lookup, UserID and MovieID fields across DataFrames are
  consistently remapped to be within [0, num_unique_ids), separately for each
  field. Movies and users without ratings are dropped. Ratings in 1-5 are
  optionally linearly transformed to be in {-1, -0.5, 0, 0.5, 1}.

  Args:
    data_directory: Local path containing MovieLens dataset files.
    normalize_ratings: Whether to normalize ratings in 1-5 to be in {-1, -0.5,
      0, 0.5, 1} via a linear scaling.

  Returns:
    ratings_df: pandas DataFrame containing ratings data with
        ["UserID", "MovieID", "Rating", "Timestamp"] columns. UserID and MovieID
        are remapped to ensure values are in [0, num_unique_ids). The remapping
        is consistent with the remapping for other returned DataFrames.
    movies_df: pandas DataFrame containing movie data with
        ["MovieID", "Title", "Genres"] columns. MovieID is remapped to ensure
        values are in [0, num_unique_ids). The remapping is consistent with the
        remapping for other returned DataFrames.
    users_df: pandas DataFrame containing user data with
        ["UserID", "Gender", "Age", "Occupation", "Zip-code"] columns. UserID is
        remapped to ensure values are in [0, num_unique_ids). The remapping is
        consistent with the remapping for other returned DataFrames.
  """
  # Load pandas DataFrames from data directory. Assuming data is formatted as
  # specified in http://files.grouplens.org/datasets/movielens/ml-1m-README.txt.
  ratings_df = pd.read_csv(
      os.path.join(data_directory, "ml-1m", "ratings.dat"),
      sep="::",
      names=["UserID", "MovieID", "Rating", "Timestamp"])
  movies_df = pd.read_csv(
      os.path.join(data_directory, "ml-1m", "movies.dat"),
      sep="::",
      names=["MovieID", "Title", "Genres"])
  users_df = pd.read_csv(
      os.path.join(data_directory, "ml-1m", "users.dat"),
      sep="::",
      names=["UserID", "Gender", "Age", "Occupation", "Zip-code"])

  # Create dictionaries mapping from old IDs to new (remapped) IDs for both
  # MovieID and UserID. Use the movies and users present in ratings_df to
  # determine the mapping, since movies and users without ratings are unneeded.
  movie_mapping = {
      old_movie: new_movie for new_movie, old_movie in enumerate(
          ratings_df.MovieID.astype("category").cat.categories)
  }
  user_mapping = {
      old_user: new_user for new_user, old_user in enumerate(
          ratings_df.UserID.astype("category").cat.categories)
  }

  # Map each DataFrame consistently using the now-fixed mapping.
  ratings_df.MovieID = ratings_df.MovieID.map(movie_mapping)
  ratings_df.UserID = ratings_df.UserID.map(user_mapping)

  movies_df.MovieID = movies_df.MovieID.map(movie_mapping)
  users_df.UserID = users_df.UserID.map(user_mapping)

  # Remove NaNs resulting from some movies or users being in movies_df or
  # users_df but not ratings_df. This effectively drops movies and users that
  # do not have ratings.
  movies_df = movies_df[pd.notnull(movies_df.MovieID)]
  users_df = users_df[pd.notnull(users_df.UserID)]

  # Optionally linearly normalize 1-5 ratings to be in [-1, 1].
  if normalize_ratings:
    ratings_df.Rating = (ratings_df.Rating - 3) / 2

  return ratings_df, movies_df, users_df


def split_ratings_df(
    ratings_df: pd.DataFrame,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Splits per-user ratings DataFrame into train/val/test by timestamp.

  Split ratings into train/val/test data while achieving the following:

  (1) The set of users in train and val/test is the same, i.e., we are not
      evaluating on users for whom we have no data.
  (2) For each user, about train_fraction of the ratings are in the training
      set, i.e., we have enough data for each user to make predictions.
  (3) For each user, timestamps of any data in the val/test set are at least as
      large as the largest timestamp in the train set, i.e., we are predicting
      future ratings.

  This function effectively does the following: for each user, find a timestamp
  such that train_fraction of the ratings for that user have an earlier or equal
  timestamp. Then for each user, any examples with earlier or equal timestamps
  are put in the training set, and others are put in the val/test sets. Between
  val/test, examples are split similarly, with test examples having the larger
  timestamps.

  Note that splitting by users does not achieve (1), (2), and (3) making
  meaningful per-user predictions on the test set impossible. Splitting by a
  single global timestamp does not achieve (1) and (2). Also note that a
  standard train/test split for this dataset does not exist.

  If splitting by users is desired, see `split_tf_datasets`. This can be
  expected to produce meaningful results when using federated reconstruction,
  which alleviates the need for evaluating on previously seen users.

  Args:
    ratings_df: pandas DataFrame containing at least the "UserID" and
      "Timestamp" fields, as returned by load_movielens_data.
    train_fraction: for each user, the approximate fraction of ratings that will
      be in the train set. The actual fraction will be upper bounded by this, so
      the number of train examples for each user is floor(num_user_examples *
      train_fraction).
    val_fraction: for each user, the approximate fraction of ratings that will
      be in the val set. The actual fraction will be upper bounded by this, so
      the number of val examples for each user is floor(num_user_examples *
      val_fraction).

  Returns:
    train_ratings_df: pandas DataFrame containing the train subset of
      ratings_df, produced using the procedure above.
    val_ratings_df: pandas DataFrame containing the val subset of ratings_df,
      produced using the procedure above.
    test_ratings_df: pandas DataFrame containing the test subset of ratings_df,
      produced using the procedure above.
  """
  if train_fraction + val_fraction > 1:
    raise ValueError(
        "train_fraction and val_fraction can't sum to greater than 1, got {}"
        "and {}.".format(train_fraction, val_fraction))

  # For each rating, calculate its rank among other ratings for that user,
  # ordered by increasing timestamp. Then normalize by counts per user to get
  # a value in (0, 1], which can be thresholded to produce example masks.
  ranks_per_user = ratings_df.groupby("UserID")["Timestamp"].rank(
      method="first")
  counts_per_user = ratings_df["UserID"].map(
      ratings_df.groupby("UserID")["Timestamp"].apply(len))
  normalized_ranks_per_user = ranks_per_user / counts_per_user

  # The first `train_fraction` belong to the training set, the next
  # `val_fraction` belongs to the val set. The rest belongs to the test set.
  train_mask = normalized_ranks_per_user <= train_fraction
  val_mask = ((normalized_ranks_per_user <=
               (train_fraction + val_fraction)) & ~train_mask)
  test_mask = ~train_mask & ~val_mask

  train_ratings_df = ratings_df[train_mask]
  val_ratings_df = ratings_df[val_mask]
  test_ratings_df = ratings_df[test_mask]
  return train_ratings_df, val_ratings_df, test_ratings_df


def get_user_examples(ratings_df: pd.DataFrame,
                      user_id: int,
                      max_examples_per_user: Optional[int] = None) -> List[Any]:
  """Gets a user's rating examples, up to a maximum.

  Args:
    ratings_df: a pandas DataFrame with ratings data, as returned by
      load_movielens_data or split_ratings_df (if using train/test split data is
      desired).
    user_id: the ID for a particular user. This ID matches the UserID field in
      ratings_df and users_df as returned by load_movielens_data (i.e. this ID
      is the mapped ID after pre-processing done in that function).
    max_examples_per_user: if not None, limit the number of rating examples for
      this user to this many examples.

  Returns:
    A List of example triples, where example[0] is the UserID, example[1] is the
        MovieID, and example[2] is the rating.
  """
  # Get subset of ratings_df belonging to a particular user.
  user_subset = ratings_df[ratings_df.UserID == user_id]
  user_examples = [(user_subset.UserID.iloc[i], user_subset.MovieID.iloc[i],
                    user_subset.Rating.iloc[i])
                   for i in range(user_subset.shape[0])]
  np.random.seed(NP_RANDOM_SEED)
  np.random.shuffle(user_examples)

  # Optionally filter number of examples per user, taking the first
  # max_examples_per_user examples.
  if max_examples_per_user is not None:
    user_examples = user_examples[:max_examples_per_user]

  return user_examples


def create_tf_dataset_for_user(ratings_df: pd.DataFrame,
                               user_id: int,
                               personal_model: bool = False,
                               batch_size: int = 1,
                               max_examples_per_user: Optional[int] = None,
                               num_local_epochs: int = 1) -> tf.data.Dataset:
  """Creates a TF Dataset containing the movies and ratings for a given user.

  Takes a ratings_df, as given by split_ratings_df, and a user_id and returns
  a `tf.data.Dataset` containing examples for the given user. Optionally filters
  to a given number of max examples per user. Optionally repeats the dataset,
  effectively enabling training
  for multiple local epochs, which was shown to significantly speed up training
  in https://arxiv.org/abs/1602.05629.

  Args:
    ratings_df: a pandas DataFrame with ratings data, as returned by
      load_movielens_data or split_ratings_df (if using train/test split data is
      desired).
    user_id: the ID for a particular user. This ID matches the UserID field in
      ratings_df and users_df as returned by load_movielens_data (i.e., this ID
      is the mapped ID after pre-processing done in that function).
    personal_model: If True, the dataset contains only a user's own data and the
      model only expects item IDs as input. If False, the dataset contains data
      for all users, and user IDs are produced as input along with item IDs.
      This should be set to True for experiments with user stateless federated
      averaging and False for experiments with server-side data.
    batch_size: the number of rating examples in a batch in the output
      `tf.data.Dataset`.
    max_examples_per_user: if not None, limit the number of rating examples for
      this user to this many examples.
    num_local_epochs: repeat the dataset the given number of times, effectively
      simulating multiple local epochs on the client. No-op if num_local_epochs
      is 1.

  Returns:
    A batched `tf.data.Dataset` containing examples in the format
        {x: (user_id, movie_id), y: rating}, with types
        {x: (tf.int64, tf.int64), y: tf.float32} if personal_model is False,
        and
        {x: movie_id, y: rating}, with types
        {x: tf.int64, y: tf.float32}
        if personal_model is True.
  """

  def rating_batch_map_fn(rating_batch):
    """Maps a rating batch to an OrderedDict with tensor values."""
    # Each tensor has final shape (None, 1).
    if personal_model:
      return collections.OrderedDict([
          ("x", tf.cast(rating_batch[:, 1:2], tf.int64)),
          ("y", tf.cast(rating_batch[:, 2:3], tf.float32))
      ])
    return collections.OrderedDict([
        ("x", (tf.cast(rating_batch[:, 0:1],
                       tf.int64), tf.cast(rating_batch[:, 1:2], tf.int64))),
        ("y", tf.cast(rating_batch[:, 2:3], tf.float32))
    ])

  user_examples = get_user_examples(ratings_df, user_id, max_examples_per_user)
  tf_dataset = tf.data.Dataset.from_tensor_slices(user_examples)

  # Apply batching before repeat to ensure batching does not straddle epoch
  # boundaries.
  return tf_dataset.batch(batch_size).map(
      rating_batch_map_fn,
      num_parallel_calls=tf.data.experimental.AUTOTUNE).repeat(num_local_epochs)


def create_tf_datasets(ratings_df: pd.DataFrame,
                       personal_model: bool = False,
                       batch_size: int = 1,
                       max_examples_per_user: Optional[int] = None,
                       num_local_epochs: int = 1) -> List[tf.data.Dataset]:
  """Creates TF Datasets containing the movies and ratings for all users.

  Performs create_tf_dataset_for_user for all users.

  Args:
    ratings_df: a pandas DataFrame with ratings data, as returned by
      load_movielens_data or split_ratings_df (if using train/test split data is
      desired).
    personal_model: If True, the dataset contains only a user's own data and the
      model only expects item IDs as input. If False, the dataset contains data
      for all users, and user IDs are expected as input along with item IDs.
      This should be set to True for experiments with user stateless federated
      averaging and False for experiments with server-side data.
    batch_size: the number of rating examples in a batch in the output
      `tf.data.Dataset`.
    max_examples_per_user: if not None, limit the number of rating examples for
      each user to this many examples.
    num_local_epochs: repeat the dataset the given number of times, effectively
      simulating multiple local epochs on the client. No-op if num_local_epochs
      is 1.

  Returns:
    A list of batched `tf.data.Dataset` containing examples in the format
        {x: (user_id, movie_id), y: rating}, with types
        {x: (tf.int64, tf.int64), y: tf.float32} if personal_model is False,
        and
        {x: movie_id, y: rating}, with types
        {x: tf.int64, y: tf.float32}
        if personal_model is True.
  """
  num_users = len(set(ratings_df.UserID))
  tf_datasets = [
      create_tf_dataset_for_user(ratings_df, i, personal_model, batch_size,
                                 max_examples_per_user, num_local_epochs)
      for i in range(num_users)
  ]
  return tf_datasets


def split_tf_datasets(
    tf_datasets: List[tf.data.Dataset],
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
) -> Tuple[List[tf.data.Dataset], List[tf.data.Dataset], List[tf.data.Dataset]]:
  """Splits list of user TF datasets into train/val/test by user.

  Can be used instead of `split_ratings_df` to split MovieLens data by user.
  Shuffles list of datasets before splitting. Note that this can produce
  meaningful results when using federated reconstruction, since this alleviates
  the requirement for testing on users we have seen before.

  Args:
    tf_datasets: a list of `tf.data.Dataset`s, one for each user.
    train_fraction: the approximate fraction of users to allocate to training
      data.
    val_fraction: the approximate fraction of users to allocate to val data.
      Remaining users not allocated to train and val (if any) are allocated to
      test.

  Returns:
    A triple of train, val, and test TF datasets.
  """
  if train_fraction + val_fraction > 1:
    raise ValueError(
        "train_fraction and val_fraction can't sum to greater than 1, got {}"
        "and {}.".format(train_fraction, val_fraction))

  np.random.seed(NP_RANDOM_SEED)
  np.random.shuffle(tf_datasets)

  train_idx = int(len(tf_datasets) * train_fraction)
  val_idx = int(len(tf_datasets) * (train_fraction + val_fraction))

  return (tf_datasets[:train_idx], tf_datasets[train_idx:val_idx],
          tf_datasets[val_idx:])


def create_merged_np_arrays(
    ratings_df: pd.DataFrame,
    max_examples_per_user: Optional[int] = None,
    shuffle_across_users: bool = True) -> ServerDataArray:
  """Creates arrays for train/val/test user data for server-side evaluation.

  Loads a server-side version of the MovieLens dataset that contains ratings
  from all users, with examples optionally shuffled across users. Note that
  unlike `create_tf_datasets` and `create_tf_dataset_for_user`, the output data
  does not generate batches (batching can be later applied during the call to
  `model.fit`).

  Produces (merged_users, merged_movies, merged_ratings), which can be used for
  training of a model produced by `get_matrix_factorization_model()` from
  models.py by calling
  `model.fit([merged_users, merged_movies], merged_ratings, ...)`.

  Differs from `create_user_split_np_arrays` in that this performs no splitting
  of data, whereas `create_user_split_np_arrays` splits such that train/val/test
  contain disjoint users.

  Args:
    ratings_df: a pandas DataFrame with ratings data, as returned by
      `load_movielens_data` or `split_ratings_df` (if using train/test split
      data within users is desired).
    max_examples_per_user: if not None, limit the number of rating examples for
      each user to this many examples.
    shuffle_across_users: if True, shuffle examples between different users in
      the output TF Dataset. If false, examples are ordered sequentially by
      UserID. Note that examples for a given user are shuffled regardless.

  Returns:
    merged_users: a np.ndarray with type np.int64 and shape (num_examples, 1)
        containing UserIDs. The order of entries corresponds to the order of the
        examples: merged_users[i] is the UserID for example i.
    merged_movies: a np.ndarray with type np.int64 and shape (num_examples, 1)
        containing MovieIDs. The order of entries corresponds to the order of
        the examples: merged_movies[i] is the MovieID for example i.
    merged_ratings: a np.ndarray with type np.float32 and shape
        (num_examples, 1) containing Ratings. The order of entries corresponds
        to the order of the examples: merged_ratings[i] is the Rating for
        example i.
  """
  num_users = len(set(ratings_df.UserID))

  merged_examples = []
  for user_id in range(num_users):
    user_examples = get_user_examples(ratings_df, user_id,
                                      max_examples_per_user)
    merged_examples += user_examples

  if shuffle_across_users:
    np.random.seed(NP_RANDOM_SEED)
    np.random.shuffle(merged_examples)

  # Produce a list of sublists of size 1, so that the output np.ndarrays have
  # shape (num_examples, 1), as expected by the model.
  merged_users = np.array([[x[0]] for x in merged_examples], dtype=np.int64)
  merged_movies = np.array([[x[1]] for x in merged_examples], dtype=np.int64)
  merged_ratings = np.array([[x[2]] for x in merged_examples], dtype=np.float32)

  return merged_users, merged_movies, merged_ratings


def create_user_split_np_arrays(
    ratings_df: pd.DataFrame,
    max_examples_per_user: Optional[int] = None,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
) -> Tuple[ServerDataArray, ServerDataArray, ServerDataArray]:
  """Creates arrays for train/val/test user data for server-side evaluation.

  Loads a server-side version of the MovieLens dataset that contains ratings
  from users partitioned into train/val/test populations. Note that
  unlike `create_tf_datasets` and `create_tf_dataset_for_user`, the output data
  does not generate batches (batching can be later applied during the call to
  `model.fit`).

  This produces datasets for user with server-side Keras where the split is
  by users, so train/val/test sets contain disjoint users. For standard
  server evaluation, this is expected to perform less well since the user
  embeddings for users seen at val/test time will not be trained. If splitting
  within users (so each user's data is split into train/val/test) is desired,
  see `split_ratings_df` and `create_merged_np_arrays`.

  For each of train/val/test, produces
  (merged_users, merged_movies, merged_ratings), which can be used for
  training of a model produced by `get_matrix_factorization_model()` from
  models.py by calling
  `model.fit([merged_users, merged_movies], merged_ratings, ...)`.

  Args:
    ratings_df: a pandas DataFrame with ratings data, as returned by
      `load_movielens_data` or `split_ratings_df` (if using train/test split
      data is desired).
    max_examples_per_user: if not None, limit the number of rating examples for
      each user to this many examples.
    train_fraction: the approximate fraction of users to allocate to training
      data.
    val_fraction: the approximate fraction of users to allocate to val data.
      Remaining users not allocated to train and val (if any) are allocated to
      test.

  Returns:
    A 3-tuple of 3-tuples:
      ((train_users, train_movies, train_ratings),
       (val_users, val_movies, val_ratings),
       (test_users, test_movies, test_ratings)).

    For each of train/val/test, the tuple contains:
      users: a np.ndarray with type np.int64 and shape (num_examples, 1)
          containing UserIDs. The order of entries corresponds to the order of
          the examples: users[i] is the UserID for example i.
      movies: a np.ndarray with type np.int64 and shape (num_examples, 1)
          containing MovieIDs. The order of entries corresponds to the order of
          the examples: movies[i] is the MovieID for example i.
      ratings: a np.ndarray with type np.float32 and shape
          (num_examples, 1) containing Ratings. The order of entries corresponds
          to the order of the examples: ratings[i] is the Rating for
          example i.
  """
  num_users = len(set(ratings_df.UserID))

  all_user_examples = []
  for user_id in range(num_users):
    all_user_examples.append(
        get_user_examples(ratings_df, user_id, max_examples_per_user))

  np.random.seed(NP_RANDOM_SEED)
  np.random.shuffle(all_user_examples)

  train_idx = int(len(all_user_examples) * train_fraction)
  val_idx = int(len(all_user_examples) * (train_fraction + val_fraction))

  # Each of these is a list of lists of examples per user.
  train_user_examples = all_user_examples[:train_idx]
  val_user_examples = all_user_examples[train_idx:val_idx]
  test_user_examples = all_user_examples[val_idx:]

  def get_users_movies_ratings(user_examples):
    """Helper for getting users/movies/ratings for each of train/val/test."""
    users = []
    movies = []
    ratings = []
    for user in user_examples:
      for example in user:
        users.append(example[0])
        movies.append(example[1])
        ratings.append(example[2])

    users = np.array(users, dtype=np.int64)
    movies = np.array(movies, dtype=np.int64)
    ratings = np.array(ratings, dtype=np.float32)

    # The output np.ndarrays have shape (num_examples, 1), as expected by the
    # model.
    users = np.reshape(users, [np.shape(users)[0], 1])
    movies = np.reshape(movies, [np.shape(movies)[0], 1])
    ratings = np.reshape(ratings, [np.shape(ratings)[0], 1])
    return (users, movies, ratings)

  train_users_movies_ratings = get_users_movies_ratings(train_user_examples)
  val_users_movies_ratings = get_users_movies_ratings(val_user_examples)
  test_users_movies_ratings = get_users_movies_ratings(test_user_examples)

  return (train_users_movies_ratings, val_users_movies_ratings,
          test_users_movies_ratings)
