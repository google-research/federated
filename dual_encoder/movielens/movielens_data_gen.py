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
"""Generate TF.Examples from the MovieLens data for dual encoder model.

More information about the dataset can be found here:
https://grouplens.org/datasets/movielens/
"""
import collections
import functools
import os
import random
from typing import Callable, Dict, List, Optional, Tuple

from absl import app
from absl import flags
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff

# MovieLens 1M constants.
_NUM_MOVIE_IDS = 3952
_NUM_USER_IDS = 6040
_RATINGS_FILE_NAME = "ratings.dat"
_LOCAL_DIR = "/tmp"
_OUTPUT_TRAINING_DATA_FILENAME = "train_movielens_1m.tfrecord"
_OUTPUT_VALIDATION_DATA_FILENAME = "val_movielens_1m.tfrecord"
_OUTPUT_TESTING_DATA_FILENAME = "test_movielens_1m.tfrecord"
_PAD_ID = 0
# Constant for generating dataset.
_SHUFFLE_BUFFER_SIZE = 100
_PREFETCH_BUFFER_SIZE = 1


FLAGS = flags.FLAGS

flags.DEFINE_string("movielens_data_dir", None,
                    "Path to the cns directory of movielens data.")
flags.DEFINE_string("output_dir", None,
                    "Path to the directory of output tfrecord file.")
flags.DEFINE_integer("min_timeline_length", 3,
                     "The minimum timeline length to construct examples."
                     "Timeline with length less than this number are filtered"
                     "out.")
flags.DEFINE_integer("max_context_length", 10,
                     "The maximum length of user context history. All the"
                     "contexts get padded to this length.")


def read_ratings(data_dir: str, tmp_dir: str = _LOCAL_DIR) -> pd.DataFrame:
  """Read movielens ratings data into dataframe."""
  if not tf.io.gfile.exists(os.path.join(tmp_dir, _RATINGS_FILE_NAME)):
    tf.io.gfile.copy(
        os.path.join(data_dir, _RATINGS_FILE_NAME),
        os.path.join(tmp_dir, _RATINGS_FILE_NAME))
  ratings_df = pd.read_csv(
      os.path.join(tmp_dir, _RATINGS_FILE_NAME),
      sep="::",
      names=["UserID", "MovieID", "Rating", "Timestamp"])
  ratings_df["Timestamp"] = ratings_df["Timestamp"].apply(int)
  return ratings_df


def split_ratings_df(
    ratings_df: pd.DataFrame,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
  """Split per-user rating DataFrame into train/val/test by user id.

  Args:
    ratings_df: Pandas DataFrame containing ratings data with ["UserID",
      "MovieID", "Rating", "Timestamp"] columns. The "UserID" and "MovieID" in
      [1, num_unique_ids].
    train_fraction: The approximate fraction of users be in the train set. The
      actual number will be rounded to the nearest integer.
    val_fraction: The approximate fraction of users be in the validation set.
      The actual number will be rounded to the nearest integer.

  Returns:
    train_ratings_df: Pandas DataFrame containing the train subset of
      ratings_df, produced using the procedure above.
    test_ratings_df: Pandas DataFrame containing the validation subset of
      ratings_df, produced using the procedure above.
    test_ratings_df: Pandas DataFrame containing the test subset of ratings_df,
        produced using the procedure above.
  """

  user_ids = sorted(pd.unique(ratings_df.UserID))

  # Splitting the ratings_df into train/val/test data frames.
  # Current setting assumes the data distribution doesn't vary by UserID
  # so that the train, validation and test datasets will have similar
  # distribution over the number of examples per user.
  # TODO(b/181596254): consistently randomly permute train/test/val mask
  # before using them to generate the train/val/test data frames.
  # Using a single permutation, otherwise there'll be some overlap.
  last_train_user_id = user_ids[round(len(user_ids) * train_fraction) - 1]
  last_val_user_id = user_ids[
      round(len(user_ids) * (train_fraction + val_fraction)) - 1]

  train_mask = ratings_df.UserID <= last_train_user_id
  test_mask = ratings_df.UserID > last_val_user_id
  val_mask = ~train_mask & ~test_mask

  train_ratings_df = ratings_df[train_mask]
  val_ratings_df = ratings_df[val_mask]
  test_ratings_df = ratings_df[test_mask]
  return train_ratings_df, val_ratings_df, test_ratings_df


def convert_to_timelines(ratings_df: pd.DataFrame) -> Dict[int, List[int]]:
  """Convert ratings_df to user timelines."""
  timelines = collections.defaultdict(list)
  # Sort per-user timeline by ascending timestamp
  sorted_ratings_df = ratings_df.groupby("UserID").apply(
      lambda x: x.sort_values("Timestamp"))
  for user_id in pd.unique(ratings_df.UserID):
    timelines[user_id] = sorted_ratings_df.loc[user_id].MovieID.to_list()
  return timelines


def generate_examples_from_a_single_timeline(
    timeline: List[int], max_context_len: int,
    pad_id: int) -> List[tf.train.Example]:
  """Convert a single user timeline to `tf.train.Example`s.

  Convert a single user timeline to `tf.train.Example`s by adding all possible
  context-label pairs in the examples pool.

  Args:
    timeline: The user timelines to process.
    max_context_len: The maximum length of context signals in an example. All
      the contexts get padded to this length.
    pad_id: The value being used to pad the context signals.

  Returns:
    examples: A `tf.train.Example` list.
  """

  examples = []
  for label_idx in range(1, len(timeline)):
    context_start_idx = max(0, label_idx - max_context_len)
    context = timeline[context_start_idx : label_idx]
    # Pad context with out-of-vocab movie id 0.
    context = context + [pad_id] * (max_context_len - len(context))
    label = timeline[label_idx]
    feature = {
        "context":
            tf.train.Feature(int64_list=tf.train.Int64List(value=context)),
        "label":
            tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
    }
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    examples.append(tf_example.SerializeToString())

  return examples


def generate_examples_from_timelines(
    timelines: Dict[int, List[int]],
    min_timeline_len: int = 3,
    max_context_len: int = 100,
    pad_id: int = _PAD_ID,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[List[tf.train.Example], List[tf.train.Example],
           List[tf.train.Example]]:
  """Convert user timelines to `tf.train.Example`s.

  Convert user timelines to `tf.train.Example`s by adding all possible
  context-label pairs in the examples pool. Splitting the examples into train,
  validation and test sets by the given train_fraction and val_fraction.

  Note that this function splits the examples by individual example instead of
  by user.

  Args:
    timelines: The user timelines to process.
    min_timeline_len: The minimum timeline length to construct examples.
      Timeline with length less than this number are filtered out.
    max_context_len: The maximum length of context signals in an example. All
      the contexts get padded to this length.
    pad_id: The value being used to pad the context signals.
    train_fraction: The approximate fraction of examples be in the train set.
      The actual number will be rounded to the nearest integer.
    val_fraction: The approximate fraction of examples be in the validation set.
      The actual number will be rounded to the nearest integer. Any remaining
      examples after generating the train and validation sets will go to the
      test set.
    seed: The seed used for initializing the random number generator.

  Returns:
    train_examples: A `tf.train.Example` list for training.
    val_examples: A `tf.train.Example` list for evaluation (validation).
    test_examples: A `tf.train.Example` list for testing.
  """

  examples = []
  for timeline in timelines.values():
    # Skip if timeline is shorter than min_timeline_len.
    if len(timeline) < min_timeline_len:
      continue
    examples += (
        generate_examples_from_a_single_timeline(
            timeline,
            max_context_len=max_context_len,
            pad_id=pad_id))

  # Split the examples into train, validation and test sets.
  random.seed(seed)
  random.shuffle(examples)
  last_train_index = round(len(examples) * train_fraction)
  last_val_index = round(len(examples) * (train_fraction + val_fraction))

  train_examples = examples[:last_train_index]
  val_examples = examples[last_train_index:last_val_index]
  test_examples = examples[last_val_index:]
  return train_examples, val_examples, test_examples


def write_tfrecords(tf_examples: List[tf.train.Example], filename: str):
  """Write `tf.train.Example`s to tfrecord file."""
  with tf.io.TFRecordWriter(filename) as file_writer:
    for example in tf_examples:
      file_writer.write(example)


def generate_examples_per_user(
    timelines: Dict[int, List[int]],
    min_timeline_len: int = 3,
    max_context_len: int = 100,
    pad_id: int = _PAD_ID,
    max_examples_per_user: Optional[int] = None
) -> Dict[int, List[tf.train.Example]]:
  """Convert user timelines to `tf.train.Example`s for each user.

  Args:
    timelines: The user timelines to process. It is a dictionary {user_id :
      timeline} as returned by `convert_to_timelines`.
    min_timeline_len: The minimum timeline length to construct examples.
      Timeline with length less than this number are filtered out.
    max_context_len: The maximum length of context signals in an example. All
      the contexts get padded to this length.
    pad_id: The value being used to pad the context signals.
    max_examples_per_user: If not None, it limit the maximum number of examples
      being generated for each user. If being set to 0 or None, it will also be
      ignored.

  Returns:
    examples_per_user: A dictionary {user_id : `tf.train.Example` list for
      user_id}.
  """
  examples_per_user = {}
  for user_id, timeline in timelines.items():
    # Skip if timeline is shorter than min_timeline_len.
    if len(timeline) < min_timeline_len:
      continue
    examples = (
        generate_examples_from_a_single_timeline(
            timeline,
            max_context_len=max_context_len,
            pad_id=pad_id))

    if max_examples_per_user is not None and max_examples_per_user > 0:
      examples = examples[:max_examples_per_user]

    examples_per_user[user_id] = examples
  return examples_per_user


def shuffle_examples_across_users(
    examples_per_user: Dict[int, List[tf.train.Example]],
    seed: Optional[int] = None) -> Dict[int, List[tf.train.Example]]:
  """Randomly shuffle the data across users.

    This function randomly shuffles the data across users while maintaining
    the number of examples distribution per user. It helps generate iid data
    distribution for the TFF simulation.

    Don't use this function unless testing the effect of iid vs non-iid data
    distribution in the federated learning.

  Args:
    examples_per_user: a dictionary {user_id : a `tf.train.Example` list for
      user_id}.
    seed: the seed used for initializing the random number generator.

  Returns:
    shuffled_examples_per_user: a dictionary with shuffled data across users
      {user_id : a `tf.train.Example` list for user_id}.
  """

  shuffled_examples_per_user = {}

  # Randomly shuffle all the 'tf.train.Example's.
  flattened_shuffled_examples = []
  for sublist in list(examples_per_user.values()):
    flattened_shuffled_examples.extend(sublist)

  if seed is not None:
    random.seed(seed)
  random.shuffle(flattened_shuffled_examples)

  # Split the shuffled examples into lists for the users.
  # For each user, the length of the example list remains the same as the
  # unshuffled case.
  curr_example_position = 0
  for user_id, examples in examples_per_user.items():
    num_examples = len(examples)
    shuffled_examples_per_user[user_id] = (
        flattened_shuffled_examples[curr_example_position : (
            curr_example_position + num_examples)])
    curr_example_position += num_examples

  assert curr_example_position == len(flattened_shuffled_examples)

  return shuffled_examples_per_user


def decode_example(
    serialized_proto: str,
    use_example_weight: bool = True) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
  """Decode single serialized example."""
  name_to_features = dict(
      context=tf.io.VarLenFeature(tf.int64),
      label=tf.io.FixedLenFeature([1], tf.int64))
  examples = tf.io.parse_single_example(serialized_proto, name_to_features)
  features = collections.OrderedDict()
  for name in examples:
    feature_content = examples[name]
    if feature_content.dtype == tf.int64:
      tf.cast(feature_content, tf.int32)
    if isinstance(feature_content, tf.SparseTensor):
      feature_content = tf.sparse.to_dense(feature_content)
    features[name] = feature_content

  if use_example_weight:
    # The returned example is in the format of ({'context': a list of movie IDs,
    # 'label': next movie ID}, example weight). Using 1.0 as the weight here.
    output = (features, tf.constant(1.0))
  else:
    # If using global similarity and global recall, return (features,
    # features['label']) instead.
    output = (features, features["label"])
  return output


def create_tf_datasets(
    examples_per_user: Dict[int, List[tf.train.Example]],
    batch_size: int = 1,
    num_local_epochs: int = 1,
    use_example_weight: bool = True) -> List[tf.data.Dataset]:
  """Create TF Datasets containing per user movie id timeline examples.

  Args:
    examples_per_user: A dictionary {user_id : a `tf.train.Example` list for
      user_id}.
    batch_size: The number of `tf.train.Example`s in a batch in the output
      `tf.data.Dataset`.
    num_local_epochs: Repeat the dataset the given number of times, effectively
      simulating multiple local epochs on the client.
    use_example_weight: If True, the format of the generated example is
      (features, example weight). Otherwise, the example format is (features,
      label). Here, features is {'context', 'label'}.

  Returns:
    datasets_per_user: A list of `tf.data.Dataset`s. The ith element in the
    list is the `tf.data.Dataset` for the (i+1)th user.
  """
  datasets_per_user = []
  for examples in examples_per_user.values():
    d = tf.data.Dataset.from_tensor_slices(examples)
    d = d.shuffle(_SHUFFLE_BUFFER_SIZE)
    d = d.map(
        functools.partial(
            decode_example,
            use_example_weight=use_example_weight),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    d = d.repeat(num_local_epochs)
    d = d.batch(batch_size, drop_remainder=True)
    d = d.prefetch(_PREFETCH_BUFFER_SIZE)
    datasets_per_user.append(d)
  return datasets_per_user


def create_client_datasets(
    *,  # Callers pass below args by name.
    ratings_df: pd.DataFrame,
    min_timeline_len: int = 3,
    max_context_len: int = 100,
    max_examples_per_user: Optional[int] = None,
    pad_id: int = _PAD_ID,
    shuffle_across_users: bool = False,
    batch_size: int = 1,
    num_local_epochs: int = 1,
    use_example_weight: bool = True) -> List[tf.data.Dataset]:
  """Create TF Datasets containing per user movie id examples from ratings_df.

  Args:
    ratings_df: Pandas DataFrame containing ratings data with ["UserID",
      "MovieID", "Rating", "Timestamp"] columns. The "UserID" and "MovieID"
      values are in the range [1, num_unique_ids].
    min_timeline_len: The minimum timeline length to construct examples.
      Timeline with length less than this number are filtered out.
    max_context_len: The maximum length of context signals in an example. All
      the contexts get padded to this length.
    max_examples_per_user: If not None, it limit the maximum number of examples
      being generated for each user. If being set to 0 or None, it will also be
      ignored.
    pad_id: The value being used to pad the context signals.
    shuffle_across_users: If it is true, calling `shuffle_examples_across_users`
      and randomly shuffle the data across users.
    batch_size: The number of `tf.train.Example`s in a batch in the output
      `tf.data.Dataset`.
    num_local_epochs: Repeat the dataset the given number of times, effectively
      simulating multiple local epochs on the client.
    use_example_weight: If True, the format of the generated example is
      (features, example weight). Otherwise, the example format is (features,
      label). Here, features is {'context', 'label'}.

  Returns:
    datasets: A list of `tf.data.Dataset`s. The ith element in the list
      is the `tf.data.Dataset` for the (i+1)th user.
  """

  timelines = convert_to_timelines(ratings_df)
  examples = generate_examples_per_user(
      timelines=timelines,
      min_timeline_len=min_timeline_len,
      max_context_len=max_context_len,
      pad_id=pad_id,
      max_examples_per_user=max_examples_per_user)
  if shuffle_across_users:
    examples = shuffle_examples_across_users(examples)
  datasets = create_tf_datasets(examples_per_user=examples,
                                batch_size=batch_size,
                                num_local_epochs=num_local_epochs,
                                use_example_weight=use_example_weight)
  return datasets


def client_dataset_preprocess_fn(
    batch_size: int = 1,
    num_local_epochs: int = 1,
    use_example_weight: bool = True
) -> Callable[[tf.data.Dataset], tf.data.Dataset]:
  """A preprocess function applied to each client's data.

  Given a `tff.simulation.datasets.ClientData` client_data, an example usage is
  `client_data.preprocess(preprocess_fn=client_dataset_preprocess_fn)`.

  Args:
    batch_size: The number of `tf.train.Example`s in a batch in the output
      `tf.data.Dataset`.
    num_local_epochs: Repeat the dataset the given number of times, effectively
      simulating multiple local epochs on the client.
    use_example_weight: If True, the format of the generated example is
      (features, example weight). Otherwise, the example format is (features,
      label). Here, features is {'context', 'label'}.

  Returns:
    A callable performing the preprocessing described above.
  """
  if num_local_epochs < 1:
    raise ValueError("num_epochs must be a positive integer.")

  def preprocess_fn(dataset):
    d = dataset.shuffle(_SHUFFLE_BUFFER_SIZE)
    d = d.map(
        functools.partial(
            decode_example, use_example_weight=use_example_weight),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    d = d.repeat(num_local_epochs)
    d = d.batch(batch_size, drop_remainder=True)
    d = d.prefetch(_PREFETCH_BUFFER_SIZE)
    return d

  return preprocess_fn


def build_client_data_from_examples_per_user_dict(
    examples_per_user: Dict[int, List[tf.train.Example]]
) -> tff.simulation.datasets.ClientData:
  tensor_slices_dict = dict()
  for user_id, examples in examples_per_user.items():
    tensor_slices_dict[str(user_id)] = examples
  return tff.simulation.datasets.TestClientData(tensor_slices_dict)


def build_client_data(
    *,  # Callers pass below args by name.
    ratings_df: pd.DataFrame,
    min_timeline_len: int = 3,
    max_context_len: int = 100,
    max_examples_per_user: Optional[int] = None,
    pad_id: int = _PAD_ID,
    shuffle_across_users: bool = False) -> tff.simulation.datasets.ClientData:
  """Create a `tff.simulation.datasets.ClientData` for federated simulation.

  Args:
    ratings_df: Pandas DataFrame containing ratings data with ["UserID",
      "MovieID", "Rating", "Timestamp"] columns. The "UserID" and "MovieID"
      values are in the range [1, num_unique_ids].
    min_timeline_len: The minimum timeline length to construct examples.
      Timeline with length less than this number are filtered out.
    max_context_len: The maximum length of context signals in an example. All
      the contexts get padded to this length.
    max_examples_per_user: If not None, it limit the maximum number of examples
      being generated for each user. If being set to 0 or None, it will also be
      ignored.
    pad_id: The value being used to pad the context signals.
    shuffle_across_users: If it is true, calling `shuffle_examples_across_users`
      and randomly shuffle the data across users.

  Returns:
    client_data: A `tff.simulation.datasets.ClientData` for simulation.
  """

  timelines = convert_to_timelines(ratings_df)
  examples = generate_examples_per_user(
      timelines=timelines,
      min_timeline_len=min_timeline_len,
      max_context_len=max_context_len,
      pad_id=pad_id,
      max_examples_per_user=max_examples_per_user)
  if shuffle_across_users:
    examples = shuffle_examples_across_users(examples)
  client_data = build_client_data_from_examples_per_user_dict(examples)

  return client_data


def main(_):
  ratings_df = read_ratings(data_dir=FLAGS.movielens_data_dir)
  timelines = convert_to_timelines(ratings_df)
  train_examples, val_examples, test_examples = (
      generate_examples_from_timelines(
          timelines=timelines,
          min_timeline_len=FLAGS.min_timeline_length,
          max_context_len=FLAGS.max_context_length))

  if not tf.io.gfile.exists(FLAGS.output_dir):
    tf.io.gfile.makedirs(FLAGS.output_dir)
  write_tfrecords(
      tf_examples=train_examples,
      filename=os.path.join(FLAGS.output_dir, _OUTPUT_TRAINING_DATA_FILENAME))
  write_tfrecords(
      tf_examples=val_examples,
      filename=os.path.join(FLAGS.output_dir, _OUTPUT_VALIDATION_DATA_FILENAME))
  write_tfrecords(
      tf_examples=test_examples,
      filename=os.path.join(FLAGS.output_dir, _OUTPUT_TESTING_DATA_FILENAME))


if __name__ == "__main__":
  flags.mark_flag_as_required("movielens_data_dir")
  flags.mark_flag_as_required("output_dir")
  app.run(main)
