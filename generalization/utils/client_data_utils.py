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
"""Utils functions for basic operations with `tff.simulation.datasets.ClientData`."""

import collections
from collections import abc
import random
from typing import Any, List, Mapping, Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

ClientData = tff.simulation.datasets.ClientData

_INTERLEAVE_BLOCK_LENGTH = 16


def convert_list_of_elems_to_tensor_slices(
    list_of_elems: List[Mapping[str, tf.Tensor]]) -> Mapping[str, tf.Tensor]:
  """Convert a list of elements (of type mapping) to tensor slices.

  For any `tf.data.Dataset` of element_type OrderedDict[str, tf.Tensor], this
  function guarantees
  `tf.data.Dataset.from_tensor_slices(convert_list_of_elems_to_tensor_slices(list(dataset)))`
  has the same elements as the original `dataset`.

  Args:
    list_of_elems: A list of elements of type `Mapping`.

  Returns:
    An OrderedDict of tensor slices.

  """
  tensor_slices = collections.OrderedDict()

  for key in list_of_elems[0]:
    tensor_slices[key] = tf.stack([elem[key] for elem in list_of_elems])

  return tensor_slices


def subsample_list_by_proportion(input_list: List[Any],
                                 proportion: float,
                                 seed: Optional[int] = None) -> List[Any]:
  """Subsample a list based on proportion in [0.0, 1.0].

  Args:
    input_list: The input list to subsample.
    proportion: A floating number indicating the proportion.
    seed: An optional random seed.

  Returns:
    A subsampled list of length round(len(input_list) * proportion).
  """
  if proportion is None or proportion < 0.0 or proportion > 1.0:
    raise ValueError(
        f'proportion must be a floating number in [0.0, 1.0], get {proportion}')

  elements_to_sample = round(len(input_list) * proportion)
  return random.Random(seed).sample(input_list, elements_to_sample)


def vertical_subset_client_data(input_cd: ClientData,
                                subset_client_ids: List[str],
                                check_validity=True) -> ClientData:
  """Extract a subset of ClientData by client_ids.

  Args:
    input_cd: A `ClientData` instance to extract subset from.
    subset_client_ids: The list of client_ids to select from. subset_client_ids
      must be a subset of input_cd.client_ids.
    check_validity: Whether to check the validity of subset_client_ids.

  Returns:
    A `ClientData` instance that holds all the dataset of subset_client_ids.

  Raises:
    ValueError: if subset_client_ids is not a subset of input_cd.client_ids, if
    check_validity is True.
  """

  if check_validity:
    if not set(subset_client_ids).issubset(set(input_cd.client_ids)):
      raise ValueError(
          'subset_client_ids must be a subset of input_cd.client_ids.')

  return tff.simulation.datasets.ClientData.from_clients_and_tf_fn(
      client_ids=subset_client_ids,
      serializable_dataset_fn=input_cd.serializable_dataset_fn)


def interleave_create_tf_dataset_from_all_clients(
    cd: tff.simulation.datasets.ClientData,
    seed: Optional[int]) -> tf.data.Dataset:
  """Creates a new `tf.data.Dataset` containing _all_ client examples.

  This function is intended to accelerate the original TFF function
  `ClientData.create_tf_dataset_from_all_clients()` by invoking
  `tf.data.Dataset.interleave()` to speed up loading.

  Note that this function  does not guarantee the same order as the original
  `ClientData.create_tf_dataset_from_all_clients()`.

  Args:
    cd: A ClientData to be constructed from.
    seed: A seed to determine the order in which clients are processed in the
      joined dataset. The seed can be any nonnegative 32-bit integer, an array
      of such integers, or `None`.

  Returns:
    A `tf.data.Dataset` object.
  """
  client_ids = cd.client_ids.copy()
  np.random.RandomState(seed=seed).shuffle(client_ids)
  nested_dataset = tf.data.Dataset.from_tensor_slices(client_ids)
  # We apply serializable_dataset_fn here to avoid loading all client datasets
  # in memory, which is slow. Note that tf.data.Dataset.map implicitly wraps
  # the input mapping in a tf.function.
  result_dataset = nested_dataset.interleave(
      map_func=cd.serializable_dataset_fn,
      block_length=_INTERLEAVE_BLOCK_LENGTH,
      num_parallel_calls=tf.data.AUTOTUNE)
  return result_dataset


def horizontal_split_client_data(
    cd: ClientData,
    first_second_ratio_for_each_client: int,
    shuffle_before_split: bool = True,
    shuffle_seed: Optional[int] = None,
    shuffle_buffer_size: int = 100,
    remove_single_elem_input_clients: bool = False
) -> Tuple[ClientData, ClientData]:
  """Split a ClientData horizontally.

  This function will accept one ClientData and return two ClientData, named
  first_cd and second_cd. First_cd and second_cd will have the same set of
  clients as the original input cd. For each client, `second_cd` will hold
  1/(first_second_ratio_for_each_client+1) elements of the local dataset,
  round up if fractional. `first_cd` will hold the remaining local dataset.

            +-----------+                  +-----------+
            |           |                  | second_cd |  1
            |           |   (ratio = 4)    +-----------+
            |  (input)  | ===============> |           |
            |     cd    |                  |  first_cd |  4
            |           |                  |           |
            |           |                  |           |
            +-----------+                  +-----------+
               clients                        clients

  Warning: Note that this function may result in a first_cd with empty client(s)
  if a client of input cd has only 1 element. This may result in unexpected
  behaviors when such a clientdata is consumed by the downstream functions.
  (The `ClientData` class is mostly intended for all non-empty clients.)

  We provide an option to circumvent this issue. If
  `remove_single_elem_input_clients` is True, this function will iterate over
  all the clients, removing the clients with only one element from cd.

  The default choice of `remove_single_elem_input_clients` is False since
  iterating over the entire ClientData (especially when it is file-based)
  can be prohibitively costly. We encourate users to manually record the
  single-element clients or implement the single-element-client-remover when a
  efficient approach is possible (e.g. when external metadata is available.)

  Args:
    cd: A `ClientData` instance to split.
    first_second_ratio_for_each_client: A positive integer representing the
      ratio of local dataset split for each client, see above descriptions for
      details.
    shuffle_before_split: A boolean indicating whether to shuffle the data
      before splitting each client dataset.
    shuffle_seed: An optional integer-valued random seed for pre-shuffling
      before split, if shuffle_before_split is True.
    shuffle_buffer_size: An integer representing the shuffling buffer, if
      shuffle_before_split is True.
    remove_single_elem_input_clients: Whether to remove single-element client
      from input cd. The default choice is False since it is very costly. See
      above descriptions for details.

  Returns:
    Two `ClientData` instances split as described above.
  """

  if remove_single_elem_input_clients:
    client_ids_with_more_than_one_elems = []

    for client_id in cd.client_ids:
      ds = cd.create_tf_dataset_for_client(client_id)
      try:
        it = iter(ds)
        next(it)
        next(it)
        client_ids_with_more_than_one_elems.append(client_id)
      except StopIteration:
        pass

    cd = vertical_subset_client_data(
        cd, subset_client_ids=client_ids_with_more_than_one_elems)

  if shuffle_before_split:
    # If shuffle_seed is None, a shuffle_seed will be randomly generated to
    # seed the shuffler. This ensures the same shuffling results when
    # generating first_cd and second_cd.
    if shuffle_seed is None:
      shuffle_seed = np.random.RandomState().randint(65535)

    def shuffler(ds: tf.data.Dataset) -> tf.data.Dataset:
      return ds.shuffle(
          buffer_size=shuffle_buffer_size,
          seed=shuffle_seed,
          reshuffle_each_iteration=False)

    cd = cd.preprocess(shuffler)

  divisor = first_second_ratio_for_each_client + 1
  get_entry = lambda i, entry: entry

  first_pred = lambda i, entry: tf.greater(tf.math.floormod(i, divisor), 0)
  first_cd = cd.preprocess(
      lambda ds: ds.enumerate().filter(first_pred).map(get_entry))

  second_pred = lambda i, entry: tf.equal(tf.math.floormod(i, divisor), 0)
  second_cd = cd.preprocess(
      lambda ds: ds.enumerate().filter(second_pred).map(get_entry))

  return first_cd, second_cd


def horizontal_concat_client_data(first_cd: ClientData,
                                  second_cd: ClientData) -> ClientData:
  """Concatenate two ClientData with the same set of clients.

  This is the inverse operation of `horizontal_split_client_data`, up to the
  order of local dataset.

            +-----------+                  +-----------+
            | second_cd |                  |           |
            +-----------+                  |           |
            |           | ===============> |  result   |
            |  first_cd |                  |    cd     |
            |           |                  |           |
            |           |                  |           |
            +-----------+                  +-----------+
               clients                        clients

  Args:
    first_cd: The first ClientData instance.
    second_cd: The second ClientData instance.

  Returns:
    The resulting ClientData by concatenating the two instances.

  Raises:
    ValueError: if the two input ClientData instances do not have the same set
    of clients.
  """

  if set(first_cd.client_ids) != set(second_cd.client_ids):
    raise ValueError(
        'first_cd and second_cd must possess the same set of clients.')

  def concat_dataset_fn(client_id: str) -> tf.data.Dataset:
    ds1 = first_cd.serializable_dataset_fn(client_id)
    ds2 = second_cd.serializable_dataset_fn(client_id)
    return ds1.concatenate(ds2)

  return tff.simulation.datasets.ClientData.from_clients_and_tf_fn(
      client_ids=first_cd.client_ids, serializable_dataset_fn=concat_dataset_fn)


def construct_three_way_split_from_predefined_horizontal_split(
    train_cd_orig: ClientData,
    val_cd_orig: ClientData,
    *,  # Caller passes below args by name.
    unpart_clients_proportion: float,
    part_clients_subsampling_rate: float,
    include_unpart_train_for_val: bool,
    max_elements_per_client: Optional[int],
    seed: Optional[int] = None,
) -> Tuple[ClientData, ClientData, ClientData]:
  """Construct three-way split from a pre-defined horizontal split.

  If include_unpart_train_for_val is True:

    +----------------+              +-----------+----+
    |  val_cd_orig   |              | part_val  | u  |
    +----------------+              +-----------+ n  |
    |                | ===========> |           | p  |
    | train_cd_orig  |              |  particip | a  |
    |                |              |   train     r  |
    |                |              |           | t  |
    +----------------+              +-----------+----+
        all clients                   partcip.  unparticip.

  If include_unpart_train_for_val is False (XX means discarded):

    +----------------+              +-----------+----+
    |  val_cd_orig   |              |  part_val | unp|
    +----------------+              +-----------+----+
    |                | ===========> |           |XXXX|
    | train_cd_orig  |              |  particip |XXXX|
    |                |              |   train   |XXXX|
    |                |              |           |XXXX|
    +----------------+              +-----------+----+
        all clients                   partcip.  unparticip.

  Args:
    train_cd_orig: The pre-defined "training" client data.
    val_cd_orig: The pre-defined "validation" client data.
    unpart_clients_proportion:  A floating number in (0.0, 1.0) representing the
      proportion of un-participating clients among the total clients.
    part_clients_subsampling_rate: A floating number in (0.0, 1.0] representing
      the actual proportion of candidate participating clients.
    include_unpart_train_for_val:  Whether to include the training dataset of
      unparticipated clients for validation.
    max_elements_per_client: An optional integer controlling the maximum number
      of elements to take per client. If None, keep all elements for each
      client. This is intended primarily to contend with the small set of
      clients with tens of thousands of examples or ablation study.
    seed: An optional integer representing the random seed for partitioning (and
      subsampling, if applicable). If None, no seed is used.

  Returns:
    Three ClientData instances representing participating training,
    participating validation and unparticipating (validation).
  """
  if unpart_clients_proportion is None:
    raise ValueError(
        'unpart_clients_proportion cannot be None since EMNIST does not provide '
        'a vertical split.')
  if unpart_clients_proportion <= 0.0 or unpart_clients_proportion >= 1.0:
    raise ValueError('unpart_clients_proportion must be in (0.0, 1.0), '
                     f'got {unpart_clients_proportion}.')
  if part_clients_subsampling_rate <= 0.0 or part_clients_subsampling_rate > 1.0:
    raise ValueError('part_clients_subsampling_rate must be in (0.0, 1.0], '
                     f'get {part_clients_subsampling_rate}.')

  unpart_client_ids = subsample_list_by_proportion(
      train_cd_orig.client_ids, unpart_clients_proportion, seed=seed)

  part_client_ids = subsample_list_by_proportion(
      list(set(train_cd_orig.client_ids) - set(unpart_client_ids)),
      part_clients_subsampling_rate,
      seed=seed)

  part_train_cd = vertical_subset_client_data(train_cd_orig, part_client_ids)
  unpart_train_cd = vertical_subset_client_data(train_cd_orig,
                                                unpart_client_ids)

  part_val_cd = vertical_subset_client_data(val_cd_orig, part_client_ids)
  unpart_val_cd = vertical_subset_client_data(val_cd_orig, unpart_client_ids)

  if include_unpart_train_for_val:
    unpart_cd = horizontal_concat_client_data(unpart_train_cd, unpart_val_cd)
  else:
    unpart_cd = unpart_val_cd

  if max_elements_per_client is not None:
    truncate_fn = lambda ds: ds.take(max_elements_per_client)

    part_train_cd = part_train_cd.preprocess(truncate_fn)
    part_val_cd = part_val_cd.preprocess(truncate_fn)
    unpart_cd = unpart_cd.preprocess(truncate_fn)

  return part_train_cd, part_val_cd, unpart_cd


def canonical_three_way_partition_client_data(
    cd: ClientData,
    *,  # Caller passes below args by name.
    unpart_clients_proportion: float,
    train_val_ratio_intra_client: int,
    part_clients_subsampling_rate: float,
    include_unpart_train_for_val: bool,
    max_elements_per_client: Optional[int],
    shuffle_buffer_size: int = 100,
    seed: Optional[int] = None) -> Tuple[ClientData, ClientData, ClientData]:
  """Standard three-way parititons of a bulk ClientData.

  This function will accept a ClientData instance and create a three-way
  partition of the given ClientData: part_train_cd, part_val_cd and unpart_cd.

  If include_unpart_train_for_val is True:

      +----------------+              +-----------+----+
      |                |              | part_val  | u  |
      |                |              +-----------+ n  |
      |      cd        | ===========> |           | p  |
      |                |              |  particip | a  |
      |                |              |   train     r  |
      |                |              |           | t  |
      +----------------+              +-----------+----+
         all clients                   partcip.  unparticip.

     This mode may be preferred for obtaining low-variance metrics.

   If include_unpart_train_for_val is False (XX means discarded):

      +----------------+              +-----------+----+
      |                |              |  part_val | unp|
      |                |              +-----------+----+
      |      cd        | ===========> |           |XXXX|
      |                |              |  particip |XXXX|
      |                |              |   train   |XXXX|
      |                |              |           |XXXX|
      +----------------+              +-----------+----+
         all clients                  partcip.  unparticip.

     This mode may be preferred if one wants to compare the percentile /
          variance of part_val and unpart (since they will have the same scale
          of elements.)

  Args:
    cd: A `ClientData` instance to partition.
    unpart_clients_proportion:  A floating number in (0.0, 1.0) representing the
      proportion of un-participating clients among the total clients.
    train_val_ratio_intra_client: An integer representing the ratio of
      train-validation split for each client. For example, setting this value to
      4 will yield a 80%/20% train validation split for each client.
    part_clients_subsampling_rate: A floating number in (0.0, 1.0] representing
      the actual proportion of clients that participates, out of the clients
      that are not in unpart_clients_proportion.
    include_unpart_train_for_val:  Whether to include the training dataset of
      unparticipated clients for validation.
    max_elements_per_client: An optional integer controlling the maximum number
      of elements to take per client. If None, keep all elements for each
      client. This is intended primarily to contend with the small set of
      clients with tens of thousands of examples or ablation study. This
      truncation is applied after all the previous splits, and effective for all
      the three-way split.
    shuffle_buffer_size: An integer value representing the buffer size for
      shuffling the local dataset during horizontal split.
    seed: An optional integer representing the random seed. If None, no seed is
      used.

  Returns:
    Three ClientData instances: part_train_cd, part_val_cd, and unpart_cd.
  """

  if unpart_clients_proportion is None:
    raise ValueError(
        'unpart_clients_proportion cannot be None for canonical 3-way split.')
  if train_val_ratio_intra_client is None:
    raise ValueError(
        'train_val_ratio_intra_client cannot be None for canonical 3-way split.'
    )
  if train_val_ratio_intra_client < 1:
    raise ValueError('train_val_ratio_intra_client must be a integer >= 1, '
                     f'get {train_val_ratio_intra_client}.')
  if part_clients_subsampling_rate <= 0.0 or part_clients_subsampling_rate > 1.0:
    raise ValueError('part_clients_subsampling_rate must be in (0.0, 1.0], '
                     f'get {part_clients_subsampling_rate}.')

  unpart_client_ids = subsample_list_by_proportion(
      cd.client_ids, unpart_clients_proportion, seed=seed)

  part_client_ids = subsample_list_by_proportion(
      list(set(cd.client_ids) - set(unpart_client_ids)),
      part_clients_subsampling_rate,
      seed=seed)

  part_cd = vertical_subset_client_data(cd, part_client_ids)
  unpart_cd = vertical_subset_client_data(cd, unpart_client_ids)

  part_train_cd, part_val_cd = horizontal_split_client_data(
      part_cd,
      train_val_ratio_intra_client,
      shuffle_before_split=True,
      shuffle_seed=seed,
      shuffle_buffer_size=shuffle_buffer_size)

  if not include_unpart_train_for_val:
    _, unpart_cd = horizontal_split_client_data(
        unpart_cd,
        train_val_ratio_intra_client,
        shuffle_before_split=True,
        shuffle_seed=seed,
        shuffle_buffer_size=shuffle_buffer_size)

  if max_elements_per_client is not None:

    def truncate_fn(ds: tf.data.Dataset) -> tf.data.Dataset:
      ds = ds.take(max_elements_per_client)
      return ds

    part_train_cd = part_train_cd.preprocess(truncate_fn)
    part_val_cd = part_val_cd.preprocess(truncate_fn)
    unpart_cd = unpart_cd.preprocess(truncate_fn)

  return part_train_cd, part_val_cd, unpart_cd


def construct_client_data_from_mapping(
    cid_to_ds_dict: Mapping[str, tf.data.Dataset]) -> ClientData:
  """Construct a `ClientData` instance based on a mapping from client_id to dataset.

  Unlike `tff.simulation.datasets.TestClientData`, which embeds the entire
    dataset proper as a tensor into the serialization graph, this function will
    only embed the dataset a placeholder of data pipeline. This allows for more
    efficient construction and reduces memory cost.

  Known issues: This function may not work well with accelerators due to known
    issues of `@tf.function`, see
    https://github.com/tensorflow/tensorflow/issues/34112%23issuecomment-611034242&sa=D

  Args:
    cid_to_ds_dict: A mapping from client_id to `tf.data.Dataset`. All the
      datasets must share the same `element_spec`.

  Returns:
    A `ClientData` instance.
  """
  client_ids = list(cid_to_ds_dict.keys())
  structure = cid_to_ds_dict[client_ids[0]].element_spec

  def bridge(ds: tf.data.Dataset) -> tf.data.Dataset:
    # We convert to __iter__ and then consume by `from_generator` to
    # "isolate" the dataset from its underlying computation graph.
    ds = tf.data.Dataset.from_generator(ds.__iter__, output_signature=structure)
    return ds

  @tf.function
  def get_variant_tensor() -> tf.Tensor:
    with tf.device('/CPU:0'):
      return tf.convert_to_tensor([
          tf.data.experimental.to_variant(bridge(cid_to_ds_dict[cid]))
          for cid in client_ids
      ])

  def get_hashtable():
    return tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=client_ids, values=list(range(len(client_ids)))),
        default_value=-1)

  def serializable_dataset_fn(client_id: str) -> tf.data.Dataset:
    # The serializable function will
    # i) Convert the client_id to a Tensor (to prepare for hashtable lookup)
    # ii) Look up for the index in variant_tensor.
    # iii) Find the corresponding tf.variant from variant_tensor by indexing.
    # iv) Retrieve the dataset from `tf.variant` tensor.
    variant_tensor = get_variant_tensor()
    hashtable = get_hashtable()
    return tf.data.experimental.from_variant(
        variant_tensor[hashtable.lookup(tf.convert_to_tensor(client_id))],
        structure)

  return tff.simulation.datasets.ClientData.from_clients_and_tf_fn(
      client_ids=client_ids, serializable_dataset_fn=serializable_dataset_fn)


def cache_client_data(cd: ClientData) -> ClientData:
  """Cache a ClientData instance.

  Args:
    cd: A `ClientData` instance to cache.

  Returns:
    A `ClientData` instance.
  """

  cid_to_ds_dict = {
      client_id: cd.create_tf_dataset_for_client(client_id).cache()
      for client_id in cd.client_ids
  }

  return construct_client_data_from_mapping(cid_to_ds_dict)


class FederatedDatasetSampler(abc.Iterator):
  """An infinite iterator used to sample dataset from client_data."""

  def __init__(self,
               client_data: tff.simulation.datasets.ClientData,
               num_sample_clients: Optional[int] = None,
               resample: bool = False,
               use_cache: bool = False,
               seed: Optional[int] = None):
    """Initialize the sampling iterator.

    Args:
      client_data: A `tff.simulation.datasets.ClientData` to be sampled from.
      num_sample_clients: An optional integer representing the maximum number of
        clients taken from client_data. If set to `None` or greater than the
        total number of clients, all clients will be used.
      resample: Whether or not to resample clients every time invoked.
      use_cache: Whether to cache the presampled federated dataset.
      seed: An optional integer used to seed which validation clients are
        sampled. If `None`, no seed is used.
    """

    self._client_data = client_data
    self._resample = resample
    self._rng = np.random.default_rng(seed=seed)
    self._use_cache = use_cache

    self._total_clients = len(self._client_data.client_ids)
    if num_sample_clients is None or num_sample_clients > self._total_clients:
      self._num_sample_clients = self._total_clients
    else:
      self._num_sample_clients = num_sample_clients

    self._presampled_federated_dataset = None

  def _sample_federated_dataset(self) -> List[tf.data.Dataset]:
    sample_client_id_indices = self._rng.choice(
        self._total_clients, self._num_sample_clients, replace=False)

    sample_federated_dataset = [
        self._client_data.create_tf_dataset_for_client(
            self._client_data.client_ids[index])
        for index in sample_client_id_indices
    ]

    if self._use_cache:
      sample_federated_dataset = [ds.cache() for ds in sample_federated_dataset]

    return sample_federated_dataset

  def __next__(self) -> List[tf.data.Dataset]:
    if not self._resample:
      if self._presampled_federated_dataset is None:
        self._presampled_federated_dataset = self._sample_federated_dataset()
      return self._presampled_federated_dataset
    else:
      return self._sample_federated_dataset()
