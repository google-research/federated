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
"""Tests for client_data_utils."""

import collections
import math

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_federated as tff

from generalization.utils import client_data_utils

_unpack_fn = lambda fds: [list(ds) for ds in fds]
_unpack_ds_to_list = lambda ds: [t.numpy() for t in ds]


def _create_test_client_data(num_clients=10, samples_per_client=10):
  return tff.simulation.datasets.TestClientData({
      str(i): [i + j for j in range(samples_per_client)
              ] for i in range(num_clients)
  })


def _build_fake_elem(x):
  return collections.OrderedDict([('a', 1 + x), ('b', 3 + x), ('c', 5.0 + x)])


class TestConvertToTensorSlices(tf.test.TestCase):
  """Test for convert_to_tensor_slices."""

  def test_convert_list_of_elems_to_tensor_slices(self):

    list_of_elems = [_build_fake_elem(x) for x in range(5)]

    tensor_slices = client_data_utils.convert_list_of_elems_to_tensor_slices(
        list_of_elems)

    self.assertEqual(type(tensor_slices), collections.OrderedDict)
    self.assertEqual(list(tensor_slices.keys()), ['a', 'b', 'c'])

    self.assertAllEqual(tensor_slices['a'], tf.constant(range(1, 6)))
    self.assertAllEqual(tensor_slices['b'], tf.constant(range(3, 8)))
    self.assertAllEqual(tensor_slices['c'],
                        tf.constant(range(5, 10), dtype=float))

  def test_convert_dataset_to_tensor_slices_reconstructs_client_data(self):
    cd = tff.simulation.datasets.TestClientData({
        str(i): collections.OrderedDict(a=[i + j for j in range(5)])
        for i in range(6)
    })

    tensor_slices_dict = dict()
    for client_id in cd.client_ids:
      local_ds = cd.create_tf_dataset_for_client(client_id)
      tensor_slices_dict[
          client_id] = client_data_utils.convert_list_of_elems_to_tensor_slices(
              list(local_ds))

    reconstruct_cd = tff.simulation.datasets.TestClientData(tensor_slices_dict)

    self.assertCountEqual(reconstruct_cd.client_ids, cd.client_ids)
    self.assertEqual(reconstruct_cd.element_type_structure,
                     cd.element_type_structure)
    for client_id in cd.client_ids:
      reconstruct_ds = reconstruct_cd.create_tf_dataset_for_client(client_id)
      reconstruct_ds_list = [elem['a'].numpy() for elem in reconstruct_ds]

      ds = cd.create_tf_dataset_for_client(client_id)
      ds_list = [elem['a'].numpy() for elem in ds]

      self.assertEqual(reconstruct_ds_list, ds_list)


class TestSubsampleListByProportion(tf.test.TestCase, parameterized.TestCase):
  """Test for subsample_list_by_proportion."""

  @parameterized.product(
      input_len=[5, 9], proportion=[0.0, 0.5, 1.0], seed=[0, 1, None])
  def test_correct_length(self, input_len, proportion, seed):
    input_list = list(range(input_len))

    result_list = client_data_utils.subsample_list_by_proportion(
        input_list, proportion, seed=seed)
    self.assertLen(result_list, round(input_len * proportion))

  @parameterized.product(
      input_len=[5, 9],
      proportion=[0.0, 0.5, 1.0],
  )
  def test_use_seed(self, input_len, proportion):
    input_list = list(range(input_len))
    result_list_1 = client_data_utils.subsample_list_by_proportion(
        input_list, proportion, seed=1)
    result_list_2 = client_data_utils.subsample_list_by_proportion(
        input_list, proportion, seed=1)
    self.assertEqual(result_list_1, result_list_2)


class InterleaveCreateTFDatasetFromAllClients(tf.test.TestCase,
                                              parameterized.TestCase):
  """Test for `interleave_create_tf_dataset_from_all_clients`."""

  @parameterized.named_parameters(('seed=None', None), ('seed=1', 1))
  def test_interleave_create_tf_dataset_from_all_clients(self, seed):
    client_data = _create_test_client_data(5, 4)
    tf_dataset = client_data_utils.interleave_create_tf_dataset_from_all_clients(
        cd=client_data, seed=seed)
    self.assertIsInstance(tf_dataset, tf.data.Dataset)

    expected_dataset = client_data.create_tf_dataset_from_all_clients()

    self.assertCountEqual([t.numpy() for t in tf_dataset],
                          [t.numpy() for t in expected_dataset])

  def test_interleave_create_tf_dataset_from_all_clients_uses_random_seed(self):
    client_data = _create_test_client_data(5, 4)

    tf_dataset1 = client_data_utils.interleave_create_tf_dataset_from_all_clients(
        client_data, seed=1)
    tf_dataset2 = client_data_utils.interleave_create_tf_dataset_from_all_clients(
        client_data, seed=1)
    tf_dataset3 = client_data_utils.interleave_create_tf_dataset_from_all_clients(
        client_data, seed=2)

    # We take only the 'x' elements to do exact comparisons.
    dataset1_elements = [t.numpy() for t in tf_dataset1]
    dataset2_elements = [t.numpy() for t in tf_dataset2]
    dataset3_elements = [t.numpy() for t in tf_dataset3]

    self.assertAllEqual(dataset1_elements, dataset2_elements)
    self.assertNotAllEqual(dataset1_elements, dataset3_elements)
    self.assertCountEqual(dataset1_elements, dataset3_elements)


class HorizontalSplitClientDataTest(tf.test.TestCase, parameterized.TestCase):
  """Test for client_data_utils.horizontal_split_client_data."""

  @parameterized.named_parameters(
      (f'first_second_ratio={ratio}, num_elems_per_client={elems}', ratio,
       elems) for ratio, elems in ((1, 2), (3, 2), (3, 6), (4, 10)))
  def test_dataset_has_correct_length(self, ratio, num_elems_per_client):
    num_clients = 3

    cd = _create_test_client_data(num_clients, num_elems_per_client)
    first_cd, second_cd = client_data_utils.horizontal_split_client_data(
        cd, first_second_ratio_for_each_client=ratio, shuffle_before_split=True)

    self.assertListEqual(cd.client_ids, first_cd.client_ids)
    self.assertListEqual(cd.client_ids, second_cd.client_ids)

    second_cd_expected_elems_per_client = math.ceil(num_elems_per_client /
                                                    (ratio + 1))
    first_cd_expected_elems_per_client = num_elems_per_client - second_cd_expected_elems_per_client

    for client_id in cd.client_ids:
      self.assertLen(
          list(first_cd.create_tf_dataset_for_client(client_id)),
          first_cd_expected_elems_per_client)
      self.assertLen(
          list(second_cd.create_tf_dataset_for_client(client_id)),
          second_cd_expected_elems_per_client)

  @parameterized.named_parameters(
      (f'first_second_ratio={ratio}', ratio) for ratio in range(1, 4))
  def test_remove_single_elem_input_clients(self, ratio):

    cd = tff.simulation.datasets.TestClientData({
        '1': [1.0],
        '2': [1.0, 2.0],
        '3': [1.0, 2.0, 3.0]
    })

    first_cd, second_cd = client_data_utils.horizontal_split_client_data(
        cd,
        first_second_ratio_for_each_client=ratio,
        remove_single_elem_input_clients=True)

    self.assertCountEqual(first_cd.client_ids, ['2', '3'])
    self.assertCountEqual(second_cd.client_ids, ['2', '3'])

  @parameterized.named_parameters(
      ('unshuffled', False, None, None),
      ('unshuffled_with_redundant_seed', False, 1, 2),
      ('shuffle_with_same_seed', True, 1, 1))
  def test_split_is_the_same_when_intended(self, shuffle_before_split,
                                           shuffle_seed1, shuffle_seed2):
    num_clients = 3
    num_elems_per_client = 10
    ratio = 3

    cd = _create_test_client_data(num_clients, num_elems_per_client)

    first_cd1, second_cd1 = client_data_utils.horizontal_split_client_data(
        cd,
        first_second_ratio_for_each_client=ratio,
        shuffle_before_split=shuffle_before_split,
        shuffle_seed=shuffle_seed1)

    first_cd2, second_cd2 = client_data_utils.horizontal_split_client_data(
        cd,
        first_second_ratio_for_each_client=ratio,
        shuffle_before_split=shuffle_before_split,
        shuffle_seed=shuffle_seed2)

    for client_id in cd.client_ids:
      self.assertListEqual(
          list(first_cd1.create_tf_dataset_for_client(client_id)),
          list(first_cd2.create_tf_dataset_for_client(client_id)))
      self.assertListEqual(
          list(second_cd1.create_tf_dataset_for_client(client_id)),
          list(second_cd2.create_tf_dataset_for_client(client_id)))

  @parameterized.named_parameters(('unshuffled', False, None),
                                  ('unshuffled_with_redundant_seed', False, 1),
                                  ('shuffle_with_none_seed', True, None),
                                  ('shuffle_with_int_seed', True, 1))
  def test_not_reshuffled_when_repeated(self, shuffle_before_split,
                                        shuffle_seed):
    num_clients = 3
    num_elems_per_client = 10
    ratio = 3

    cd = _create_test_client_data(num_clients, num_elems_per_client)

    first_cd, second_cd = client_data_utils.horizontal_split_client_data(
        cd,
        first_second_ratio_for_each_client=ratio,
        shuffle_before_split=shuffle_before_split,
        shuffle_seed=shuffle_seed)

    first_cd = first_cd.preprocess(lambda ds: ds.repeat(2))
    second_cd = second_cd.preprocess(lambda ds: ds.repeat(2))

    for client_id in cd.client_ids:
      for preproc_cd in (first_cd, second_cd):
        ds = preproc_cd.create_tf_dataset_for_client(client_id)
        list_of_ds = list(ds)

        self.assertListEqual(list_of_ds[:len(list_of_ds) // 2],
                             list_of_ds[len(list_of_ds) // 2:])


class HorizontalConcatClientDataTest(tf.test.TestCase, parameterized.TestCase):
  """Test for client_data_utils.horizontal_concat_client_data."""

  def test_horizontal_concat(self):
    cd1 = _create_test_client_data(5, 3)
    cd2 = _create_test_client_data(5, 4)

    result_cd = client_data_utils.horizontal_concat_client_data(cd1, cd2)

    self.assertCountEqual(result_cd.client_ids, cd1.client_ids)

    for client_id in result_cd.client_ids:
      expected_data = list(cd1.create_tf_dataset_for_client(client_id)) + list(
          cd2.create_tf_dataset_for_client(client_id))

      self.assertLen(
          list(result_cd.create_tf_dataset_for_client(client_id)),
          len(expected_data))

  @parameterized.named_parameters(
      (f'first_second_ratio={ratio}, num_elems_per_client={elems}', ratio,
       elems) for ratio, elems in ((1, 2), (3, 2), (3, 6), (4, 10)))
  def test_split_and_concat_are_reversible_up_to_local_order(
      self, ratio, elems):
    original_cd = _create_test_client_data(5, elems)

    cd1, cd2 = client_data_utils.horizontal_split_client_data(
        original_cd, first_second_ratio_for_each_client=ratio)

    concat_cd = client_data_utils.horizontal_concat_client_data(cd1, cd2)

    self.assertCountEqual(concat_cd.client_ids, original_cd.client_ids)

    for cid in concat_cd.client_ids:
      concat_ds = concat_cd.create_tf_dataset_for_client(cid)
      original_ds = original_cd.create_tf_dataset_for_client(cid)

      self.assertCountEqual([t.numpy() for t in concat_ds],
                            [t.numpy() for t in original_ds])

  def test_raises_value_error_if_client_ids_are_different(self):
    cd1 = _create_test_client_data(5, 3)
    cd2 = _create_test_client_data(4, 3)

    with self.assertRaises(ValueError):
      client_data_utils.horizontal_concat_client_data(cd1, cd2)


class VerticalSubsetClientData(tf.test.TestCase, parameterized.TestCase):
  """Test for client_data_utils.vertical_subset_client_data."""

  def test_vertical_subset(self):
    cd = _create_test_client_data(5, 3)
    subset_client_ids = ['0', '1', '2', '3']

    subset_cd = client_data_utils.vertical_subset_client_data(
        cd, subset_client_ids)

    self.assertCountEqual(subset_cd.client_ids, subset_client_ids)

    expected_subset_cd = _create_test_client_data(4, 3)

    for cid in subset_client_ids:
      result_ds = subset_cd.dataset_computation(cid)
      expected_ds = expected_subset_cd.create_tf_dataset_for_client(cid)

      self.assertCountEqual([t.numpy() for t in result_ds],
                            [t.numpy() for t in expected_ds])

  def test_raises_value_error_if_client_ids_are_not_subset(self):
    cd = _create_test_client_data(5, 3)
    subset_client_ids = ['1', '6']

    with self.assertRaises(ValueError):
      client_data_utils.vertical_subset_client_data(cd, subset_client_ids)


class ThreeWaySplitFromHorizontalSplitTest(tf.test.TestCase,
                                           parameterized.TestCase):
  """Test for client_data_utils.construct_three_way_split_from_predefined_horizontal_split."""

  @parameterized.product(
      include_unpart_train_for_val=[True, False],
      max_elements_per_client=[None, 2])
  def test_load_partitioned_tff_original_emnist_client_data(
      self, include_unpart_train_for_val, max_elements_per_client):

    unpart_clients_proportion = 0.5
    part_clients_subsampling_rate = 0.5

    num_clients = 5
    num_train_elems_per_client = 7
    num_val_elems_per_client = 3

    train_cd_orig = _create_test_client_data(num_clients,
                                             num_train_elems_per_client)
    val_cd_orig = _create_test_client_data(num_clients,
                                           num_val_elems_per_client)

    (part_train_cd, part_val_cd, unpart_cd
    ) = client_data_utils.construct_three_way_split_from_predefined_horizontal_split(
        train_cd_orig,
        val_cd_orig,
        unpart_clients_proportion=unpart_clients_proportion,
        part_clients_subsampling_rate=part_clients_subsampling_rate,
        include_unpart_train_for_val=include_unpart_train_for_val,
        max_elements_per_client=max_elements_per_client)

    # Assert the returned client_datas have the correct number of clients.
    all_client_ids = train_cd_orig.client_ids
    total_clients = len(all_client_ids)

    expected_unpart_clients = round(total_clients * unpart_clients_proportion)
    expected_part_clients = round((total_clients - expected_unpart_clients) *
                                  part_clients_subsampling_rate)

    self.assertLen(part_train_cd.client_ids, expected_part_clients)
    self.assertLen(part_val_cd.client_ids, expected_part_clients)
    self.assertLen(unpart_cd.client_ids, expected_unpart_clients)

    # Assert the correctness of client_ids.
    self.assertCountEqual(part_train_cd.client_ids, part_val_cd.client_ids)

    # Assert detailed equivalence.
    test_part_client_id = part_train_cd.client_ids[0]

    part_train_cd_ds = part_train_cd.create_tf_dataset_for_client(
        test_part_client_id)
    expected_len = len(
        list(train_cd_orig.create_tf_dataset_for_client(test_part_client_id)))
    if max_elements_per_client is not None:
      expected_len = min(max_elements_per_client, expected_len)
    self.assertLen(list(part_train_cd_ds), expected_len)

    part_val_cd_ds = part_val_cd.create_tf_dataset_for_client(
        test_part_client_id)
    expected_len = len(
        list(val_cd_orig.create_tf_dataset_for_client(test_part_client_id)))
    if max_elements_per_client is not None:
      expected_len = min(max_elements_per_client, expected_len)
    self.assertLen(list(part_val_cd_ds), expected_len)

    test_unpart_client_id = unpart_cd.client_ids[0]
    unpart_cd_ds = unpart_cd.create_tf_dataset_for_client(test_unpart_client_id)

    expected_ds = val_cd_orig.create_tf_dataset_for_client(
        test_unpart_client_id)
    if include_unpart_train_for_val:
      expected_ds = expected_ds.concatenate(
          train_cd_orig.create_tf_dataset_for_client(test_unpart_client_id))
    expected_len = len(list(expected_ds))
    if max_elements_per_client is not None:
      expected_len = min(max_elements_per_client, expected_len)
    self.assertLen(list(unpart_cd_ds), expected_len)


class CanonicalThreeWayPartitionTest(tf.test.TestCase, parameterized.TestCase):
  """Test for client_data_utils.canonical_three_way_partition_client_data."""

  @parameterized.product(
      part_clients_subsampling_rate=[0.5, 1.0],
      include_unpart_train_for_val=[True, False],
      max_elements_per_client=[None, 2])
  def test_dataset_has_correct_length(self, part_clients_subsampling_rate,
                                      include_unpart_train_for_val,
                                      max_elements_per_client):
    num_clients = 5
    num_elems_per_client = 7

    unpart_clients_proportion = 0.2
    train_val_ratio_intra_client = 4

    cd = _create_test_client_data(num_clients, num_elems_per_client)

    (part_train_cd, part_val_cd,
     unpart_cd) = client_data_utils.canonical_three_way_partition_client_data(
         cd,
         unpart_clients_proportion=unpart_clients_proportion,
         train_val_ratio_intra_client=train_val_ratio_intra_client,
         part_clients_subsampling_rate=part_clients_subsampling_rate,
         include_unpart_train_for_val=include_unpart_train_for_val,
         max_elements_per_client=max_elements_per_client,
     )

    expected_num_unpart_clients = round(num_clients * unpart_clients_proportion)
    expected_num_part_clients = round(
        (num_clients - expected_num_unpart_clients) *
        part_clients_subsampling_rate)

    self.assertLen(part_train_cd.client_ids, expected_num_part_clients)
    self.assertLen(part_val_cd.client_ids, expected_num_part_clients)
    self.assertLen(unpart_cd.client_ids, expected_num_unpart_clients)
    self.assertCountEqual(part_train_cd.client_ids, part_val_cd.client_ids)

    self.assertEmpty(set(part_train_cd.client_ids) & set(unpart_cd.client_ids))
    self.assertTrue(
        set(part_train_cd.client_ids + unpart_cd.client_ids).issubset(
            cd.client_ids))

    # Fine-grained check:
    expected_val_per_client = (num_elems_per_client +
                               train_val_ratio_intra_client) // (
                                   train_val_ratio_intra_client + 1)
    expected_train_per_client = num_elems_per_client - expected_val_per_client

    if max_elements_per_client is not None:
      expected_train_per_client = min(expected_train_per_client,
                                      max_elements_per_client)
      expected_val_per_client = min(expected_val_per_client,
                                    max_elements_per_client)

    for client_id in part_train_cd.client_ids:
      part_train_ds = part_train_cd.create_tf_dataset_for_client(client_id)
      part_val_ds = part_val_cd.create_tf_dataset_for_client(client_id)

      part_train_ds_list = list(part_train_ds)
      part_val_ds_list = list(part_val_ds)

      self.assertLen(part_train_ds_list, expected_train_per_client)
      self.assertLen(part_val_ds_list, expected_val_per_client)

    if include_unpart_train_for_val:
      expected_unpart_len = num_elems_per_client
    else:
      expected_unpart_len = expected_val_per_client

    if max_elements_per_client is not None:
      expected_unpart_len = min(max_elements_per_client, expected_unpart_len)

    for client_id in unpart_cd.client_ids:
      unpart_ds = unpart_cd.create_tf_dataset_for_client(client_id)
      unpart_ds_list = list(unpart_ds)
      self.assertLen(unpart_ds_list, expected_unpart_len)

  @parameterized.product(
      part_clients_subsampling_rate=[0.5, 1.0],
      include_unpart_train_for_val=[True, False],
      max_elements_per_client=[None, 2])
  def test_three_way_partition_use_seed(self, part_clients_subsampling_rate,
                                        include_unpart_train_for_val,
                                        max_elements_per_client):
    num_clients = 6
    num_elems_per_client = 7

    unpart_clients_proportion = 0.2
    train_val_ratio_intra_client = 4

    cd = _create_test_client_data(num_clients, num_elems_per_client)

    (part_train_cd, part_val_cd,
     unpart_cd) = client_data_utils.canonical_three_way_partition_client_data(
         cd,
         unpart_clients_proportion=unpart_clients_proportion,
         train_val_ratio_intra_client=train_val_ratio_intra_client,
         part_clients_subsampling_rate=part_clients_subsampling_rate,
         include_unpart_train_for_val=include_unpart_train_for_val,
         max_elements_per_client=max_elements_per_client,
         seed=1)
    part_train_cd_same, part_val_cd_same, unpart_cd_same = client_data_utils.canonical_three_way_partition_client_data(
        cd,
        unpart_clients_proportion=unpart_clients_proportion,
        train_val_ratio_intra_client=train_val_ratio_intra_client,
        part_clients_subsampling_rate=part_clients_subsampling_rate,
        include_unpart_train_for_val=include_unpart_train_for_val,
        max_elements_per_client=max_elements_per_client,
        seed=1)

    for cd1, cd2 in zip([part_train_cd, part_val_cd, unpart_cd],
                        [part_train_cd_same, part_val_cd_same, unpart_cd_same]):
      self.assertCountEqual(cd1.client_ids, cd2.client_ids)

      for client_id in cd1.client_ids:
        ds1 = cd1.create_tf_dataset_for_client(client_id)
        ds2 = cd2.create_tf_dataset_for_client(client_id)

        self.assertCountEqual(_unpack_ds_to_list(ds1), _unpack_ds_to_list(ds2))


class ConstructClientDataFromMappingTest(tf.test.TestCase):
  """Test for `construct_client_data_from_mapping`."""

  def test_mapping_with_various_lengths(self):
    test_mapping = {str(i): tf.data.Dataset.range(i) for i in range(1, 8, 3)}

    cd = client_data_utils.construct_client_data_from_mapping(test_mapping)

    self.assertCountEqual(cd.client_ids, [str(i) for i in range(1, 8, 3)])

    for cid in cd.client_ids:
      local_ds = cd.dataset_computation(cid)
      self.assertLen(list(local_ds), int(cid))
      self.assertEqual([t.numpy() for t in local_ds], list(range(int(cid))))

    global_ds = cd.create_tf_dataset_from_all_clients()
    self.assertLen(list(global_ds), 1 + 4 + 7)
    self.assertCountEqual([t.numpy() for t in global_ds],
                          list(range(1)) + list(range(4)) + list(range(7)))

  def test_dataset_with_nested_structure(self):
    test_data = collections.OrderedDict(
        label=([tf.constant(0, dtype=tf.int32)]),
        pixels=([tf.zeros((28, 28), dtype=tf.float32)]),
    )
    ds = tf.data.Dataset.from_tensor_slices(test_data)
    test_mapping = {'1': ds, '2': ds}

    cd = client_data_utils.construct_client_data_from_mapping(test_mapping)

    self.assertCountEqual(cd.client_ids, ('1', '2'))

    for cid in cd.client_ids:
      local_ds = cd.dataset_computation(cid)
      self.assertLen(list(local_ds), 1)

      local_data = next(iter(local_ds))
      self.assertEqual(local_ds.element_spec, ds.element_spec)
      self.assertCountEqual(list(local_data.keys()), list(test_data.keys()))
      self.assertAllEqual(local_data['label'], test_data['label'][0])
      self.assertAllEqual(local_data['pixels'], test_data['pixels'][0])

  def test_dataset_with_very_large_cardinality(self):
    """Test the dataset will not be eagerly computed unexpectedly."""
    test_mapping = {
        'short': tf.data.Dataset.range(10),
        'long': tf.data.Dataset.range(9999999999999999)
    }

    cd = client_data_utils.construct_client_data_from_mapping(test_mapping)

    self.assertCountEqual(cd.client_ids, ('short', 'long'))

    short_ds = cd.create_tf_dataset_for_client('short')
    self.assertEqual([t.numpy() for t in short_ds], list(range(10)))

    long_ds = cd.create_tf_dataset_for_client('long')
    long_ds_take_10 = long_ds.take(10)

    self.assertEqual([t.numpy() for t in long_ds_take_10], list(range(10)))


class CacheClientDataTest(tf.test.TestCase):
  """Tests for cache_client_data."""

  def test_cached_cd_has_same_elements(self):
    cd = _create_test_client_data(5, 5)

    cached_cd = client_data_utils.cache_client_data(cd)

    self.assertCountEqual(cached_cd.client_ids, cd.client_ids)

    for cid in cached_cd.client_ids:
      cached_cd_ds = cached_cd.create_tf_dataset_for_client(cid)
      expected_cd_ds = cd.create_tf_dataset_for_client(cid)
      self.assertCountEqual(
          _unpack_ds_to_list(cached_cd_ds), _unpack_ds_to_list(expected_cd_ds))


class FederatedDatasetSamplerTest(tf.test.TestCase, parameterized.TestCase):
  """Tests for iterator class FederatedDatasetSampler."""

  @parameterized.product(resample=[True, False])
  def test_len_with_specified_num_sample_clients(self, resample):
    """Test sampler returns the correct length."""
    cd = _create_test_client_data(10, 10)
    num_sample_clients = 5
    it = client_data_utils.FederatedDatasetSampler(
        cd, num_sample_clients, resample=resample)
    self.assertLen(next(it), num_sample_clients)

  @parameterized.product(resample=[True, False])
  def test_len_with_unspecified_num_sample_clients(self, resample):
    """Test sampler returns the correct length when num_sample_clients is not specified."""
    cd = _create_test_client_data(10, 10)
    it = client_data_utils.FederatedDatasetSampler(
        cd, num_sample_clients=None, resample=resample)
    self.assertLen(next(it), 10)

  @parameterized.product(use_cache=[True, False])
  def test_sample_emits_the_same_datasets_if_no_resample(self, use_cache):
    """Test sampler emits the same dataset if resample is False."""
    cd = _create_test_client_data(10, 10)
    it = client_data_utils.FederatedDatasetSampler(
        cd, 5, resample=False, seed=1, use_cache=use_cache)

    self.assertListEqual(_unpack_fn(next(it)), _unpack_fn(next(it)))

  @parameterized.product(use_cache=[True, False])
  def test_sample_emits_different_datasets_if_resample(self, use_cache):
    """Test sampler emits a different dataset if resample is True."""
    cd = _create_test_client_data(100, 2)

    # This should not be flaky given the seed.
    it = client_data_utils.FederatedDatasetSampler(
        cd, 50, resample=True, seed=1, use_cache=use_cache)

    self.assertNotEqual(_unpack_fn(next(it)), _unpack_fn(next(it)))

  @parameterized.product(resample=[True, False], use_cache=[True, False])
  def test_two_samplers_with_the_same_int_seed_emit_the_same_datasets(
      self, resample, use_cache):
    """Test two samplers emit the same datasets if the seeds are the same integer."""
    cd = _create_test_client_data(10, 10)
    it1 = client_data_utils.FederatedDatasetSampler(
        cd, 5, resample=resample, seed=1, use_cache=use_cache)
    it2 = client_data_utils.FederatedDatasetSampler(
        cd, 5, resample=resample, seed=1, use_cache=use_cache)

    for _ in range(3):
      self.assertListEqual(_unpack_fn(next(it1)), _unpack_fn(next(it2)))

  @parameterized.product(resample=[True, False], use_cache=[True, False])
  def test_two_samplers_with_none_seed_emit_different_datasets(
      self, resample, use_cache):
    """Test two samplers emit different datasets if seed is the None."""
    cd = _create_test_client_data(100, 2)
    it1 = client_data_utils.FederatedDatasetSampler(
        cd, 50, resample=resample, seed=None, use_cache=use_cache)
    it2 = client_data_utils.FederatedDatasetSampler(
        cd, 50, resample=resample, seed=None, use_cache=use_cache)

    for _ in range(3):
      # This test may be flaky. With <1e-29 probability this test my fail.
      self.assertNotEqual(_unpack_fn(next(it1)), _unpack_fn(next(it2)))

  @parameterized.product(resample=[True, False], use_cache=[True, False])
  def test_two_samplers_with_different_seeds_emit_different_datasets(
      self, resample, use_cache):
    """Test two samplers emit different datasets if seeds are different."""
    cd = _create_test_client_data(100, 2)
    it1 = client_data_utils.FederatedDatasetSampler(
        cd, 50, resample=resample, seed=0, use_cache=use_cache)
    it2 = client_data_utils.FederatedDatasetSampler(
        cd, 50, resample=resample, seed=1, use_cache=use_cache)

    for _ in range(3):
      self.assertNotEqual(_unpack_fn(next(it1)), _unpack_fn(next(it2)))


if __name__ == '__main__':
  tf.test.main()
