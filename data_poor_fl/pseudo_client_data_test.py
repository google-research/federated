# Copyright 2022, Google LLC.
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

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_federated as tff

from data_poor_fl import pseudo_client_data


def _build_balanced_client_data() -> tff.simulation.datasets.ClientData:
  """Creates a simple `ClientData` instance.

  The resulting `ClientData` has the following clients and datasets (written as
  lists):
  *   client `0`: [0, 1, 2]
  *   client `1`: [0, 1, 2]
  *   client `2`: [0, 1, 2]

  Returns:
    A `ConcreteClientData` instance.
  """
  client_ids = ['0', '1', '2']

  def create_dataset_fn(client_id):
    del client_id
    return tf.data.Dataset.range(3)

  return tff.simulation.datasets.ClientData.from_clients_and_tf_fn(
      client_ids=client_ids, serializable_dataset_fn=create_dataset_fn)


def _build_unbalanced_client_data() -> tff.simulation.datasets.ClientData:
  """Creates a simple `ClientData` instance.

  The resulting `ClientData` has the following clients and datasets (written as
  lists):
  *   client `3`: [0, 1, 2]
  *   client `4`: [0, 1, 2, 3]
  *   client `5`: [0, 1, 2, 3, 4]

  Returns:
    A `ConcreteClientData` instance.
  """
  client_ids = ['3', '4', '5']

  def create_dataset_fn(client_id):
    num_examples = tf.strings.to_number(client_id, out_type=tf.int64)
    return tf.data.Dataset.range(num_examples)

  return tff.simulation.datasets.ClientData.from_clients_and_tf_fn(
      client_ids=client_ids, serializable_dataset_fn=create_dataset_fn)


class PseudoClientDataTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('test_case_1', 1),
      ('test_case_2', 2),
      ('test_case_3', 3),
  )
  def test_creates_from_balanced_with_correct_num_clients(
      self, examples_per_pseudo_client):
    base_client_data = _build_balanced_client_data()
    extended_client_data = pseudo_client_data.create_pseudo_client_data(
        base_client_data, examples_per_pseudo_client)
    expected_len = int(tf.math.ceil(3 / examples_per_pseudo_client)) * len(
        base_client_data.client_ids)
    self.assertLen(extended_client_data.client_ids, expected_len)

  @parameterized.named_parameters(
      ('test_case_1', 1),
      ('test_case_2', 2),
      ('test_case_3', 3),
  )
  def test_creates_from_unbalanced_with_correct_num_clients(
      self, examples_per_pseudo_client):
    base_client_data = _build_unbalanced_client_data()
    extended_client_data = pseudo_client_data.create_pseudo_client_data(
        base_client_data, examples_per_pseudo_client)
    expected_len = sum([
        int(tf.math.ceil(i / examples_per_pseudo_client)) for i in range(3, 6)
    ])
    self.assertLen(extended_client_data.client_ids, expected_len)

  def test_pseudo_clients_have_expected_data_from_balanced_data(self):
    base_client_data = _build_balanced_client_data()
    extended_client_data = pseudo_client_data.create_pseudo_client_data(
        base_client_data, examples_per_pseudo_client=1)
    for base_client_id in base_client_data.client_ids:
      for j in range(3):
        pseudo_client_id = base_client_id + '-' + str(j)
        self.assertIn(pseudo_client_id, extended_client_data.client_ids)
        dataset = extended_client_data.create_tf_dataset_for_client(
            pseudo_client_id)
        self.assertEqual(list(dataset.as_numpy_iterator()), [j])

  def test_pseudo_clients_have_expected_data_from_unbalanced_data(self):
    base_client_data = _build_unbalanced_client_data()
    extended_client_data = pseudo_client_data.create_pseudo_client_data(
        base_client_data, examples_per_pseudo_client=1)
    for i in [3, 4, 5]:
      for j in range(i):
        pseudo_client_id = str(i) + '-' + str(j)
        self.assertIn(pseudo_client_id, extended_client_data.client_ids)
        dataset = extended_client_data.create_tf_dataset_for_client(
            pseudo_client_id)
        self.assertEqual(list(dataset.as_numpy_iterator()), [j])

  def test_num_examples_larger_than_max_dataset_size(self):
    base_client_data = _build_unbalanced_client_data()
    extended_client_data = pseudo_client_data.create_pseudo_client_data(
        base_client_data, examples_per_pseudo_client=10)
    self.assertCountEqual(extended_client_data.client_ids,
                          ['3-0', '4-0', '5-0'])
    for i in [3, 4, 5]:
      pseudo_client_id = str(i) + '-0'
      dataset = extended_client_data.create_tf_dataset_for_client(
          pseudo_client_id)
      self.assertEqual(list(dataset.as_numpy_iterator()), list(range(i)))

  def test_create_dataset_from_all_clients_equals_original(self):
    if tf.config.list_logical_devices('GPU'):
      self.skipTest('skip GPU test')
    base_client_data = _build_unbalanced_client_data()
    extended_client_data = pseudo_client_data.create_pseudo_client_data(
        base_client_data, examples_per_pseudo_client=2)
    dataset1 = base_client_data.create_tf_dataset_from_all_clients()
    dataset2 = extended_client_data.create_tf_dataset_from_all_clients()
    self.assertCountEqual(
        list(dataset1.as_numpy_iterator()), list(dataset2.as_numpy_iterator()))

  def test_input_pseudo_client_ids(self):
    base_client_data = _build_unbalanced_client_data()
    pseudo_client_ids = ['3-0', '3-1', '4-2', '5-4']
    extended_client_data = pseudo_client_data.create_pseudo_client_data(
        base_client_data,
        examples_per_pseudo_client=1,
        pseudo_client_ids=pseudo_client_ids)
    self.assertCountEqual(extended_client_data.client_ids, pseudo_client_ids)
    for client_id in pseudo_client_ids:
      dataset = extended_client_data.create_tf_dataset_for_client(client_id)
      expected_dataset_as_list = [int(client_id.split('-')[-1])]
      self.assertEqual(
          list(dataset.as_numpy_iterator()), expected_dataset_as_list)

  def test_pseudo_client_index_split_with_multiple_separator_occurences(self):
    base_client_ids = ['0-a', '1-b', '2-c']

    def create_dataset_fn(client_id):
      del client_id
      return tf.data.Dataset.range(3)

    base_client_data = tff.simulation.datasets.ClientData.from_clients_and_tf_fn(
        client_ids=base_client_ids, serializable_dataset_fn=create_dataset_fn)
    extended_client_data = pseudo_client_data.create_pseudo_client_data(
        base_client_data, examples_per_pseudo_client=1)
    for base_client_id in base_client_ids:
      for i in range(3):
        pseudo_id = base_client_id + '-' + str(i)
        self.assertIn(pseudo_id, extended_client_data.client_ids)
        pseudo_dataset = extended_client_data.create_tf_dataset_for_client(
            pseudo_id)
        self.assertEqual(list(pseudo_dataset.as_numpy_iterator()), [i])

  def test_pseudo_client_index_split_with_non_default_separator(self):
    base_client_ids = ['0-a', '1-b', '2-c']

    def create_dataset_fn(client_id):
      del client_id
      return tf.data.Dataset.range(3)

    base_client_data = tff.simulation.datasets.ClientData.from_clients_and_tf_fn(
        client_ids=base_client_ids, serializable_dataset_fn=create_dataset_fn)
    extended_client_data = pseudo_client_data.create_pseudo_client_data(
        base_client_data, examples_per_pseudo_client=1, separator='_')
    expected_pseudo_client_ids = []
    for base_client_id in base_client_ids:
      for i in range(3):
        expected_pseudo_client_ids.append(base_client_id + '_' + str(i))
    self.assertCountEqual(extended_client_data.client_ids,
                          expected_pseudo_client_ids)


if __name__ == '__main__':
  tf.test.main()
