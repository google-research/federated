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
import tensorflow as tf
import tensorflow_federated as tff
from personalization_benchmark.cross_device import constants
from personalization_benchmark.cross_device.datasets import stackoverflow


class StackoverflowTest(tf.test.TestCase):

  def test_creation_date_string_to_integer(self):
    synthetic_data = tff.simulation.datasets.stackoverflow.get_synthetic()
    dataset = synthetic_data.create_tf_dataset_for_client(
        synthetic_data.client_ids[0])
    integer_list = [
        stackoverflow._creation_date_string_to_integer(example['creation_date'])
        for example in dataset
    ]
    # The synthetic examples' creation dates are: '2010-01-08 09:34:05 UTC',
    # '2008-08-10 08:28:52.1 UTC', and '2008-08-10 08:28:52.1 UTC'.
    expected_integer_list = [20100108093405, 20080810082852, 20080810082852]
    self.assertEqual(integer_list, expected_integer_list)

  def test_sort_examples_by_date(self):
    synthetic_data = tff.simulation.datasets.stackoverflow.get_synthetic()
    dataset = synthetic_data.create_tf_dataset_for_client(
        synthetic_data.client_ids[0])
    # The synthetic client has 3 examples in total.
    sorted_dataset = dataset.batch(3).map(
        stackoverflow._sort_examples_by_date).unbatch()
    sorted_examples_list = list(sorted_dataset)
    # The synthetic examples' creation dates are: '2010-01-08 09:34:05 UTC',
    # '2008-08-10 08:28:52.1 UTC', and '2008-08-10 08:28:52.1 UTC', so after
    # sorting, the first example is moved to be after the other two examples.
    original_examples_list = list(dataset)
    expected_sorted_examples_list = [
        original_examples_list[1], original_examples_list[2],
        original_examples_list[0]
    ]
    self.assertEqual(sorted_examples_list, expected_sorted_examples_list)

  def test_create_model_and_data(self):
    train_batch_size = 1
    model_fn, datasets, train_preprocess_fn, split_data_fn, accuracy_name = (
        stackoverflow.create_model_and_data(
            num_local_epochs=1,
            train_batch_size=train_batch_size,
            use_synthetic_data=True))
    self.assertEqual(accuracy_name, stackoverflow._ACCURACY_NAME)
    model = model_fn()
    self.assertIsInstance(model, tff.learning.Model)
    self.assertIn(accuracy_name, model.report_local_unfinalized_metrics())
    self.assertEqual(
        list(datasets.keys()), [
            constants.TRAIN_CLIENTS_KEY, constants.VALID_CLIENTS_KEY,
            constants.TEST_CLIENTS_KEY
        ])
    train_client_data = datasets[constants.TRAIN_CLIENTS_KEY]
    valid_client_data = datasets[constants.VALID_CLIENTS_KEY]
    test_client_data = datasets[constants.TEST_CLIENTS_KEY]
    for client_data in [train_client_data, valid_client_data, test_client_data]:
      self.assertIsInstance(client_data, tff.simulation.datasets.ClientData)
    # Assert that we can train the model on a single client's data.
    train_data_for_first_client = train_preprocess_fn(
        train_client_data.create_tf_dataset_for_client(
            train_client_data.client_ids[0]))
    train_batch_for_first_client = iter(train_data_for_first_client).next()
    batch_output = model.forward_pass(train_batch_for_first_client)
    self.assertEqual(batch_output.num_examples, train_batch_size)
    # Assert that we can split the validation and test clients' data.
    for client_data in [valid_client_data, test_client_data]:
      client_data_before_split = client_data.create_tf_dataset_for_client(
          client_data.client_ids[0])
      client_data_after_split = split_data_fn(client_data_before_split)
      self.assertEqual(
          list(client_data_after_split.keys()),
          [constants.PERSONALIZATION_DATA_KEY, constants.TEST_DATA_KEY])
      personalization_data = client_data_after_split[
          constants.PERSONALIZATION_DATA_KEY]
      test_data = client_data_after_split[constants.TEST_DATA_KEY]
      # Before splitting, the client's local dataset has 3 examples, after
      # splitting, the two datasets have 1 and 2 examples.
      expected_size_before_split = 3
      expected_personalization_size = 1
      expected_test_size = 2
      self.assertLen(list(client_data_before_split), expected_size_before_split)
      self.assertLen(list(personalization_data), expected_personalization_size)
      self.assertLen(list(test_data), expected_test_size)
      # Assert that `model.forward_pass` works on both datasets after splitting.
      for data in [personalization_data, test_data]:
        first_batch = iter(data.batch(train_batch_size)).next()
        batch_output = model.forward_pass(first_batch)
        self.assertEqual(batch_output.num_examples, train_batch_size)


if __name__ == '__main__':
  tf.test.main()
