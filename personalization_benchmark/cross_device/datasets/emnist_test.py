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

from personalization_benchmark.cross_device import constants
from personalization_benchmark.cross_device.datasets import emnist


class EmnistTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      ('even_examples', tf.data.Dataset.range(8), [0, 1, 2, 3], [4, 5, 6, 7]),
      ('odd_examples', tf.data.Dataset.range(7), [0, 1, 2], [3, 4, 5, 6]))
  def test_split_half(self, input_data, expected_first, expected_second):
    first_data, second_data = emnist.split_half(input_data)
    self.assertListEqual(list(first_data), expected_first)
    self.assertListEqual(list(second_data), expected_second)

  def test_create_model_and_data(self):
    train_batch_size = 2
    model_fn, datasets, train_preprocess_fn, split_data_fn, accuracy_name = (
        emnist.create_model_and_data(
            num_local_epochs=1,
            train_batch_size=train_batch_size,
            use_synthetic_data=True))
    self.assertEqual(accuracy_name, emnist._ACCURACY_NAME)
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
      # Before splitting, the client's local dataset has 10 examples, after
      # splitting, each of the two datasets have 5 examples.
      expected_size_before_split = 10
      expected_personalization_size = 5
      expected_test_size = 5
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
