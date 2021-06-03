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

import functools

import tensorflow as tf
import tensorflow_federated as tff

from large_cohort import data_utils


def _create_concrete_client_data():
  client_ids = [str(x) for x in range(10)]

  def create_dataset_fn(client_id):
    num_examples = tf.strings.to_number(client_id, out_type=tf.int64)
    return tf.data.Dataset.range(num_examples)

  # TODO(b/186139255) Change call to ClientData once from_clients_and_tf_fn has
  # been propagated to the ClientData class.
  return tff.simulation.datasets.TestClientData.from_clients_and_tf_fn(
      client_ids, create_dataset_fn)


class TrainValidationSplitTest(tf.test.TestCase):

  def test_random_seed_changes_sampling(self):
    client_data = _create_concrete_client_data()
    train_data1, validation_data1 = data_utils.create_train_validation_split(
        client_data, seed=1)
    train_data2, validation_data2 = data_utils.create_train_validation_split(
        client_data, seed=2)
    train_data3, validation_data3 = data_utils.create_train_validation_split(
        client_data, seed=1)
    self.assertEqual(train_data1.client_ids, train_data3.client_ids)
    self.assertEqual(validation_data1.client_ids, validation_data3.client_ids)
    self.assertNotEqual(train_data1.client_ids, train_data2.client_ids)
    self.assertNotEqual(validation_data1.client_ids,
                        validation_data2.client_ids)

  def test_split_partitions_into_80_20_split(self):
    client_data = _create_concrete_client_data()
    train_data, validation_data = data_utils.create_train_validation_split(
        client_data, seed=1)
    self.assertLen(train_data.client_ids, 8)
    self.assertLen(validation_data.client_ids, 2)
    self.assertContainsSubset(train_data.client_ids, client_data.client_ids)
    self.assertContainsSubset(validation_data.client_ids,
                              client_data.client_ids)
    self.assertCountEqual(client_data.client_ids,
                          train_data.client_ids + validation_data.client_ids)

  def test_split_leaves_dataset_creation_unchanged(self):
    client_data = _create_concrete_client_data()
    train_data, validation_data = data_utils.create_train_validation_split(
        client_data, seed=1)
    for client_id in train_data.client_ids:
      actual_examples = list(
          train_data.create_tf_dataset_for_client(
              client_id).as_numpy_iterator())
      expected_examples = list(
          client_data.create_tf_dataset_for_client(
              client_id).as_numpy_iterator())
      self.assertEqual(actual_examples, expected_examples)

    for client_id in validation_data.client_ids:
      actual_examples = list(
          validation_data.create_tf_dataset_for_client(
              client_id).as_numpy_iterator())
      expected_examples = list(
          client_data.create_tf_dataset_for_client(
              client_id).as_numpy_iterator())
      self.assertEqual(actual_examples, expected_examples)

  def test_split_leaves_dataset_computation_unchanged(self):
    client_data = _create_concrete_client_data()
    train_data, validation_data = data_utils.create_train_validation_split(
        client_data, seed=1)
    for client_id in train_data.client_ids:
      actual_examples = list(
          train_data.dataset_computation(client_id).as_numpy_iterator())
      expected_examples = list(
          client_data.dataset_computation(client_id).as_numpy_iterator())
      self.assertEqual(actual_examples, expected_examples)

    for client_id in validation_data.client_ids:
      actual_examples = list(
          validation_data.dataset_computation(client_id).as_numpy_iterator())
      expected_examples = list(
          client_data.dataset_computation(client_id).as_numpy_iterator())
      self.assertEqual(actual_examples, expected_examples)


class CreateSamplingFnTest(tf.test.TestCase):

  def test_seeds_are_deterministic(self):
    sample_size = 3
    with self.subTest('no_doubling'):
      for _ in range(3):
        # Re-create and re-run the test to ensure the sampling sequence
        # restarts.
        sampling_fn = data_utils.create_sampling_fn(
            seed=1,
            client_ids=[str(i) for i in range(10)],
            clients_per_round=sample_size,
            rounds_to_double_cohort=None)
        for round_num, expected_sample in enumerate([('2', '0', '4'),
                                                     ('7', '0', '6')]):
          self.assertCountEqual(sampling_fn(round_num), expected_sample)
    with self.subTest('doubling'):
      for _ in range(3):
        # Re-create and re-run the test to ensure the sampling sequence
        # restarts.
        sampling_fn = data_utils.create_sampling_fn(
            seed=2,
            client_ids=[str(i) for i in range(10)],
            clients_per_round=sample_size,
            rounds_to_double_cohort=1)
        for round_num, expected_sample in enumerate([
            ('9', '3', '7'),
            ('5', '0', '7', '8', '6', '1'),
            ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'),
        ]):
          sampled_clients = sampling_fn(round_num)
          self.assertCountEqual(
              sampled_clients, expected_sample, msg=sampled_clients)

  def test_different_seeds_produce_different_samples(self):
    client_ids = [str(i) for i in range(1000)]
    create_sample_fn = functools.partial(
        data_utils.create_sampling_fn,
        client_ids=client_ids,
        clients_per_round=50,
        rounds_to_double_cohort=None)
    # Build a set of the first samples for 100 smapling functions. The set
    # will de-duplicate any repeats, so we assert on the size of the set
    # to ensure there were no duplicates.
    num_samples = 100
    first_samples = set([
        tuple(create_sample_fn(seed=seed)(round_num=0))
        for seed in range(num_samples)
    ])
    self.assertLen(first_samples, num_samples)

  def test_no_doubling(self):
    sample_size = 3
    sampling_fn = data_utils.create_sampling_fn(
        seed=1,
        client_ids=[str(i) for i in range(10)],
        clients_per_round=sample_size,
        rounds_to_double_cohort=None)
    for round_num in range(1_000):
      cohort = sampling_fn(round_num)
      self.assertLen(cohort, sample_size)

  def test_doubling(self):
    with self.subTest('every_round'):
      sampling_fn = data_utils.create_sampling_fn(
          seed=1,
          client_ids=[str(i) for i in range(10)],
          clients_per_round=3,
          # Double the size every round
          rounds_to_double_cohort=1)
      expected_sizes = [3, 6, 10, 10, 10] + [10] * 100
      for round_num, expected_size in enumerate(expected_sizes):
        cohort = sampling_fn(round_num)
        self.assertLen(cohort, expected_size)
    with self.subTest('every_third_round'):
      sampling_fn = data_utils.create_sampling_fn(
          seed=1,
          client_ids=[str(i) for i in range(10)],
          clients_per_round=3,
          # Double the size every round
          rounds_to_double_cohort=3)
      expected_sizes = [3, 3, 3, 6, 6, 6, 10, 10, 10] + [10] * 100
      for round_num, expected_size in enumerate(expected_sizes):
        cohort = sampling_fn(round_num)
        self.assertLen(cohort, expected_size)

  def test_doubling_fails_invalid_argument(self):
    with self.subTest('non_integer'):
      with self.assertRaisesRegex(ValueError, 'positive integer'):
        data_utils.create_sampling_fn(
            seed=1,
            client_ids=[str(i) for i in range(10)],
            clients_per_round=5,
            # Double the size every round
            rounds_to_double_cohort=5.0)
    with self.subTest('non_positive'):
      with self.assertRaisesRegex(ValueError, 'positive integer'):
        data_utils.create_sampling_fn(
            seed=1,
            client_ids=[str(i) for i in range(10)],
            clients_per_round=5,
            # Double the size every round
            rounds_to_double_cohort=0)


if __name__ == '__main__':
  tf.test.main()
