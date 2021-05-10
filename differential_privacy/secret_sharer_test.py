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
"""Tests for secret_sharer."""

import collections
import string

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from differential_privacy import secret_sharer

TEST_SEED = 0xBAD5EED


def _construct_test_client_data(num_clients, num_examples, secrets, seed):
  np.random.seed(TEST_SEED)

  def random_string():
    return ''.join(np.random.choice(list(string.ascii_uppercase), 5))

  words = ['a' for _ in range(num_examples)]

  def data_for_client():
    return collections.OrderedDict(
        # We hash on date to seed client randomness, so they must be distinct.
        creation_date=[random_string() for _ in range(num_examples)],
        score=words,
        tags=words,
        title=words,
        tokens=words,
        type=words)

  client_data = tff.simulation.datasets.TestClientData(
      {random_string(): data_for_client() for _ in range(num_clients)})

  return secret_sharer.stackoverflow_with_secrets(client_data, secrets, seed)


class SecretSharerTest(tf.test.TestCase):

  def test_secret_insertion(self):
    if tf.config.list_logical_devices('GPU'):
      self.skipTest('skip GPU test')

    secrets = dict(b=(50, 0.6), c=(20, 1.0), d=(1, 1.0))
    num_clients = 100
    num_examples = 12
    client_data = _construct_test_client_data(num_clients, num_examples,
                                              secrets, TEST_SEED)

    secret_count = {secret: 0 for secret in secrets}
    client_with_secret_count = {secret: 0 for secret in secrets}

    for client_id in client_data.client_ids:
      dataset = client_data.dataset_computation(client_id)
      has_secret = None
      for example in dataset.enumerate():
        tokens = example[1]['tokens'].numpy().decode('utf-8')
        if tokens in secrets:
          secret_count[tokens] += 1
          has_secret = tokens
      if has_secret:
        client_with_secret_count[has_secret] += 1

    # The probability that there exists even a single client b that has no
    # insertion is an acceptable 1-(1-0.4^12)^50 < 1e-3. So we assert equal.
    self.assertAllEqual(50, client_with_secret_count['b'])
    self.assertAllEqual(20, client_with_secret_count['c'])
    self.assertAllEqual(1, client_with_secret_count['d'])

    # Count of secret b is Bin(600, 0.6) with mean 360, stddev 12.
    # Assert result within 40: over three stddevs of mean.
    self.assertAllClose(360, secret_count['b'], atol=40)
    self.assertAllEqual(20 * num_examples, secret_count['c'])
    self.assertAllEqual(num_examples, secret_count['d'])

  def test_dataset_computation_equals_create_tf_dataset(self):
    if tf.config.list_logical_devices('GPU'):
      self.skipTest('skip GPU test')

    secrets = dict(b=(5, 0.6), c=(4, 1.0), d=(3, 0.2), e=(2, 0.8), f=(1, 1.0))
    num_clients = 50
    num_examples = 12
    client_data = _construct_test_client_data(num_clients, num_examples,
                                              secrets, TEST_SEED)

    for client_id in client_data.client_ids:
      dataset_with_comp = client_data.dataset_computation(client_id)
      dataset_no_comp = client_data.create_tf_dataset_for_client(client_id)
      for x, y in zip(dataset_with_comp.enumerate(),
                      dataset_no_comp.enumerate()):
        self.assertAllEqual(x[1]['tokens'], y[1]['tokens'])

  def test_seed_used(self):
    # We will construct datasets with two seeds and test that a) at least one
    # client has different secrets in the two datasets and b) for some client
    # that has the same secret in both datasets, the chosen examples are
    # different. Note that the number of clients and examples have been chosen
    # so the probability of either of these things failing to hold is < 1e-3.

    if tf.config.list_logical_devices('GPU'):
      self.skipTest('skip GPU test')

    secrets = dict(b=(9, 0.5), c=(1, 1.0))
    num_clients = 20
    num_examples = 10
    client_data_1 = _construct_test_client_data(num_clients, num_examples,
                                                secrets, 1)
    client_data_2 = _construct_test_client_data(num_clients, num_examples,
                                                secrets, 2)

    self.assertCountEqual(client_data_1.client_ids, client_data_2.client_ids)

    def get_secret(dataset):
      for example in dataset.enumerate():
        tokens = example[1]['tokens'].numpy().decode('utf-8')
        if tokens in secrets:
          return tokens
      return None

    chosen_clients_differ = False
    chosen_examples_differ = False
    for client_id in client_data_1.client_ids:
      dataset_1 = client_data_1.dataset_computation(client_id)
      dataset_2 = client_data_2.dataset_computation(client_id)

      secret_1 = get_secret(dataset_1)
      secret_2 = get_secret(dataset_2)

      if secret_1 != secret_2:
        chosen_clients_differ = True
      elif secret_1 is not None:
        # This client has the same secret in both datasets.
        # Test that the selected examples are different.
        for x, y in zip(dataset_1.enumerate(), dataset_2.enumerate()):
          if not all(
              s == t
              for s, t in zip(x[1]['tokens'].numpy(), y[1]['tokens'].numpy())):
            chosen_examples_differ = True

    self.assertTrue(chosen_clients_differ)
    self.assertTrue(chosen_examples_differ)


if __name__ == '__main__':
  tf.test.main()
