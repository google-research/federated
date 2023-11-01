# Copyright 2023, Google LLC.
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
"""Tests for canary_insertion."""

import collections

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_federated as tff

from one_shot_epe import canary_insertion


class CanaryInsertionTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('string', ('foo',)),
      ('float', (2.71828,)),
      ('string_and_float', ('bar', 3.14159)),
  )
  def test_canary_insertion(self, example):
    num_canary_clients = 9
    num_examples = 3

    data = collections.OrderedDict(
        ('feature_' + str(i), [element] * num_examples)
        for i, element in enumerate(example)
    )

    real_client_ids = ['alice', 'bob', 'carol', 'dennis', 'evelyn']
    client_data = tff.simulation.datasets.TestClientData(
        {client_id: data for client_id in real_client_ids}
    )
    client_data = canary_insertion.add_canaries(client_data, num_canary_clients)

    canary_suffixes = set()
    real_suffixes = set()

    for client_id in client_data.client_ids:
      client_tf_dataset = client_data.dataset_computation(client_id)
      example_count = client_tf_dataset.cardinality()

      prefix, suffix = client_id.split(':', maxsplit=1)
      if prefix == 'canary':
        self.assertEqual(example_count, 0)
        canary_suffixes.add(suffix)
      else:
        self.assertEqual(prefix, 'real')
        self.assertEqual(example_count, num_examples)
        real_suffixes.add(suffix)

    self.assertEqual(
        canary_suffixes, set(str(i) for i in range(num_canary_clients))
    )
    self.assertEqual(real_suffixes, set(real_client_ids))

    dataset_from_all_clients = client_data.create_tf_dataset_from_all_clients()
    total_examples = sum(1 for _ in dataset_from_all_clients)
    self.assertEqual(len(real_client_ids) * num_examples, total_examples)

  def test_raises_on_data_without_fully_defined_shape(self):
    num_real = 2
    num_canary = 3

    @tf.function
    def get_data(x):
      del x  # Unused.
      return tf.data.Dataset.from_tensor_slices(
          [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
      ).batch(2)

    client_data = tff.simulation.datasets.TestClientData.from_clients_and_tf_fn(
        [str(i) for i in range(num_real)], get_data
    )

    with self.assertRaisesRegex(
        ValueError, 'Tensor shape must be fully defined'
    ):
      canary_insertion.add_canaries(client_data, num_canary)

  def test_canary_insertion_batched(self):
    num_real = 2
    num_canary = 3

    @tf.function
    def get_data(x):
      del x  # Unused.
      return tf.data.Dataset.from_tensor_slices(
          [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
      )

    client_data = tff.simulation.datasets.TestClientData.from_clients_and_tf_fn(
        [str(i) for i in range(num_real)], get_data
    )
    client_data = canary_insertion.add_canaries(client_data, num_canary)

    client_data = client_data.preprocess(lambda dataset: dataset.batch(2))

    real_count = sum(1 for c in client_data.client_ids if c.startswith('real'))
    canary_count = sum(
        1 for c in client_data.client_ids if c.startswith('canary')
    )
    self.assertEqual(real_count, num_real)
    self.assertEqual(canary_count, num_canary)

  def test_error_on_negative_num_canaries(self):
    data = tff.simulation.datasets.TestClientData({'the_client_id': [0]})
    with self.assertRaisesRegex(ValueError, 'canaries must be non-negative'):
      canary_insertion.add_canaries(data, -1)


if __name__ == '__main__':
  tf.test.main()
