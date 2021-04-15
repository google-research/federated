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

import tensorflow as tf
import tensorflow_federated as tff

from utils.datasets import infinite_emnist


def _compute_dataset_length(dataset):
  return dataset.reduce(0, lambda x, _: x + 1)


class InfiniteEmnistTest(tf.test.TestCase):

  def test_get_infinite_creates_transforming_client_data(self):
    raw_client_data = tff.simulation.datasets.emnist.get_synthetic()
    inf_client_data = infinite_emnist.get_infinite(
        raw_client_data, num_pseudo_clients=2)
    self.assertIsInstance(inf_client_data,
                          tff.simulation.datasets.TransformingClientData)

  def test_get_infinite_preserves_element_type_structure(self):
    raw_client_data = tff.simulation.datasets.emnist.get_synthetic()
    inf_client_data = infinite_emnist.get_infinite(
        raw_client_data, num_pseudo_clients=5)
    self.assertEqual(raw_client_data.element_type_structure,
                     inf_client_data.element_type_structure)

  def test_get_infinite_creates_pseudo_clients(self):
    raw_client_data = tff.simulation.datasets.emnist.get_synthetic()
    self.assertLen(raw_client_data.client_ids, 1)
    inf_client_data = infinite_emnist.get_infinite(
        raw_client_data, num_pseudo_clients=10)
    self.assertLen(inf_client_data.client_ids, 10)

  def test_get_infinite_preserves_original_client(self):
    raw_client_data = tff.simulation.datasets.emnist.get_synthetic()
    self.assertLen(raw_client_data.client_ids, 1)
    raw_dataset = raw_client_data.create_tf_dataset_for_client(
        raw_client_data.client_ids[0])
    inf_client_data = infinite_emnist.get_infinite(
        raw_client_data, num_pseudo_clients=1)
    self.assertLen(inf_client_data.client_ids, 1)
    inf_dataset = inf_client_data.create_tf_dataset_for_client(
        inf_client_data.client_ids[0])
    length1 = _compute_dataset_length(raw_dataset)
    length2 = _compute_dataset_length(inf_dataset)
    self.assertEqual(length1, length2)
    raw_dataset_iter = iter(raw_dataset)
    inf_dataset_iter = iter(inf_dataset)
    for _ in range(int(length1)):
      raw_batch = next(raw_dataset_iter)
      inf_batch = next(inf_dataset_iter)
      self.assertAllClose(raw_batch, inf_batch)


if __name__ == '__main__':
  tf.test.main()
