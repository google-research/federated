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

  def test_element_type_structure_preserved(self):
    raw_client_data = tff.simulation.datasets.emnist.get_synthetic()
    inf_client_data = infinite_emnist.get_infinite(raw_client_data, 5)
    self.assertEqual(raw_client_data.element_type_structure,
                     inf_client_data.element_type_structure)

  def test_pseudo_client_count(self):
    raw_client_data = tff.simulation.datasets.emnist.get_synthetic()
    self.assertLen(raw_client_data.client_ids, 1)
    inf_client_data = infinite_emnist.get_infinite(raw_client_data, 10)
    self.assertLen(inf_client_data.client_ids, 10)

  def test_first_pseudo_client_preserves_original(self):
    raw_client_data = tff.simulation.datasets.emnist.get_synthetic()
    inf_client_data = infinite_emnist.get_infinite(raw_client_data, 5)
    raw_dataset = raw_client_data.dataset_computation(
        raw_client_data.client_ids[0])
    inf_dataset = inf_client_data.dataset_computation(
        inf_client_data.client_ids[0])
    length1 = _compute_dataset_length(raw_dataset)
    length2 = _compute_dataset_length(inf_dataset)
    self.assertEqual(length1, length2)
    for raw_batch, inf_batch in zip(raw_dataset, inf_dataset):
      self.assertAllClose(raw_batch, inf_batch)

  def test_transform_modifies_data(self):
    data = infinite_emnist.get_infinite(
        tff.simulation.datasets.emnist.get_synthetic(), 3)
    datasets = [data.dataset_computation(id) for id in data.client_ids]
    lengths = [_compute_dataset_length(datasets[i]) for i in [0, 1, 2]]
    self.assertEqual(lengths[0], lengths[1])
    self.assertEqual(lengths[1], lengths[2])
    for batch0, batch1, batch2 in zip(datasets[0], datasets[1], datasets[2]):
      self.assertNotAllClose(batch0, batch1)
      self.assertNotAllClose(batch1, batch2)

  def test_dataset_computation_equals_create_tf_dataset(self):
    synth_data = tff.simulation.datasets.emnist.get_synthetic()
    data = infinite_emnist.get_infinite(synth_data, 3)

    for client_id in data.client_ids:
      comp_dataset = data.dataset_computation(client_id)
      create_tf_dataset = data.create_tf_dataset_for_client(client_id)
      for batch1, batch2 in zip(comp_dataset, create_tf_dataset):
        # For some reason it appears tf.quantization.quantize_and_dequantize
        # sometimes (very rarely-- on one pixel for this test) gives results
        # that differ by a single bit between the serialized and the
        # non-serialized versions. Hence we use atol just larger than 1 bit.
        self.assertAllClose(batch1['pixels'], batch2['pixels'], atol=1.5 / 255)


if __name__ == '__main__':
  tf.test.main()
