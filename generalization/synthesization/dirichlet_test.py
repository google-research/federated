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
"""Tests for dirichlet."""

import collections
import itertools

from absl.testing import parameterized
import tensorflow as tf

from generalization.synthesization import dirichlet


class DirichletTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      (f'num_clients={num_clients}, rotate={rotate}', num_clients, rotate)
      for num_clients, rotate in itertools.product([1, 2, 3], [True, False]))
  def test_synthesize_by_dirichlet_over_labels(self, num_clients, rotate):
    test_dataset = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(
            x=list(range(9)), foo=['a', 'b', 'c'] * 3, label=[0, 1, 6] * 3))
    cd = dirichlet.synthesize_by_dirichlet_over_labels(
        dataset=test_dataset, num_clients=num_clients, use_rotate_draw=rotate)

    self.assertCountEqual(cd.client_ids, map(str, range(num_clients)))

    expected_num_elements_per_client = (9 // num_clients)

    for client_id in cd.client_ids:
      local_ds = cd.create_tf_dataset_for_client(client_id)
      self.assertLen(list(local_ds), expected_num_elements_per_client)


if __name__ == '__main__':
  tf.test.main()
