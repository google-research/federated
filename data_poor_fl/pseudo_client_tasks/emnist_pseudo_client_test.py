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

from absl.testing import absltest
import pandas as pd
import tensorflow_federated as tff

from data_poor_fl.pseudo_client_tasks import emnist_pseudo_client


class EmnistPseudoClientTest(absltest.TestCase):

  def test_pseudo_client_id_generation(self):
    data = dict(client_id=['A', 'B'], num_examples=[3, 5])
    df = pd.DataFrame(data=data)
    actual_pseudo_client_ids = emnist_pseudo_client._get_pseudo_client_ids(
        examples_per_pseudo_clients=2, base_client_examples_df=df)
    expected_pseudo_client_ids = ['A-0', 'A-1', 'B-0', 'B-1', 'B-2']
    self.assertEqual(actual_pseudo_client_ids, expected_pseudo_client_ids)

  def test_constructs_baseline_task(self):
    train_client_spec = tff.simulation.baselines.ClientSpec(
        num_epochs=1, batch_size=10)
    base_task = tff.simulation.baselines.emnist.create_character_recognition_task(
        train_client_spec=train_client_spec)
    pseudo_client_task = emnist_pseudo_client.build_task(
        base_task, examples_per_pseudo_client=10)
    self.assertIsInstance(pseudo_client_task,
                          tff.simulation.baselines.BaselineTask)


if __name__ == '__main__':
  absltest.main()
