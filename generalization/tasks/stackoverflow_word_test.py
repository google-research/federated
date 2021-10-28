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
"""Tests for stackoverflow_word."""

import tensorflow as tf
import tensorflow_federated as tff

from generalization.tasks import stackoverflow_word


class StackoverflowNWPTest(tf.test.TestCase):

  def test_train_client_ids_with_single_sample(self):
    singleton_client_ids = stackoverflow_word._TRAIN_CLIENT_IDS_WITH_SINGLE_ELEMENT

    train_cd, _, _ = tff.simulation.datasets.stackoverflow.load_data()

    # Test every 10 clients since loading all singleton clients is costly.
    for client_id in singleton_client_ids[::10]:
      self.assertLen(list(train_cd.create_tf_dataset_for_client(client_id)), 1)


if __name__ == '__main__':
  tf.test.main()
