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
import tempfile

from absl.testing import flagsaver
from absl.testing import parameterized
import tensorflow as tf

from personalization_benchmark.cross_device import hypcluster_trainer


class GlobalTrainerTest(tf.test.TestCase, parameterized.TestCase):

  @flagsaver.flagsaver(
      root_output_dir=tempfile.mkdtemp(),
      experiment_name='test_experiment',
      dataset_name='emnist',
      clients_per_train_round=1,
      num_clusters=2,
      total_rounds=2,
      rounds_per_evaluation=1,
      rounds_per_checkpoint=1,
      client_optimizer='sgd',
      client_learning_rate=0.01,
      server_optimizer='adam',
      server_learning_rate=1.0,
      valid_clients_per_evaluation=1,
      test_clients_per_evaluation=1,
      use_synthetic_data=True)
  def test_executes_emnist(self):
    hypcluster_trainer.main([])

  @flagsaver.flagsaver(
      root_output_dir=tempfile.mkdtemp(),
      experiment_name='test_experiment',
      dataset_name='stackoverflow',
      clients_per_train_round=1,
      num_clusters=2,
      total_rounds=2,
      rounds_per_evaluation=1,
      rounds_per_checkpoint=1,
      client_optimizer='sgd',
      client_learning_rate=0.01,
      server_optimizer='adam',
      server_learning_rate=1.0,
      valid_clients_per_evaluation=1,
      test_clients_per_evaluation=1,
      use_synthetic_data=True)
  def test_executes_stackoverflow(self):
    hypcluster_trainer.main([])

  @flagsaver.flagsaver(
      root_output_dir=tempfile.mkdtemp(),
      experiment_name='test_experiment',
      dataset_name='landmark',
      clients_per_train_round=1,
      num_clusters=2,
      total_rounds=2,
      rounds_per_evaluation=1,
      rounds_per_checkpoint=1,
      client_optimizer='sgd',
      client_learning_rate=0.01,
      server_optimizer='adam',
      server_learning_rate=1.0,
      valid_clients_per_evaluation=1,
      test_clients_per_evaluation=1,
      use_synthetic_data=True)
  def test_executes_landmark(self):
    hypcluster_trainer.main([])

  @flagsaver.flagsaver(
      root_output_dir=tempfile.mkdtemp(),
      experiment_name='test_experiment',
      dataset_name='ted_multi',
      clients_per_train_round=1,
      num_clusters=2,
      total_rounds=2,
      rounds_per_evaluation=1,
      rounds_per_checkpoint=1,
      client_optimizer='sgd',
      client_learning_rate=0.01,
      server_optimizer='sgd',
      server_learning_rate=1.0,
      valid_clients_per_evaluation=1,
      test_clients_per_evaluation=1,
      use_synthetic_data=True)
  def test_executes_ted_multi(self):
    hypcluster_trainer.main([])


if __name__ == '__main__':
  tf.test.main()
