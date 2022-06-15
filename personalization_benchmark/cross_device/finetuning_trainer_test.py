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

from absl.testing import absltest
from absl.testing import flagsaver

from personalization_benchmark.cross_device import finetuning_trainer


class GlobalTrainerTest(absltest.TestCase):

  @flagsaver.flagsaver(
      root_output_dir=tempfile.mkdtemp(),
      experiment_name='test_experiment',
      dataset_name='emnist',
      clients_per_train_round=1,
      total_rounds=2,
      rounds_per_evaluation=1,
      rounds_per_checkpoint=1,
      client_optimizer='sgd',
      client_learning_rate=0.01,
      server_optimizer='sgd',
      server_learning_rate=1.0,
      finetune_optimizer='sgd',
      finetune_learning_rate=0.01,
      finetune_max_epochs=2,
      valid_clients_per_evaluation=1,
      test_clients_per_evaluation=1,
      use_synthetic_data=True)
  def test_executes_emnist(self):
    finetuning_trainer.main([])

  @flagsaver.flagsaver(
      root_output_dir=tempfile.mkdtemp(),
      experiment_name='test_experiment',
      dataset_name='stackoverflow',
      clients_per_train_round=1,
      total_rounds=2,
      rounds_per_evaluation=1,
      rounds_per_checkpoint=1,
      client_optimizer='sgd',
      client_learning_rate=0.01,
      server_optimizer='sgd',
      server_learning_rate=1.0,
      finetune_optimizer='sgd',
      finetune_learning_rate=0.01,
      finetune_max_epochs=2,
      valid_clients_per_evaluation=1,
      test_clients_per_evaluation=1,
      use_synthetic_data=True)
  def test_executes_stackoverflow(self):
    finetuning_trainer.main([])

  @flagsaver.flagsaver(
      root_output_dir=tempfile.mkdtemp(),
      experiment_name='test_experiment',
      dataset_name='landmark',
      landmark_extra_test_over_original_test_ratio=0.5,
      clients_per_train_round=1,
      total_rounds=2,
      rounds_per_evaluation=1,
      rounds_per_checkpoint=1,
      client_optimizer='sgd',
      client_learning_rate=0.01,
      server_optimizer='sgd',
      server_learning_rate=1.0,
      finetune_optimizer='sgd',
      finetune_learning_rate=0.01,
      finetune_max_epochs=2,
      valid_clients_per_evaluation=1,
      test_clients_per_evaluation=1,
      use_synthetic_data=True)
  def test_executes_landmark(self):
    finetuning_trainer.main([])

  @flagsaver.flagsaver(
      root_output_dir=tempfile.mkdtemp(),
      experiment_name='test_experiment',
      dataset_name='ted_multi',
      clients_per_train_round=1,
      total_rounds=2,
      rounds_per_evaluation=1,
      rounds_per_checkpoint=1,
      client_optimizer='sgd',
      client_learning_rate=0.01,
      server_optimizer='sgd',
      server_learning_rate=1.0,
      finetune_optimizer='sgd',
      finetune_learning_rate=0.01,
      finetune_max_epochs=2,
      valid_clients_per_evaluation=1,
      test_clients_per_evaluation=1,
      use_synthetic_data=True)
  def test_executes_ted_multi(self):
    finetuning_trainer.main([])


if __name__ == '__main__':
  absltest.main()
