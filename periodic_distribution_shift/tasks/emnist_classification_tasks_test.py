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
from unittest import mock

import tensorflow as tf
import tensorflow_federated as tff

from periodic_distribution_shift.tasks import dist_shift_task
from periodic_distribution_shift.tasks import emnist_classification_tasks


class CreateCharacterRecognitionModelTest(tf.test.TestCase):

  @mock.patch.object(
      emnist_classification_tasks,
      'create_single_branch_cnn_model',
      wraps=emnist_classification_tasks.create_single_branch_cnn_model)
  def test_get_character_recognition_model_constructs_single_branch_cnn(
      self, mock_model_builder):
    emnist_classification_tasks._get_character_recognition_model(
        model_id='single_branch_cnn')
    mock_model_builder.assert_called_once()

  @mock.patch.object(
      emnist_classification_tasks,
      'create_dual_branch_cnn_model',
      wraps=emnist_classification_tasks.create_dual_branch_cnn_model)
  def test_get_character_recognition_model_constructs_dual_branch_cnn(
      self, mock_model_builder):
    emnist_classification_tasks._get_character_recognition_model(
        model_id='dual_branch_cnn')
    mock_model_builder.assert_called_once()

  def test_raises_on_unsupported_model(self):
    with self.assertRaises(ValueError):
      emnist_classification_tasks._get_character_recognition_model(
          model_id='unsupported_model')


class CreateCharacterRecognitionTaskTest(tf.test.TestCase):

  def test_constructs_with_eval_client_spec(self):
    train_client_spec = tff.simulation.baselines.ClientSpec(
        num_epochs=2, batch_size=10, max_elements=3, shuffle_buffer_size=5)
    eval_client_spec = tff.simulation.baselines.ClientSpec(
        num_epochs=1, batch_size=2, max_elements=5, shuffle_buffer_size=10)
    baseline_task_spec = emnist_classification_tasks.create_character_recognition_task(
        train_client_spec,
        eval_client_spec=eval_client_spec,
        use_synthetic_data=True)
    self.assertIsInstance(baseline_task_spec, dist_shift_task.DistShiftTask)

  def test_constructs_with_no_eval_client_spec(self):
    train_client_spec = tff.simulation.baselines.ClientSpec(
        num_epochs=2, batch_size=10, max_elements=3, shuffle_buffer_size=5)
    baseline_task_spec = emnist_classification_tasks.create_character_recognition_task(
        train_client_spec, use_synthetic_data=True)
    self.assertIsInstance(baseline_task_spec, dist_shift_task.DistShiftTask)


if __name__ == '__main__':
  tf.test.main()
