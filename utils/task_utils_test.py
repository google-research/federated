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

from absl import flags
from absl.testing import flagsaver
from absl.testing import parameterized
import tensorflow as tf
import tensorflow_federated as tff

from utils import task_utils

FLAGS = flags.FLAGS


def setUpModule():
  # Create flags here to ensure duplicate flags are not created.
  task_utils.define_task_flags()


TASKS_TO_TEST = [
    (task_name, task_name) for task_name in task_utils.SUPPORTED_TASKS
]


class TaskUtilsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(TASKS_TO_TEST)
  @flagsaver.flagsaver
  def test_create_task_from_flags(self, task_name):
    FLAGS.task = task_name
    train_client_spec = tff.simulation.baselines.ClientSpec(
        num_epochs=1, batch_size=10)
    actual_task = task_utils.create_task_from_flags(
        train_client_spec, use_synthetic_data=True)
    self.assertIsInstance(actual_task, tff.simulation.baselines.BaselineTask)
    expected_task = task_utils.TASK_CONSTRUCTORS[task_name](
        train_client_spec, use_synthetic_data=True)
    self.assertEqual(actual_task.datasets.element_type_structure,
                     expected_task.datasets.element_type_structure)

  @flagsaver.flagsaver
  def test_create_cifar100_image_uses_crop_sizes(self):
    crop_height = 7
    crop_width = 17
    FLAGS.task = 'cifar100_image'
    FLAGS.cifar100_image_crop_height = crop_height
    FLAGS.cifar100_image_crop_width = crop_width

    batch_size = 5
    train_client_spec = tff.simulation.baselines.ClientSpec(
        num_epochs=1, batch_size=batch_size)
    task = task_utils.create_task_from_flags(
        train_client_spec, use_synthetic_data=True)
    sample_batch = next(iter(task.datasets.get_centralized_test_data()))
    self.assertEqual(sample_batch[0].shape,
                     (batch_size, crop_height, crop_width, 3))

  @flagsaver.flagsaver
  def test_create_stackoverflow_tag_uses_vocab_sizes(self):
    word_vocab_size = 5
    tag_vocab_size = 3
    FLAGS.task = 'stackoverflow_tag'
    FLAGS.stackoverflow_tag_word_vocab_size = word_vocab_size
    FLAGS.stackoverflow_tag_tag_vocab_size = tag_vocab_size

    train_client_spec = tff.simulation.baselines.ClientSpec(
        num_epochs=1, batch_size=10)
    task = task_utils.create_task_from_flags(
        train_client_spec, use_synthetic_data=True)
    model = task.model_fn()
    model_weights = tff.learning.ModelWeights.from_model(model).trainable
    self.assertLen(model_weights, 2)
    self.assertEqual(model_weights[0].shape, (word_vocab_size, tag_vocab_size))
    self.assertEqual(model_weights[1].shape, (tag_vocab_size,))

  @flagsaver.flagsaver
  def test_raises_on_empty_task_flag(self):
    train_client_spec = tff.simulation.baselines.ClientSpec(
        num_epochs=1, batch_size=10)
    with self.assertRaises(ValueError):
      task_utils.create_task_from_flags(train_client_spec)

  @flagsaver.flagsaver
  def test_raises_on_nondefault_flags_for_multiple_tasks(self):
    FLAGS.task = 'emnist_autoencoder'
    FLAGS.emnist_character_only_digits = True
    train_client_spec = tff.simulation.baselines.ClientSpec(
        num_epochs=1, batch_size=10)
    with self.assertRaises(ValueError):
      task_utils.create_task_from_flags(train_client_spec)


if __name__ == '__main__':
  tf.test.main()
