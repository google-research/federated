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
"""Tests for build_task."""

import tensorflow as tf

from dp_visual_embeddings.tasks import build_task
from dp_visual_embeddings.tasks import task_utils


class BuildTaskTest(tf.test.TestCase):

  def test_get_task_types(self):
    self.assertListEqual(list(build_task.TaskType), build_task.get_task_types())

  def test_build_task(self):
    task = build_task.configure_task(
        task_type=build_task.TaskType.EMNIST,
        client_batch_size=4,
        eval_batch_size=7,
        train_max_examples_per_client=4)
    self.assertIsInstance(task, task_utils.EmbeddingTask)
    self.assertIsInstance(task.inference_model, tf.keras.Model)


if __name__ == '__main__':
  tf.test.main()
