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
"""Tests for task_utils."""
import collections
import os
from typing import Any, Optional
import unittest

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from dp_visual_embeddings.models import keras_utils
from dp_visual_embeddings.tasks import task_data
from dp_visual_embeddings.tasks import task_utils


_INPUT_SIZE = 32
_INPUT_SHAPE = (_INPUT_SIZE,)
_HIDDEN_SIZE = 18
_OUTPUT_SIZE = 32


def create_scalar_metrics():
  return collections.OrderedDict([
      ('a', {
          'b': 1.0,
          'c': 2.0,
      }),
  ])


def create_dataset_fn(client_id='5'):
  del client_id  # Unused.
  num_examples = 2
  x_array = np.random.rand(num_examples, _INPUT_SIZE)
  y_array = np.arange(num_examples, dtype=np.int64)

  def generate_examples():
    for x, y in zip(x_array, y_array):
      yield collections.OrderedDict(
          x=collections.OrderedDict(
              inputs=tf.constant(x),
              y=collections.OrderedDict(identity_indices=tf.constant(y))))

  output_signature = collections.OrderedDict(
      x=collections.OrderedDict(
          inputs=tf.TensorSpec(shape=(_INPUT_SIZE,), dtype=tf.float32)),
      y=collections.OrderedDict(
          identity_indices=tf.TensorSpec(shape=(), dtype=tf.int64)))
  # Create a dataset with 4 examples:
  dataset = tf.data.Dataset.from_generator(
      generate_examples, output_signature=output_signature)
  # Repeat the dataset 2 times with batches of 3 examples,
  # producing 3 minibatches (the last one with only 2 examples).
  return dataset.repeat(3).batch(2)


def create_client_data(num_clients):
  client_ids = [str(x) for x in range(num_clients)]
  return tff.simulation.datasets.ClientData.from_clients_and_tf_fn(
      client_ids, create_dataset_fn)


class FakeTask(task_utils.EmbeddingTask):
  """Task definition of a simple small model for testing."""

  def __init__(self, datasets: task_data.EmbeddingTaskDatasets):
    super().__init__('Fake', datasets)

  def keras_model_fn(self) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=_INPUT_SHAPE, name='input')
    hidden = tf.keras.layers.Dense(
        _HIDDEN_SIZE, activation='relu', use_bias=False, name='bottleneck')(
            inputs)
    outputs = tf.keras.layers.Dense(
        _OUTPUT_SIZE, activation=None, use_bias=False, name='output')(
            hidden)
    return tf.keras.models.Model(inputs=inputs, outputs=[hidden, outputs])

  def keras_model_and_variables_fn(
      self,
      model_output_size: Optional[int] = None) -> keras_utils.EmbeddingModel:
    base_model = self.keras_model_fn()
    global_variables = tff.learning.ModelWeights(
        trainable=base_model.trainable_variables,
        non_trainable=base_model.non_trainable_variables)
    client_variables = tff.learning.ModelWeights(trainable=[], non_trainable=[])
    return keras_utils.EmbeddingModel(
        model=base_model,
        global_variables=global_variables,
        client_variables=client_variables)

  @property
  def model_input_spec(self) -> collections.OrderedDict[str, Any]:
    """Specifies the shapes and types of input fields for the model."""
    return collections.OrderedDict(
        x=collections.OrderedDict(
            inputs=tf.TensorSpec(
                shape=(None, _INPUT_SIZE), dtype=tf.float32, name=None)),
        y=collections.OrderedDict(
            identity_indices=tf.TensorSpec(
                shape=(None,), dtype=tf.int64, name=None)))

  @property
  def inference_model(self) -> tf.keras.Model:
    return self.keras_model_fn()


def _fake_model_weights():
  first_layer_weights = np.random.rand(_INPUT_SIZE,
                                       _HIDDEN_SIZE).astype(np.float32)
  second_layer_weights = np.random.rand(_HIDDEN_SIZE,
                                        _OUTPUT_SIZE).astype(np.float32)
  return tff.learning.ModelWeights(
      trainable=[first_layer_weights, second_layer_weights], non_trainable=[])


class ManagerTest(unittest.IsolatedAsyncioTestCase, tf.test.TestCase):

  async def test_program_state_manager_saves_to_correct_dir(self):
    train_data = create_client_data(5)
    test_data = create_dataset_fn()
    fake_datasets = task_data.EmbeddingTaskDatasets(
        train_data=train_data, validation_data=test_data, test_data=test_data)
    task = FakeTask(datasets=fake_datasets)

    root_output_dir = self.get_temp_dir()
    experiment_name = 'test'

    def fake_export_fn(server_state, export_dir):
      del server_state  # Unused.
      os.makedirs(export_dir)
      with open(os.path.join(export_dir, 'fake_file'), 'w') as f:
        f.write('Contents')

    program_state_manager = task.configure_federated_checkpoint_manager(
        root_output_dir, experiment_name, fake_export_fn, rounds_per_export=10)
    self.assertIsInstance(program_state_manager,
                          tff.program.FileProgramStateManager)

    server_state = tff.learning.framework.ServerState(
        model=_fake_model_weights(),
        optimizer_state=[],
        delta_aggregate_state=[],
        model_broadcast_state=[])
    await program_state_manager.save(server_state, 10)
    expected_state_dir = os.path.join(root_output_dir, 'checkpoints',
                                      experiment_name, 'program_state_10')
    self.assertTrue(os.path.isdir(expected_state_dir))

    expected_export_dir = os.path.join(root_output_dir, 'export',
                                       experiment_name, 'inference_000010')
    self.assertTrue(
        os.path.isfile(os.path.join(expected_export_dir, 'fake_file')))

  def test_tensorboard_manager_saves_to_correct_dir(self):
    root_output_dir = self.get_temp_dir()
    experiment_name = 'test'
    metrics_managers = task_utils._configure_release_managers(
        root_output_dir, experiment_name)
    _, tensorboard_manager = metrics_managers
    self.assertIsInstance(tensorboard_manager,
                          tff.program.TensorBoardReleaseManager)


if __name__ == '__main__':
  tf.test.main()
