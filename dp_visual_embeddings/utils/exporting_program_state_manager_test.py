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
"""Tests for exporting_program_state_manager."""

import os
import tempfile
import unittest

from absl import flags
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from dp_visual_embeddings.utils import export
from dp_visual_embeddings.utils import exporting_program_state_manager

FLAGS = flags.FLAGS

_INPUT_SIZE = 32
_INPUT_SHAPE = (_INPUT_SIZE,)
_HIDDEN_SIZE = 18
_OUTPUT_SIZE = 32


def _fake_keras_model():
  inputs = tf.keras.layers.Input(shape=_INPUT_SHAPE, name='input')
  hidden = tf.keras.layers.Dense(
      _HIDDEN_SIZE, activation='relu', use_bias=False, name='bottleneck')(
          inputs)
  outputs = tf.keras.layers.Dense(
      _OUTPUT_SIZE, activation=None, use_bias=False, name='output')(
          hidden)
  return tf.keras.models.Model(inputs=inputs, outputs=outputs)


def _fake_tff_model():
  keras_model = _fake_keras_model()
  loss = tf.keras.losses.MeanSquaredError()
  batch_input_shape = (None,) + _INPUT_SHAPE
  batch_output_shape = (None, _OUTPUT_SIZE)
  input_spec = (tf.TensorSpec(shape=batch_input_shape, dtype=tf.float32),
                tf.TensorSpec(shape=batch_output_shape, dtype=tf.float32))
  return tff.learning.from_keras_model(keras_model, loss, input_spec)


def _fake_model_weights():
  first_layer_weights = np.random.rand(_INPUT_SIZE,
                                       _HIDDEN_SIZE).astype(np.float32)
  second_layer_weights = np.random.rand(_HIDDEN_SIZE,
                                        _OUTPUT_SIZE).astype(np.float32)
  return tff.learning.ModelWeights(
      trainable=[first_layer_weights, second_layer_weights], non_trainable=[])


class ExportingProgramStateManagerTest(unittest.IsolatedAsyncioTestCase,
                                       tf.test.TestCase):

  async def test_exports_inference_saved_model(self):
    with tempfile.TemporaryDirectory(dir=FLAGS.test_tmpdir) as root_dir:
      checkpoint_dir = os.path.join(root_dir, 'checkpoints')
      export_dir = os.path.join(root_dir, 'exports')

      def _export_fn(server_state, export_dir):
        export.export_state(_fake_tff_model, server_state.model,
                            _fake_keras_model(), export_dir)

      state_manager = (
          exporting_program_state_manager.ExportingProgramStateManager(
              checkpoint_dir,
              _export_fn,
              rounds_per_export=10,
              export_root_dir=export_dir))

      model_weights = _fake_model_weights()
      server_state = tff.learning.framework.ServerState(
          model=model_weights,
          optimizer_state=[],
          delta_aggregate_state=[],
          model_broadcast_state=[])
      await state_manager.save(server_state, 10)

      expected_save_dir = os.path.join(export_dir, 'inference_000010')
      self.assertTrue(os.path.isdir(expected_save_dir))

      # Re-load the exported model, to check contents.
      reloaded_model = tf.keras.models.load_model(expected_save_dir)
      self.assertIsInstance(reloaded_model, tf.keras.Model)
      self.assertLen(reloaded_model.trainable_weights, 2)
      loaded_var = reloaded_model.trainable_weights[0]
      self.assertEqual(loaded_var.name, 'bottleneck/kernel:0')
      np.testing.assert_array_equal(loaded_var.value().numpy(),
                                    model_weights.trainable[0])

      # Test the saving of ServerState, from the base FileProgramStateManager
      # class, still functions with a round trip.
      expected_state_dir = os.path.join(checkpoint_dir, 'program_state_10')
      self.assertTrue(os.path.isdir(expected_state_dir))
      reloaded_state = await state_manager.load(10, structure=server_state)
      self.assertIsInstance(reloaded_state, tff.learning.framework.ServerState)
      self.assertLen(reloaded_state.model.trainable, 2)
      np.testing.assert_array_equal(reloaded_state.model.trainable[0],
                                    model_weights.trainable[0])

  async def test_skips_export_before_interval(self):
    with tempfile.TemporaryDirectory(dir=FLAGS.test_tmpdir) as root_dir:
      checkpoint_dir = os.path.join(root_dir, 'checkpoints')
      export_dir = os.path.join(root_dir, 'exports')

      def _export_fn(server_state, export_dir):
        export.export_state(_fake_tff_model, server_state.model,
                            _fake_keras_model(), export_dir)

      state_manager = (
          exporting_program_state_manager.ExportingProgramStateManager(
              checkpoint_dir,
              _export_fn,
              rounds_per_export=10,
              export_root_dir=export_dir))

      model_weights = _fake_model_weights()
      server_state = tff.learning.framework.ServerState(
          model=model_weights,
          optimizer_state=[],
          delta_aggregate_state=[],
          model_broadcast_state=[])
      await state_manager.save(server_state, 18)

      # Not expecting a model to be saved, because it should only save every 10
      # versions.
      expected_save_dir = os.path.join(export_dir, 'inference_000018')
      self.assertFalse(os.path.isdir(expected_save_dir))


if __name__ == '__main__':
  tf.test.main()
