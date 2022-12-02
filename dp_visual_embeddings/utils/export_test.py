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
"""Tests for model export."""

import tempfile

from absl import flags
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from dp_visual_embeddings.utils import export

FLAGS = flags.FLAGS

_INPUT_SIZE = 32
_INPUT_SHAPE = (_INPUT_SIZE,)
_HIDDEN_SIZE = 18
_OUTPUT_SIZE = 32


def _fake_keras_train_model():
  inputs = tf.keras.layers.Input(shape=_INPUT_SHAPE, name='input')
  hidden = tf.keras.layers.Dense(
      _HIDDEN_SIZE, activation='relu', use_bias=False, name='bottleneck')(
          inputs)
  non_trainable = tf.keras.layers.Dense(
      _HIDDEN_SIZE,
      activation='relu',
      use_bias=False,
      name='non_trainable',
  )(
      hidden)
  outputs = tf.keras.layers.Dense(
      _OUTPUT_SIZE,
      activation='relu',
      use_bias=False,
      name='output',
      trainable=False)(
          non_trainable)
  return tf.keras.models.Model(inputs=inputs, outputs=outputs)


def _fake_tff_train_model():
  keras_model = _fake_keras_train_model()
  loss = tf.keras.losses.MeanSquaredError()
  batch_input_shape = (None,) + _INPUT_SHAPE
  batch_output_shape = (None, _OUTPUT_SIZE)
  input_spec = (tf.TensorSpec(shape=batch_input_shape, dtype=tf.float32),
                tf.TensorSpec(shape=batch_output_shape, dtype=tf.float32))
  return tff.learning.from_keras_model(keras_model, loss, input_spec)


def _fake_keras_inference_model():
  inputs = tf.keras.layers.Input(shape=_INPUT_SHAPE, name='input')
  outputs = tf.keras.layers.Dense(
      _HIDDEN_SIZE, activation=None, use_bias=False, name='bottleneck')(
          inputs)
  return tf.keras.models.Model(inputs=inputs, outputs=outputs)


class ExportTest(tf.test.TestCase):

  def test_exports_fake_tff_model(self):
    first_layer_weights = np.random.rand(_INPUT_SIZE,
                                         _HIDDEN_SIZE).astype(np.float32)
    second_layer_weights = np.random.rand(_HIDDEN_SIZE,
                                          _HIDDEN_SIZE).astype(np.float32)
    third_layer_weights = np.random.rand(_HIDDEN_SIZE,
                                         _OUTPUT_SIZE).astype(np.float32)
    model_weights = tff.learning.ModelWeights(
        [first_layer_weights, third_layer_weights], [second_layer_weights])

    with tempfile.TemporaryDirectory(dir=FLAGS.test_tmpdir) as export_dir:
      export.export_state(_fake_tff_train_model, model_weights,
                          _fake_keras_inference_model(), export_dir)

      # Re-load the exported model, to check contents.
      reloaded_model = tf.keras.models.load_model(export_dir)
    self.assertIsInstance(reloaded_model, tf.keras.Model)
    self.assertLen(reloaded_model.trainable_weights, 1)
    self.assertEmpty(reloaded_model.non_trainable_weights)
    loaded_var = reloaded_model.trainable_weights[0]
    self.assertEqual(loaded_var.name, 'bottleneck/kernel:0')
    np.testing.assert_array_equal(loaded_var.value().numpy(),
                                  first_layer_weights)

  def test_exports_fake_keras_model(self):
    train_model = _fake_keras_train_model()
    export_model = _fake_keras_inference_model()
    with tempfile.TemporaryDirectory(dir=FLAGS.test_tmpdir) as export_dir:
      export.export_keras_model(
          train_model=train_model,
          export_model=export_model,
          export_dir=export_dir)

      # Re-load the exported model, to check contents.
      reloaded_model = tf.keras.models.load_model(export_dir)
    self.assertIsInstance(reloaded_model, tf.keras.Model)
    self.assertLen(reloaded_model.trainable_weights, 1)
    self.assertEmpty(reloaded_model.non_trainable_weights)
    loaded_var = reloaded_model.trainable_weights[0]
    self.assertEqual(loaded_var.name, 'bottleneck/kernel:0')

    train_var = train_model.trainable_weights[0]
    self.assertEqual(train_var.name, 'bottleneck/kernel:0')
    np.testing.assert_array_equal(loaded_var.value().numpy(),
                                  train_var.value().numpy())


if __name__ == '__main__':
  tf.test.main()
