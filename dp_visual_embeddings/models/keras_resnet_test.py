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
"""Tests for ResNet model."""
from absl.testing import parameterized

import tensorflow as tf
import tensorflow_federated as tff

from dp_visual_embeddings.models import keras_resnet
from dp_visual_embeddings.models import keras_utils


class ResnetModelTest(tf.test.TestCase, parameterized.TestCase):

  def test_resnet50_embedding_has_expected_inputs(self):
    keras_model = keras_resnet.resnet50_image2embedding()
    self.assertIsInstance(keras_model, tf.keras.Model)
    self.assertLen(keras_model.inputs, 1)
    model_input = keras_model.inputs[0]
    self.assertSequenceEqual(model_input.shape, (None, 224, 224, 3))
    self.assertEqual(model_input.dtype, tf.float32)

  def test_resnet50_embedding_has_expected_outputs(self):
    keras_model = keras_resnet.resnet50_image2embedding(embedding_dim_size=107)
    self.assertIsInstance(keras_model, tf.keras.Model)
    self.assertLen(keras_model.outputs, 1)
    embedding_output = keras_model.outputs[0]
    self.assertSequenceEqual(embedding_output.shape, (None, 107))
    self.assertEqual(embedding_output.dtype, tf.float32)

  def test_resnet50_classification_has_expected_inputs(self):
    backbone_model = keras_resnet.resnet50_with_head(num_classes=1000)
    self.assertIsInstance(backbone_model, keras_utils.EmbeddingModel)

    keras_model = backbone_model.model
    self.assertIsInstance(keras_model, tf.keras.Model)
    self.assertLen(keras_model.inputs, 1)
    model_input = keras_model.inputs[0]
    self.assertSequenceEqual(model_input.shape, (None, 224, 224, 3))
    self.assertEqual(model_input.dtype, tf.float32)

  def test_resnet50_classification_has_expected_outputs(self):
    backbone_model = keras_resnet.resnet50_with_head(num_classes=283)
    self.assertIsInstance(backbone_model, keras_utils.EmbeddingModel)

    keras_model = backbone_model.model
    self.assertIsInstance(keras_model, tf.keras.Model)
    self.assertLen(keras_model.outputs, 2)
    # Classification output.
    self.assertSequenceEqual(keras_model.outputs[0].shape, (None, 283))
    self.assertEqual(keras_model.outputs[0].dtype, tf.float32)
    # Embeddings output.
    self.assertSequenceEqual(keras_model.outputs[1].shape, (None, 128))
    self.assertEqual(keras_model.outputs[1].dtype, tf.float32)

  def test_resnet50_classification_runs_forward_pass(self):
    model = keras_resnet.resnet50_with_head(num_classes=151).model

    fake_input = tf.random.normal(shape=(2, 224, 224, 3))
    classifications, embeddings = model(fake_input)

    self.assertSequenceEqual(classifications.shape, (2, 151))
    self.assertSequenceEqual(embeddings.shape, (2, 128))

  def test_returns_global_and_client_variables(self):
    backbone_model = keras_resnet.resnet50_with_head(
        num_classes=227, use_normalize=False)

    global_variables = backbone_model.global_variables
    self.assertIsInstance(global_variables, tff.learning.ModelWeights)
    self.assertNotEmpty(global_variables.trainable)
    self.assertIsInstance(global_variables.trainable[0], tf.Variable)

    client_variables = backbone_model.client_variables
    self.assertIsInstance(client_variables, tff.learning.ModelWeights)
    self.assertLen(client_variables.trainable, 1)
    self.assertIsInstance(client_variables.trainable[0], tf.Variable)
    # Final dense layer is embedding_dim x num_classes.
    self.assertSequenceEqual(client_variables.trainable[0].shape, (128, 227))

  def test_returns_global_and_client_variables_normalize(self):
    backbone_model = keras_resnet.resnet50_with_head(
        num_classes=227, use_normalize=True)

    global_variables = backbone_model.global_variables
    self.assertIsInstance(global_variables, tff.learning.ModelWeights)
    self.assertNotEmpty(global_variables.trainable)
    self.assertIsInstance(global_variables.trainable[0], tf.Variable)

    client_variables = backbone_model.client_variables
    self.assertIsInstance(client_variables, tff.learning.ModelWeights)
    self.assertLen(client_variables.trainable, 2)
    self.assertIsInstance(client_variables.trainable[0], tf.Variable)
    # Final dense layer is embedding_dim x num_classes.
    self.assertSequenceEqual(client_variables.trainable[0].shape, (128, 227))

  def test_normalize_embedding(self):
    model = keras_resnet.resnet50_image2embedding()
    batch_size = 2
    synthetic_batch = tf.random.uniform(
        tf.TensorShape((batch_size,) + keras_resnet.INPUT_SHAPE),
        dtype=tf.float32)
    embeddings = model(synthetic_batch, training=False)
    norms = tf.norm(embeddings, ord='euclidean', axis=1)
    self.assertAllClose(norms, tf.ones([batch_size]))

  def test_unnormalize_embedding(self):
    model = keras_resnet.resnet50_image2embedding()
    batch_size = 3
    synthetic_batch = tf.random.uniform(
        tf.TensorShape((batch_size,) + keras_resnet.INPUT_SHAPE),
        dtype=tf.float32)
    embeddings = model(synthetic_batch, training=True)
    norms = tf.norm(embeddings, ord='euclidean', axis=1)
    self.assertNotAllClose(norms, tf.ones([batch_size]))

  @parameterized.named_parameters(('global_avg_pool', 2048, True),
                                  ('avg32_pool', 3 * 3 * 2048, False))
  def test_resnet50_pooling_size(self, expect_size, global_averaging_pooling):
    embed_size = 107
    keras_model = keras_resnet.resnet50_image2embedding(
        embedding_dim_size=embed_size,
        global_averaging_pooling=global_averaging_pooling)
    self.assertAllEqual(
        tf.shape(keras_model.layers[-2].trainable_variables[0]),
        [expect_size, embed_size])

  def test_trainable_conv(self):
    backbone_model = keras_resnet.resnet50_with_head(
        num_classes=27, trainable_conv=False)
    full_backbone_model = keras_resnet.resnet50_with_head(
        num_classes=27, trainable_conv=True)

    global_variables = backbone_model.global_variables
    self.assertIsInstance(global_variables, tff.learning.ModelWeights)
    self.assertNotEmpty(global_variables.trainable)
    self.assertNotEmpty(global_variables.non_trainable)
    self.assertEmpty(full_backbone_model.global_variables.non_trainable)
    self.assertEqual(
        len(global_variables.trainable + global_variables.non_trainable),
        len(full_backbone_model.global_variables.trainable))

    client_variables = backbone_model.client_variables
    self.assertIsInstance(client_variables, tff.learning.ModelWeights)
    self.assertLen(client_variables.trainable, 1)
    self.assertIsInstance(client_variables.trainable[0], tf.Variable)
    # Final dense layer is embedding_dim x num_classes.
    self.assertSequenceEqual(client_variables.trainable[0].shape, (128, 27))

    tf.nest.map_structure(lambda v, w: v.assign(w),
                          backbone_model.model.weights,
                          full_backbone_model.model.weights)
    self.assertAllEqual(client_variables.trainable[0],
                        full_backbone_model.client_variables.trainable[0])


if __name__ == '__main__':
  tf.test.main()
