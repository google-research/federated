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
"""Tests for model building."""

import tensorflow as tf

from dp_visual_embeddings.models import build_model
from dp_visual_embeddings.models import keras_utils


def _variable_names(variable_sequence):
  return [var.name.split(':')[0] for var in variable_sequence]


def _model_trainable_variable_names(keras_model):
  return _variable_names(keras_model.trainable_variables)


class BuildModelTest(tf.test.TestCase):

  def test_builds_mobilenet_embedding_model(self):
    mobilenet_model = build_model.embedding_model(
        build_model.ModelBackbone.MOBILENET2, input_shape=(128, 128, 3))
    self.assertIsInstance(mobilenet_model, tf.keras.Model)
    self.assertLen(mobilenet_model.inputs, 1)
    self.assertSequenceEqual(mobilenet_model.inputs[0].shape,
                             (None, 128, 128, 3))
    self.assertLen(mobilenet_model.outputs, 1)
    self.assertSequenceEqual(mobilenet_model.outputs[0].shape, (None, 128))
    self.assertIn('block_0_depthwise_conv/depthwise_kernel',
                  _model_trainable_variable_names(mobilenet_model))

  def test_builds_mobilenet_classification_model(self):
    mobilenet_backbone = build_model.classification_training_model(
        build_model.ModelBackbone.MOBILENET2,
        input_shape=(224, 224, 3),
        num_identities=233)
    self.assertIsInstance(mobilenet_backbone, keras_utils.EmbeddingModel)
    self.assertLen(mobilenet_backbone.model.inputs, 1)
    self.assertSequenceEqual(mobilenet_backbone.model.inputs[0].shape,
                             (None, 224, 224, 3))
    self.assertIn('block_0_depthwise_conv/depthwise_kernel',
                  _model_trainable_variable_names(mobilenet_backbone.model))
    self.assertLen(mobilenet_backbone.model.outputs, 2)
    # Classification output.
    self.assertSequenceEqual(mobilenet_backbone.model.outputs[0].shape,
                             (None, 233))
    # Embedding output.
    self.assertSequenceEqual(mobilenet_backbone.model.outputs[1].shape,
                             (None, 128))
    self.assertIn(
        'block_0_depthwise_conv/depthwise_kernel',
        _variable_names(mobilenet_backbone.global_variables.trainable))
    self.assertIn(
        'similarity_with_reference_embeddings/kernel',
        _variable_names(mobilenet_backbone.client_variables.trainable))

  def test_builds_resnet_embedding_model(self):
    resnet_model = build_model.embedding_model(
        build_model.ModelBackbone.RESNET50, input_shape=(224, 224, 3))
    self.assertIsInstance(resnet_model, tf.keras.Model)
    self.assertLen(resnet_model.inputs, 1)
    self.assertSequenceEqual(resnet_model.inputs[0].shape, (None, 224, 224, 3))
    self.assertLen(resnet_model.outputs, 1)
    self.assertSequenceEqual(resnet_model.outputs[0].shape, (None, 128))
    self.assertIn('res2a_branch1/kernel',
                  _model_trainable_variable_names(resnet_model))

  def test_builds_resnet_classification_model(self):
    resnet_backbone = build_model.classification_training_model(
        build_model.ModelBackbone.RESNET50,
        input_shape=(224, 224, 3),
        num_identities=281,
        embedding_dim_size=64)
    self.assertIsInstance(resnet_backbone, keras_utils.EmbeddingModel)
    self.assertIn('res2a_branch1/kernel',
                  _model_trainable_variable_names(resnet_backbone.model))
    self.assertLen(resnet_backbone.model.outputs, 2)
    # Classification output.
    self.assertSequenceEqual(resnet_backbone.model.outputs[0].shape,
                             (None, 281))
    # Embedding output.
    self.assertSequenceEqual(resnet_backbone.model.outputs[1].shape, (None, 64))

    self.assertIn('res2a_branch1/kernel',
                  _variable_names(resnet_backbone.global_variables.trainable))
    self.assertIn('similarity_with_reference_embeddings/kernel',
                  _variable_names(resnet_backbone.client_variables.trainable))

  def test_builds_mobilenet_small_embedding_model(self):
    mobilenet_model = build_model.embedding_model(
        build_model.ModelBackbone.MOBILESMALL, input_shape=(64, 64, 3))
    self.assertIsInstance(mobilenet_model, tf.keras.Model)
    self.assertLen(mobilenet_model.inputs, 1)
    self.assertSequenceEqual(mobilenet_model.inputs[0].shape,
                             (None, 64, 64, 3))
    self.assertLen(mobilenet_model.outputs, 1)
    self.assertSequenceEqual(mobilenet_model.outputs[0].shape, (None, 128))
    self.assertIn('block_0_depthwise_conv/depthwise_kernel',
                  _model_trainable_variable_names(mobilenet_model))

  def test_builds_mobilenet_small_classification_model(self):
    mobilenet_backbone = build_model.classification_training_model(
        build_model.ModelBackbone.MOBILESMALL,
        input_shape=(64, 64, 3),
        num_identities=233)
    self.assertIsInstance(mobilenet_backbone, keras_utils.EmbeddingModel)
    self.assertLen(mobilenet_backbone.model.inputs, 1)
    self.assertSequenceEqual(mobilenet_backbone.model.inputs[0].shape,
                             (None, 64, 64, 3))
    self.assertIn('block_0_depthwise_conv/depthwise_kernel',
                  _model_trainable_variable_names(mobilenet_backbone.model))
    self.assertLen(mobilenet_backbone.model.outputs, 2)
    # Classification output.
    self.assertSequenceEqual(mobilenet_backbone.model.outputs[0].shape,
                             (None, 233))
    # Embedding output.
    self.assertSequenceEqual(mobilenet_backbone.model.outputs[1].shape,
                             (None, 128))
    self.assertIn(
        'block_0_depthwise_conv/depthwise_kernel',
        _variable_names(mobilenet_backbone.global_variables.trainable))
    self.assertIn(
        'similarity_with_reference_embeddings/kernel',
        _variable_names(mobilenet_backbone.client_variables.trainable))

if __name__ == '__main__':
  tf.test.main()
