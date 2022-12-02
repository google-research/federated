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
"""Top-level builder for different model architectures."""

import enum

import tensorflow as tf

from dp_visual_embeddings.models import keras_lenet
from dp_visual_embeddings.models import keras_mobilenet_v2
from dp_visual_embeddings.models import keras_resnet
from dp_visual_embeddings.models import keras_utils


class ModelBackbone(enum.Enum):
  MOBILENET2 = enum.auto()
  MOBILESMALL = enum.auto()
  RESNET50 = enum.auto()
  LENET = enum.auto()


def get_backbone_names():
  return [backbone.name.lower() for backbone in list(ModelBackbone)]


def embedding_model(model_backbone: ModelBackbone,
                    input_shape: tuple[int, int, int],
                    embedding_dim_size: int = 128,
                    trainable_conv: bool = True) -> tf.keras.Model:
  """Builds a Keras model to map from images to embeddings.

  Args:
    model_backbone: Selects between different convolutional "backbones:" the
      main part of the model.
    input_shape: The shape of a tensor for a single input image (excludes the
      batch dimension).
    embedding_dim_size: The dimensionality of the output embeddings.
    trainable_conv: Whether to train the weights of convolutional layers. Can be
      used to reduce the size of trainable parameters when set to False.

  Returns:
    The keras Model object.
  """
  if model_backbone == ModelBackbone.MOBILENET2:
    return keras_mobilenet_v2.create_mobilenet_v2_for_embedding_prediction(
        input_shape=input_shape,
        embedding_dim_size=embedding_dim_size,
        trainable_conv=trainable_conv)
  elif model_backbone == ModelBackbone.MOBILESMALL:
    return keras_mobilenet_v2.create_small_mobilenet_v2_for_embedding_prediction(
        input_shape=input_shape,
        embedding_dim_size=embedding_dim_size,
        trainable_conv=trainable_conv)
  elif model_backbone == ModelBackbone.RESNET50:
    return keras_resnet.resnet50_image2embedding(
        input_shape=input_shape,
        embedding_dim_size=embedding_dim_size,
        trainable_conv=trainable_conv)
  elif model_backbone == ModelBackbone.LENET:
    if not trainable_conv:
      raise ValueError('Freezing not supported for LeNet.')
    return keras_lenet.image2embedding(
        image_shape=input_shape, embedding_dim_size=embedding_dim_size)
  else:
    raise ValueError('Unexpected backbone: %s' % model_backbone)


def classification_training_model(
    model_backbone: ModelBackbone,
    input_shape: tuple[int, int, int],
    num_identities: int,
    embedding_dim_size: int = 128,
    trainable_conv: bool = True) -> keras_utils.EmbeddingModel:
  """Builds a Keras model to map from images to identity classification logits.

  Args:
    model_backbone: Selects between different convolutional "backbones:" the
      main part of the model.
    input_shape: The shape of a tensor for a single input image (excludes the
      batch dimension).
    num_identities: The number of different labels to predict between.
    embedding_dim_size: The dimensionality of the embedding bottleneck.
    trainable_conv: Whether to train the weights of convolutional layers. Can be
      used to reduce the size of trainable parameters when set to False.

  Returns:
    A EmbeddingModel object with a keras model, and lists of variables to be
    optimized.
  """
  if model_backbone == ModelBackbone.MOBILENET2:
    return keras_mobilenet_v2.create_mobilenet_v2_for_backbone_training(
        input_shape=input_shape,
        num_identities=num_identities,
        embedding_dim_size=embedding_dim_size,
        trainable_conv=trainable_conv)
  elif model_backbone == ModelBackbone.MOBILESMALL:
    return keras_mobilenet_v2.create_small_mobilenet_v2_for_backbone_training(
        input_shape=input_shape,
        num_identities=num_identities,
        embedding_dim_size=embedding_dim_size,
        trainable_conv=trainable_conv)
  elif model_backbone == ModelBackbone.RESNET50:
    return keras_resnet.resnet50_with_head(
        input_shape=input_shape,
        num_classes=num_identities,
        embedding_dim_size=embedding_dim_size,
        trainable_conv=trainable_conv)
  elif model_backbone == ModelBackbone.LENET:
    if not trainable_conv:
      raise ValueError('Freezing not supported for LeNet.')
    return keras_lenet.lenet_with_head(
        num_classes=num_identities,
        image_shape=input_shape,
        embedding_dim_size=embedding_dim_size)
  else:
    raise ValueError('Unexpected backbone: %s' % model_backbone)
