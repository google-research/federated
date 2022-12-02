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
"""ResNet50 model for Keras.

Adapted from tf.keras.applications.resnet50.ResNet50().

Related papers/blogs:
- https://arxiv.org/abs/1512.03385
- https://arxiv.org/pdf/1603.05027v2.pdf
- http://torch.ch/blog/2016/02/04/resnets.html

"""
# TODO(b/227630719): Add pytype annotations to this file.

import tensorflow as tf
from tensorflow_addons import layers as tfa_layers

from dp_visual_embeddings.models import keras_utils


L2_WEIGHT_DECAY = 1e-4
GROUP_NORM_EPSILON = 1e-5
GROUP_NORM_NUM_GROUPS = 32
INPUT_SHAPE = (224, 224, 3)
_KERNEL_INIT = 'he_normal'
_WEIGHT_REG = tf.keras.regularizers.l2(L2_WEIGHT_DECAY)
_GLOBAL_AVG_POOL = True


def identity_block(input_tensor, kernel_size, filters, stage, block,
                   trainable_conv):
  """The identity block is the block that has no conv layer at shortcut.

  Args:
    input_tensor: input tensor
    kernel_size: default 3, the kernel size of middle conv layer at main path
    filters: list of integers, the filters of 3 conv layer at main path
    stage: integer, current stage label, used for generating layer names
    block: 'a','b'..., current block label, used for generating layer names
    trainable_conv: Whether to train the weights of convolutional layers. Can be
      used to reduce the size of trainable parameters when set to False.


  Returns:
      Output tensor for the block.
  """
  filters1, filters2, filters3 = filters
  if tf.keras.backend.image_data_format() == 'channels_last':
    gn_axis = 3
  else:
    gn_axis = 1
  conv_name_base = 'res' + str(stage) + block + '_branch'
  gn_name_base = 'gn' + str(stage) + block + '_branch'

  x = tf.keras.layers.Convolution2D(
      filters1, (1, 1),
      use_bias=False,
      kernel_initializer=_KERNEL_INIT,
      kernel_regularizer=_WEIGHT_REG,
      name=conv_name_base + '2a',
      trainable=trainable_conv)(
          input_tensor)
  x = tfa_layers.GroupNormalization(
      groups=GROUP_NORM_NUM_GROUPS,
      axis=gn_axis,
      epsilon=GROUP_NORM_EPSILON,
      name=gn_name_base + '2a')(
          x)
  x = tf.keras.layers.Activation('relu')(x)

  x = tf.keras.layers.Convolution2D(
      filters2,
      kernel_size,
      use_bias=False,
      padding='same',
      kernel_initializer=_KERNEL_INIT,
      kernel_regularizer=_WEIGHT_REG,
      name=conv_name_base + '2b',
      trainable=trainable_conv)(
          x)
  x = tfa_layers.GroupNormalization(
      groups=GROUP_NORM_NUM_GROUPS,
      axis=gn_axis,
      epsilon=GROUP_NORM_EPSILON,
      name=gn_name_base + '2b')(
          x)
  x = tf.keras.layers.Activation('relu')(x)

  x = tf.keras.layers.Convolution2D(
      filters3, (1, 1),
      use_bias=False,
      kernel_initializer=_KERNEL_INIT,
      kernel_regularizer=_WEIGHT_REG,
      name=conv_name_base + '2c',
      trainable=trainable_conv)(
          x)
  x = tfa_layers.GroupNormalization(
      groups=GROUP_NORM_NUM_GROUPS,
      axis=gn_axis,
      epsilon=GROUP_NORM_EPSILON,
      name=gn_name_base + '2c')(
          x)

  x = tf.keras.layers.add([x, input_tensor])
  x = tf.keras.layers.Activation('relu')(x)
  return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2),
               trainable_conv=True):
  """A block that has a conv layer at shortcut.

  Args:
    input_tensor: input tensor
    kernel_size: default 3, the kernel size of
        middle conv layer at main path
    filters: list of integers, the filters of 3 conv layer at main path
    stage: integer, current stage label, used for generating layer names
    block: 'a','b'..., current block label, used for generating layer names
    strides: Strides for the second conv layer in the block.
    trainable_conv: Whether to train the weights of convolutional layers. Can be
      used to reduce the size of trainable parameters when set to False.

  Returns:
      Output tensor for the block.

  Note that from stage 3,
  the second conv layer at main path is with strides=(2, 2)
  And the shortcut should have strides=(2, 2) as well
  """
  filters1, filters2, filters3 = filters
  if tf.keras.backend.image_data_format() == 'channels_last':
    gn_axis = 3
  else:
    gn_axis = 1
  conv_name_base = 'res' + str(stage) + block + '_branch'
  gn_name_base = 'gn' + str(stage) + block + '_branch'

  x = tf.keras.layers.Convolution2D(
      filters1, (1, 1),
      use_bias=False,
      kernel_initializer=_KERNEL_INIT,
      kernel_regularizer=_WEIGHT_REG,
      name=conv_name_base + '2a',
      trainable=trainable_conv)(
          input_tensor)
  x = tfa_layers.GroupNormalization(
      groups=GROUP_NORM_NUM_GROUPS,
      axis=gn_axis,
      epsilon=GROUP_NORM_EPSILON,
      name=gn_name_base + '2a')(
          x)
  x = tf.keras.layers.Activation('relu')(x)

  x = tf.keras.layers.Convolution2D(
      filters2,
      kernel_size,
      strides=strides,
      padding='same',
      use_bias=False,
      kernel_initializer=_KERNEL_INIT,
      kernel_regularizer=_WEIGHT_REG,
      name=conv_name_base + '2b',
      trainable=trainable_conv)(
          x)
  x = tfa_layers.GroupNormalization(
      groups=GROUP_NORM_NUM_GROUPS,
      axis=gn_axis,
      epsilon=GROUP_NORM_EPSILON,
      name=gn_name_base + '2b')(
          x)
  x = tf.keras.layers.Activation('relu')(x)

  x = tf.keras.layers.Convolution2D(
      filters3, (1, 1),
      use_bias=False,
      kernel_initializer=_KERNEL_INIT,
      kernel_regularizer=_WEIGHT_REG,
      name=conv_name_base + '2c',
      trainable=trainable_conv)(
          x)
  x = tfa_layers.GroupNormalization(
      groups=GROUP_NORM_NUM_GROUPS,
      axis=gn_axis,
      epsilon=GROUP_NORM_EPSILON,
      name=gn_name_base + '2c')(
          x)

  shortcut = tf.keras.layers.Convolution2D(
      filters3, (1, 1),
      use_bias=False,
      strides=strides,
      kernel_initializer=_KERNEL_INIT,
      kernel_regularizer=_WEIGHT_REG,
      name=conv_name_base + '1',
      trainable=trainable_conv)(
          input_tensor)
  shortcut = tfa_layers.GroupNormalization(
      groups=GROUP_NORM_NUM_GROUPS,
      axis=gn_axis,
      epsilon=GROUP_NORM_EPSILON,
      name=gn_name_base + '1')(
          shortcut)

  x = tf.keras.layers.add([x, shortcut])
  x = tf.keras.layers.Activation('relu')(x)
  return x


def resnet50_image2embedding(
    embedding_dim_size: int = 128,
    global_averaging_pooling: bool = _GLOBAL_AVG_POOL,
    always_normalize: bool = False,
    trainable_conv: bool = True,
    input_shape: tuple[int, int, int] = INPUT_SHAPE) -> tf.keras.Model:
  """Instantiates the ResNet50 architecture.

  Args:
    embedding_dim_size: The number of dimensions in the output embedding.
    global_averaging_pooling: If True, use `AveragePooling2D` before the
      embedding layer. If False, use a `AveragePooling2D` of size 3 and stride
      2.
    always_normalize: If True, the output embedding will be normalized for
      both training and inference; if False, only normalized for inference.
    trainable_conv: Whether to train the weights of convolutional layers. Can be
      used to reduce the size of trainable parameters when set to False.
    input_shape: The input image shape.

  Returns:
    A keras Model with a ResNet-50 architecture.
  """
  # Determine proper input shape
  if tf.keras.backend.image_data_format() == 'channels_first':
    raise NotImplementedError('Assuming channels last')
  gn_axis = 3

  img_input = tf.keras.Input(shape=input_shape, name='images')
  first_trainable = True
  x = tf.keras.layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
  x = tf.keras.layers.Convolution2D(
      64, (7, 7),
      use_bias=False,
      strides=(2, 2),
      padding='valid',
      kernel_initializer=_KERNEL_INIT,
      kernel_regularizer=_WEIGHT_REG,
      name='conv1',
      trainable=first_trainable)(
          x)
  x = tfa_layers.GroupNormalization(
      groups=GROUP_NORM_NUM_GROUPS,
      axis=gn_axis,
      epsilon=GROUP_NORM_EPSILON,
      name='gn_conv1')(
          x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

  x = conv_block(
      x,
      3, [64, 64, 256],
      stage=2,
      block='a',
      strides=(1, 1),
      trainable_conv=first_trainable)
  x = identity_block(
      x, 3, [64, 64, 256], stage=2, block='b', trainable_conv=first_trainable)
  x = identity_block(
      x, 3, [64, 64, 256], stage=2, block='c', trainable_conv=first_trainable)

  group2_trainable = trainable_conv
  x = conv_block(
      x,
      3, [128, 128, 512],
      stage=3,
      block='a',
      trainable_conv=group2_trainable)
  x = identity_block(
      x,
      3, [128, 128, 512],
      stage=3,
      block='b',
      trainable_conv=group2_trainable)
  x = identity_block(
      x,
      3, [128, 128, 512],
      stage=3,
      block='c',
      trainable_conv=group2_trainable)
  x = identity_block(
      x,
      3, [128, 128, 512],
      stage=3,
      block='d',
      trainable_conv=group2_trainable)

  group3_trainable = trainable_conv
  x = conv_block(
      x,
      3, [256, 256, 1024],
      stage=4,
      block='a',
      trainable_conv=group3_trainable)
  x = identity_block(
      x,
      3, [256, 256, 1024],
      stage=4,
      block='b',
      trainable_conv=group3_trainable)
  x = identity_block(
      x,
      3, [256, 256, 1024],
      stage=4,
      block='c',
      trainable_conv=group3_trainable)
  x = identity_block(
      x,
      3, [256, 256, 1024],
      stage=4,
      block='d',
      trainable_conv=group3_trainable)
  x = identity_block(
      x,
      3, [256, 256, 1024],
      stage=4,
      block='e',
      trainable_conv=group3_trainable)
  x = identity_block(
      x,
      3, [256, 256, 1024],
      stage=4,
      block='f',
      trainable_conv=group3_trainable)

  group4_trainable = True
  x = conv_block(
      x,
      3, [512, 512, 2048],
      stage=5,
      block='a',
      trainable_conv=group4_trainable)
  x = identity_block(
      x,
      3, [512, 512, 2048],
      stage=5,
      block='b',
      trainable_conv=group4_trainable)
  x = identity_block(
      x,
      3, [512, 512, 2048],
      stage=5,
      block='c',
      trainable_conv=group4_trainable)

  if global_averaging_pooling:
    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
  else:
    x = tf.keras.layers.AveragePooling2D(
        pool_size=(3, 3), strides=2, name='avg32_pool')(
            x)
    x = tf.keras.layers.Flatten()(x)

  predicted_embeddings = tf.keras.layers.Dense(
      embedding_dim_size,
      activation=None,
      use_bias=True,
      kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
      kernel_regularizer=_WEIGHT_REG,
      bias_regularizer=_WEIGHT_REG,
      name='predicted_embeddings')(
          x)

  predicted_embeddings = keras_utils.EmbedNormLayer(
      always_normalize=always_normalize)(
          predicted_embeddings)
  return tf.keras.Model(
      img_input, predicted_embeddings, name='resnet50_embedding')


def resnet50_with_head(
    num_classes: int,
    embedding_dim_size: int = 128,
    global_averaging_pooling: bool = _GLOBAL_AVG_POOL,
    use_normalize: bool = False,
    trainable_conv: bool = True,
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
) -> keras_utils.EmbeddingModel:
  """ResNet50 with head for classification and embedding training.

  Args:
    num_classes: Defines the size of the head output.
    embedding_dim_size: Defines the size of the embedding vector for an input
      image.
    global_averaging_pooling: Whether to use global averaging pooling for the
      layer before head. If False, use a `AveragePooling2D` of size 3 and stride
      2 instead.
    use_normalize: Whether to normalize the embedding and head weights during
      training.
    trainable_conv: Whether to train the weights of convolutional layers. Can be
      used to reduce the size of trainable parameters when set to False.
    input_shape: The input image shape.

  Returns:
    A `keras_utils.EmbeddingModel` with attributes (model, global_variables,
      client_variables)
  """
  base_model = resnet50_image2embedding(
      embedding_dim_size=embedding_dim_size,
      global_averaging_pooling=global_averaging_pooling,
      always_normalize=use_normalize,
      trainable_conv=trainable_conv,
      input_shape=input_shape)

  return keras_utils.add_embedding_head(
      base_model=base_model,
      num_identities=num_classes,
      use_normalize=use_normalize)
