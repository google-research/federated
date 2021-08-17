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
"""Encoders for Keras dual encoder models."""

import abc
from typing import Dict, Iterable, List, Optional, Union

import tensorflow as tf

from dual_encoder import keras_layers


class Encoder(object, metaclass=abc.ABCMeta):
  """Represents an encoder for a dual encoder model.

  A typical model will have two encoders, one for the context and one for the
  label. The encoder specifies which of its internal Keras layers can be
  aggregated globally if trained using federated reconstruction.
  """

  @abc.abstractmethod
  def call(
      self, inputs: Union[tf.keras.layers.Layer, Dict[str,
                                                      tf.keras.layers.Layer]]
  ) -> tf.keras.layers.Layer:
    """Call the encoder to produce output.

    Args:
      inputs: It can be a `tf.keras.layers.Layer` or a dictionary of Keras
        layers keyed with feature names representing the input to the encoder.
        The Keras layer can have different shapes based on the expected shape
        of the input. If it is a dictionary of Keras layers, each layer have
        different shapes based on the expected shape of the input features.

    Returns:
      Encoder output.
    """
    pass

  @abc.abstractproperty
  def global_layers(self) -> Iterable[tf.keras.layers.Layer]:
    """"Get an iterable containing layers for global aggregation.

    Useful for federated reconstruction, where some layers' variables are
    aggregated across users and some layers' variables are kept local. This
    property and local_layers must together contain all of the layers
    containing variables that are constructed in the encoder. Note that any
    layers constructed outside of the encoder are not included here.

    Returns:
      An iterable of Keras layers for global aggregation.
    """
    pass

  @abc.abstractproperty
  def local_layers(self) -> Iterable[tf.keras.layers.Layer]:
    """"Get an iterable containing local layers.

    Useful for federated reconstruction, where some layers' variables are
    aggregated across users and some layers' variables are kept local. This
    property and global_layers must together contain all of the layers
    containing variables that are constructed in the encoder. Note that any
    layers constructed outside of the encoder are not included here.

    Returns:
      An iterable of Keras layers that should not be aggregated globally.
    """
    pass

  def __call__(
      self, inputs: Union[tf.keras.layers.Layer, Dict[str,
                                                      tf.keras.layers.Layer]]
  ) -> tf.keras.layers.Layer:
    """Calls the user-defined `call` method."""
    return self.call(inputs)


class EmbeddingBOWEncoder(Encoder):
  """Encoder that embeds a sequence of input items and performs BOW.

  Takes a [batch_size, sequence_length] input of items, embeds each item,
  performs a bag-of-words averaging of the embeddings for the items for each
  example (masking padded items with ID 0). Optionally applies some number of
  hidden dense layers.
  """

  def __init__(
      self,
      *,  # Args need to be provided by name.
      item_embedding_layer: tf.keras.layers.Embedding,
      hidden_dims: Optional[List[int]],
      hidden_activations: Optional[List[Optional[str]]],
      layer_name_prefix: str = 'Context',
  ):
    """Initializes encoder.

    Args:
      item_embedding_layer: A Keras embedding layer mapping item IDs to
        embeddings. The layer must mask zero item IDs so they can be masked
        during averaging, else a `ValueError` will be raised. This is defined
        outside of the encoder so it can be shared across encoders.
      hidden_dims: A List of integers representing the number of units for each
        hidden layer. If this is an empty List or `None`, no hidden layers are
        applied.
      hidden_activations: A List of strings representing the activations for
        each hidden layer, e.g. "relu". See `tf.keras.activations.get` for
        allowed strings. If any element in the list is `None`, this is
        equivalent to a "linear" activation. If this is an empty list or `None`,
        no hidden layers are applied. The length of `hidden_activations` must be
        the same as `hidden_dims`, else a `ValueError` will be raised.
      layer_name_prefix: Prefix for any layer constructed by this encoder.
        Typically this may be set to either "Context" or "Label".

    Raises:
      ValueError: If the embedding layer has `mask_zero=False`.
      ValueError: If the length of hidden_activations is not the same as
        hidden_dims.
    """
    if not item_embedding_layer.mask_zero:
      raise ValueError('Embedding layer must have mask_zero=True to perform '
                       'bag-of-words averaging.')
    self.item_embedding_layer = item_embedding_layer

    # All layers defined in this class.
    self.layers = {}

    # Masked average is performed over the timesteps of a
    # [batch_size, timesteps_size, item_embedding_dim] tensor.
    layer_name = f'{layer_name_prefix}MaskedAverage'
    self.layers[layer_name] = keras_layers.MaskedAverage(
        1, name=layer_name)

    self.num_hidden_layers = init_dense_layers(
        layers=self.layers,
        hidden_dims=hidden_dims,
        hidden_activations=hidden_activations,
        layer_name_prefix=layer_name_prefix)

    self.layer_name_prefix = layer_name_prefix

  def call(self, inputs: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
    """Call the encoder to produce output.

    Args:
      inputs: A Keras layer representing the input to the encoder. Typically,
        this will be an instance of `tf.keras.layers.Input`, with
        `shape=(None, )` or 'shape=(fixed_timesteps_size,)`.

    Returns:
      Encoder output with shape [batch_size, output_dim].
    """

    # Output shape: [batch_size, timesteps_size, item_embedding_dim].
    output_embedding = self.item_embedding_layer(inputs)

    # Merge embedding across timesteps. Output shape
    # [batch_size, item_embedding_dim].
    output_embedding = self.layers[f'{self.layer_name_prefix}MaskedAverage'](
        output_embedding)

    # Apply hidden and output layers. Output shape: [batch_size, output_dim].
    output_embedding = apply_dense_layers(
        embedding=output_embedding,
        layers=self.layers,
        num_hidden_layers=self.num_hidden_layers,
        layer_name_prefix=self.layer_name_prefix)

    return output_embedding

  @property
  def global_layers(self) -> Iterable[tf.keras.layers.Layer]:
    # Note that the item embedding layer isn't here since it was defined
    # outside this encoder.
    return list(self.layers.values())

  @property
  def local_layers(self) -> Iterable[tf.keras.layers.Layer]:
    return []


class EmbeddingEncoder(Encoder):
  """Encoder that embeds a single input item.

  Takes a [batch_size, 1] input of items and embeds each item. Optionally
  applies some number of hidden dense layers.
  """

  def __init__(
      self,
      *,  # Args need to be provided by name.
      item_embedding_layer: tf.keras.layers.Embedding,
      hidden_dims: Optional[List[int]],
      hidden_activations: Optional[List[Optional[str]]],
      layer_name_prefix: str = 'Label',
  ):
    """Initializes encoder.

    Args:
      item_embedding_layer: A Keras embedding layer mapping item IDs to
        embeddings. This is defined outside of the encoder so it can be shared
        across encoders.
      hidden_dims: A List of integers representing the number of units for each
        hidden layer. If this is an empty List or `None`, no hidden layers are
        applied.
      hidden_activations: A List of strings representing the activations for
        each hidden layer, e.g. "relu". See `tf.keras.activations.get` for
        allowed strings. If any element in the list is `None`, this is
        equivalent to a "linear" activation. If this is an empty list or `None`,
        no hidden layers are applied. The length of `hidden_activations` must be
        the same as `hidden_dims`, else a `ValueError` will be raised.
      layer_name_prefix: Prefix for any layer constructed by this encoder.
        Typically this may be set to either "Context" or "Label".

    Raises:
      ValueError: If the length of hidden_activations is not the same as
        hidden_dims.
    """
    self.item_embedding_layer = item_embedding_layer

    # All layers defined in this class.
    self.layers = {}

    layer_name = f'{layer_name_prefix}Flatten'
    self.layers[layer_name] = tf.keras.layers.Flatten(
        name=layer_name)

    self.num_hidden_layers = init_dense_layers(
        layers=self.layers,
        hidden_dims=hidden_dims,
        hidden_activations=hidden_activations,
        layer_name_prefix=layer_name_prefix)

    self.layer_name_prefix = layer_name_prefix

  def call(self, inputs: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
    """Call the encoder to produce output.

    Args:
      inputs: A Keras layer representing the input to the encoder. Typically,
        this will be an instance of `tf.keras.layers.Input`, with shape=(1,).

    Returns:
      Encoder output with shape [batch_size, output_dim].
    """

    # Output shape: [batch_size, 1, item_embedding_dim].
    output_embedding = self.item_embedding_layer(inputs)

    # Output shape: [batch_size, item_embedding_dim]
    output_embedding = self.layers[f'{self.layer_name_prefix}Flatten'](
        output_embedding)

    # Apply hidden and output layers. Output shape: [batch_size, output_dim].
    output_embedding = apply_dense_layers(
        embedding=output_embedding,
        layers=self.layers,
        num_hidden_layers=self.num_hidden_layers,
        layer_name_prefix=self.layer_name_prefix)

    return output_embedding

  @property
  def global_layers(self) -> Iterable[tf.keras.layers.Layer]:
    # Note that the item embedding layer isn't here since it was defined
    # outside this encoder.
    return list(self.layers.values())

  @property
  def local_layers(self) -> Iterable[tf.keras.layers.Layer]:
    return []


def init_dense_layers(
    layers: Dict[str, tf.keras.layers.Layer],
    hidden_dims: Optional[List[int]],
    hidden_activations: Optional[List[Optional[str]]],
    layer_name_prefix: str,
) -> int:
  """Initializes Dense layers in `layers`, returning number of hidden layers."""
  # Validate and define hidden layers.
  if hidden_dims is None:
    hidden_dims = []
  if hidden_activations is None:
    hidden_activations = []
  if len(hidden_dims) != len(hidden_activations):
    raise ValueError(f'Expected hidden_dims and hidden_activations to have '
                     f'the same length, got {hidden_dims} and '
                     f'{hidden_activations}.')

  for i in range(len(hidden_dims)):
    layer_name = f'{layer_name_prefix}Hidden{i + 1}'
    layers[layer_name] = tf.keras.layers.Dense(
        hidden_dims[i],
        activation=hidden_activations[i],
        name=layer_name)

  return len(hidden_dims)


def apply_dense_layers(
    embedding: tf.keras.layers.Layer,
    layers: Dict[str, tf.keras.layers.Layer],
    num_hidden_layers: int,
    layer_name_prefix: str,
) -> tf.keras.layers.Layer:
  """Applies Dense layers in `layers`."""
  # Apply hidden layers. Output shape [batch_size, hidden_dim] for each
  # hidden layer. If has a dense layer as the last output layer, then the
  # output shape is [batch_size, output_dim].
  for i in range(num_hidden_layers):
    layer_name = f'{layer_name_prefix}Hidden{i + 1}'
    embedding = layers[layer_name](embedding)

  return embedding
