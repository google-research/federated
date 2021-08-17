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
"""Utilities for dual encoder model launchers."""

from typing import Callable, List, Optional, Tuple

import tensorflow as tf

from dual_encoder import encoders
from dual_encoder import keras_layers
from dual_encoder import losses
from dual_encoder import metrics
from dual_encoder import model_utils as utils

# Loss functions for training.
BATCH_SOFTMAX = 'batch_softmax'
GLOBAL_SOFTMAX = 'global_softmax'
HINGE = 'hinge'
# Type of encoders for the dual_encoder_model
FLATTEN = 'flatten'
BOW = 'bow'


def build_encoder(encoder_type: str,
                  item_embedding_layer: tf.keras.layers.Embedding,
                  hidden_dims: Optional[List[int]],
                  hidden_activations: Optional[List[Optional[str]]],
                  layer_name_prefix: str) -> encoders.Encoder:
  """Return the enocder being used to encode the context or the label.

  Args:
    encoder_type: The type of encoder being used for encoding the context or the
      label. Currently supporting `FLATTEN`, `BOW`.
    item_embedding_layer: a Keras embedding layer mapping item IDs to
      embeddings. This is defined outside of the encoder so it can be shared
      across encoders.
    hidden_dims: A List of integers representing the number of units for each
      hidden layer. If this is an empty `List` or `None`, no hidden layers are
      applied.
    hidden_activations: A List of strings representing the activations for
      each hidden layer in this encoder, e.g. "relu". See
      `tf.keras.activations.get` for allowed strings. If any element in the list
      is `None`, this is equivalent to a "linear" activation. If this is an
      empty list or `None`, no hidden layers are applied. The length of
      `hidden_activations` must be the same as `hidden_dims`, else a
      `ValueError` will be raised by the encoder.
    layer_name_prefix: Prefix for any layer constructed by this encoder.
      Typically this may be set to either "Context" or "Label".

  Returns:
    An encoder object with the specified encoder_type.

  Raises:
    ValueError: If `encoder_type` is not `FLATTEN` or `BOW`.
  """
  if encoder_type == FLATTEN:
    encoder = encoders.EmbeddingEncoder(
        item_embedding_layer=item_embedding_layer,
        hidden_dims=hidden_dims,
        hidden_activations=hidden_activations,
        layer_name_prefix=layer_name_prefix)
  elif encoder_type == BOW:
    encoder = encoders.EmbeddingBOWEncoder(
        item_embedding_layer=item_embedding_layer,
        hidden_dims=hidden_dims,
        hidden_activations=hidden_activations,
        layer_name_prefix=layer_name_prefix)
  else:
    raise ValueError(f'Got unexpected encoder type {encoder_type}.')

  return encoder


def get_loss(
    loss_function: str,
    normalization_fn: str = 'l2_normalize',
    expect_embeddings: bool = False,
    spreadout_context_lambda: float = 0.0,
    spreadout_label_lambda: float = 0.0,
    spreadout_cross_lambda: float = 0.0,
    use_global_similarity: bool = False) -> tf.keras.losses.Loss:
  """Get the loss function.

  Args:
    loss_function: The type of loss function being used for training. Currently
      supporting batch softmax loss (`BATCH_SOFTMAX`), global softmax loss
      (`GLOBAL_SOFTMAX`), and hinge loss (`HINGE`).
    normalization_fn: The normalization function to be applied to embeddings.
    expect_embeddings: If `expect_embeddings` is False, the output of the model
      is expected to be a pre-calculated similarities matrix, and normalization
      is ignored. Otherwise, the outputs of the model are expect to be the
      context embedding and the label embedding. Note that if
      `expect_embeddings` is False, `spreadout_context_lambda` and
      `spreadout_label_lambda` must be 0, since these depend on the context and
      label embeddings.
    spreadout_context_lambda: Context spreadout scaling constant. Expect to be
      non-negative.
    spreadout_label_lambda: Label spreadout scaling constant. Expect to be
      non-negative.
    spreadout_cross_lambda: Cross spreadout scaling constant. Expect to be
      non-negative.
    use_global_similarity: The boolean value to indicate whether the model
      outputs global similarity or the batch similarity. If true, use
      `losses.BatchSoftmaxWithGlobalSimilarity` to compute batch softmax loss.
      Otherwise, use `losses.BatchSoftmax` for batch softmax loss.

  Returns:
    An initialized `tf.keras.losses.Loss` object.

  Raises:
    ValueError: If `loss_function` is not in {`BATCH_SOFTMAX`, `GLOBAL_SOFTMAX`,
      `HINGE`}.
  """
  # TODO(b/179517823): Merge the "*Softmax*" implementations by supporting
  # use_global_similarity as an arg,
  if loss_function == BATCH_SOFTMAX:
    if use_global_similarity:
      loss_fn = losses.BatchSoftmaxWithGlobalSimilarity
    else:
      loss_fn = losses.BatchSoftmax
  elif loss_function == GLOBAL_SOFTMAX:
    loss_fn = losses.GlobalSoftmax
  elif loss_function == HINGE:
    loss_fn = losses.Hinge
  else:
    raise ValueError(f'Got unexpected loss function {loss_function}.')

  loss = loss_fn(
      normalization_fn=utils.NORMALIZATION_FN_MAP[normalization_fn],
      expect_embeddings=expect_embeddings,
      spreadout_context_lambda=spreadout_context_lambda,
      spreadout_label_lambda=spreadout_label_lambda,
      spreadout_cross_lambda=spreadout_cross_lambda,
      use_global_similarity=use_global_similarity)
  return loss


def get_metrics(
    eval_top_k: List[int],
    normalization_fn: str = 'l2_normalize',
    expect_embeddings: bool = False,
    use_global_similarity: bool = False) -> List[tf.keras.metrics.Mean]:
  """Gets model evaluation metrics.

  Args:
    eval_top_k: The list of recall_k values to evaluate..
    normalization_fn: The normalization function to be applied to embeddings.
      See `utils.NORMALIZATION_FN_MAP` for options.
    expect_embeddings: If `expect_embeddings` is False, the output of the model
      is expected to be a pre-calculated similarities matrix, and normalization
      is ignored. Otherwise, the outputs of the model are expect to be the
      context embedding and the label embedding.
    use_global_similarity: The boolean value to indicate whether the model
      outputs global similarity or the batch similarity. If true, compute the
      batch recall with `metrics.BatchRecallWithGlobalSimilarity` and add global
      recall to the metrics list. Otherwise, compute the batch recall with
      `metrics.BatchRecall` and do not calculate the global recall.

  Returns:
    A list of initialized `tf.keras.metrics.Mean` object.
  """
  metrics_list = []

  # TODO(b/179517823): Merge the "Recall" implementations by supporting
  # use_global_similarity as an arg,
  if use_global_similarity:
    for k in eval_top_k:
      metrics_list.append(
          metrics.BatchRecallWithGlobalSimilarity(
              recall_k=k,
              normalization_fn=utils.NORMALIZATION_FN_MAP[normalization_fn],
              expect_embeddings=expect_embeddings,
              name=f'Batch_Recall/Recall_{k}'))
      metrics_list.append(
          metrics.GlobalRecall(
              recall_k=k,
              normalization_fn=utils.NORMALIZATION_FN_MAP[normalization_fn],
              expect_embeddings=expect_embeddings,
              name=f'Global_Recall/Recall_{k}'))

    metrics_list.append(
        metrics.GlobalMeanRank(
            normalization_fn=utils.NORMALIZATION_FN_MAP[normalization_fn],
            expect_embeddings=expect_embeddings,
            name='global_mean_rank'))
  else:
    for k in eval_top_k:
      metrics_list.append(
          metrics.BatchRecall(
              recall_k=k,
              normalization_fn=utils.NORMALIZATION_FN_MAP[normalization_fn],
              expect_embeddings=expect_embeddings,
              name=f'Batch_Recall/Recall_{k}'))

    metrics_list.append(
        metrics.BatchMeanRank(
            normalization_fn=utils.NORMALIZATION_FN_MAP[normalization_fn],
            expect_embeddings=expect_embeddings,
            name='batch_mean_rank'))

  metrics_list += [metrics.NumExamplesCounter(), metrics.NumBatchesCounter()]
  return metrics_list


def build_keras_model(
    *,  # Args need to be provided by name.
    item_vocab_size: int,
    item_embedding_dim: int,
    spreadout_lambda: float = 0.0,
    l2_regularization: float = 0.0,
    normalization_fn: str = 'l2_normalize',
    context_encoder_type: str = BOW,
    label_encoder_type: str = FLATTEN,
    context_hidden_dims: Optional[List[int]],
    context_hidden_activations: Optional[List[Optional[str]]],
    label_hidden_dims: Optional[List[int]],
    label_hidden_activations: Optional[List[Optional[str]]],
    output_embeddings: bool = False,
    use_global_similarity: bool = False) -> tf.keras.Model:
  """Returns a compiled dual encoder keras model.

  Args:
    item_vocab_size: Size of the vocabulary. It doesn't include pad ID,
      which is assumed to be 0. The non-pad item IDs are expected to be
      in the range [1, item_vocab_size].
    item_embedding_dim: Output size of the item embedding layer.
    spreadout_lambda: Scaling constant for spreadout regularization on item
      embeddings. This ensures that item embeddings are generally spread far
      apart, and that random items have dissimilar embeddings. See
      `keras_layers.EmbeddingSpreadoutRegularizer` for details. Expect to be
      non-negative. A value of 0.0 indicates no regularization.
    l2_regularization: The constant to use to scale L2 regularization on all
      item embeddings. A value of 0.0 indicates no regularization.
    normalization_fn: The normalization function to be applied to embeddings.
    context_encoder_type: Type of the encoder for the context.
    label_encoder_type: Type of the encoder for the label.
    context_hidden_dims: A List of integers representing the number of units for
      each hidden layer in context tower. If this is an empty `List` or `None`,
      no hidden layers are applied.
    context_hidden_activations: A List of strings representing the activations
      for each hidden layer in context tower, e.g. "relu". See
      `tf.keras.activations.get` for allowed strings. If any element in the list
      is `None`, this is equivalent to a "linear" activation. If this is an
      empty list or `None`, no hidden layers are applied. The length of
      `context_hidden_activations` must be the same as `context_hidden_dims`,
      else a `ValueError` will be raised by the encoder.
    label_hidden_dims: A List of integers representing the number of units for
      each hidden layer in label tower. If this is an empty `List` or `None`,
      no hidden layers are applied.
    label_hidden_activations: A List of strings representing the activations
      for each hidden layer in label tower, e.g. "relu". See
      `tf.keras.activations.get` for allowed strings. If any element in the list
      is `None`, this is equivalent to a "linear" activation. If this is an
      empty list or `None`, no hidden layers are applied. The length of
      `label_hidden_activations` must be the same as `label_hidden_dims`,
      else a `ValueError` will be raised by the encoder.
    output_embeddings: The output of the model are the context and label
      embeddings if `output_embeddings` is True. Otherwise, the model outputs
      batch or global similarities.
    use_global_similarity: The boolean value to set whether the model use the
      global similarity or the batch similarity. If true, the model computes the
      global similarity (the similarities between the context embeddings of the
      batch examples and the label embeddings of the full vocabulary) or outputs
      the context embeddings of the batch examples and the label embeddings of
      the full vocabulary. If false, the model computes the batch similarity (
      the similarities between the context and label embeddings of the batch
      examples) or outputs the context and label embeddings of the batch
      examples.

  Returns:
    A Keras model for dual encoder.
  """

  # Shared item embedding layer for both context and label towers.
  # Context is the previous item ID sequence and label is the next item ID.
  embedding_vocab_size = item_vocab_size + 1
  item_embedding_layer = tf.keras.layers.Embedding(
      embedding_vocab_size,
      item_embedding_dim,
      embeddings_regularizer=keras_layers.EmbeddingSpreadoutRegularizer(
          spreadout_lambda=spreadout_lambda,
          normalization_fn=utils.NORMALIZATION_FN_MAP[normalization_fn],
          l2_regularization=l2_regularization),
      mask_zero=True,
      name='ItemEmbeddings'
  )

  # Create the context and label encoders for building the dual encoder model.
  context_encoder = build_encoder(
      encoder_type=context_encoder_type,
      item_embedding_layer=item_embedding_layer,
      hidden_dims=context_hidden_dims,
      hidden_activations=context_hidden_activations,
      layer_name_prefix='Context')

  label_encoder = build_encoder(
      encoder_type=label_encoder_type,
      item_embedding_layer=item_embedding_layer,
      hidden_dims=label_hidden_dims,
      hidden_activations=label_hidden_activations,
      layer_name_prefix='Label')

  # Create the dual encoder model.
  model = build_id_based_dual_encoder_model(
      context_input_shape=(None,),
      label_input_shape=(1,),
      context_encoder=context_encoder,
      label_encoder=label_encoder,
      normalization_fn=utils.NORMALIZATION_FN_MAP[normalization_fn],
      output_embeddings=output_embeddings,
      use_global_similarity=use_global_similarity,
      item_vocab_size=item_vocab_size)

  return model


def build_id_based_dual_encoder_model(
    *,  # Args need to be provided by name.
    context_input_shape: Tuple[Optional[int], ...],
    label_input_shape: Tuple[Optional[int], ...],
    context_encoder: Callable[[tf.Tensor], tf.Tensor],
    label_encoder: Callable[[tf.Tensor], tf.Tensor],
    normalization_fn: utils.NormalizationFnType = utils.l2_normalize_fn,
    output_embeddings: bool = False,
    use_global_similarity: bool = False,
    item_vocab_size: Optional[int] = None) -> tf.keras.Model:
  """Define the dual encoder model.

  The context and the label share the same item embedding layer in this version.

  Args:
    context_input_shape: The shape of the input(s) to the context encoder.
    label_input_shape: The shape of the input(s) to the label encoder.
    context_encoder: A callable for encoding the context input into the context
      embedding. This will typically be an instance of `encoders.Encoder`.
    label_encoder: A callable for encoding the context input into the context
      embedding. This will typically be an instance of `encoders.Encoder`.
    normalization_fn: The normalization function to be applied to embeddings.
    output_embeddings: The output of the model are the context and label
      embeddings if `output_embeddings` is True. Otherwise, the model outputs
      batch or global similarities.
    use_global_similarity: The boolean value to set whether the model use the
      global similarity or the batch similarity. If true, the model computes the
      global similarity (the similarities between the context embeddings of the
      batch examples and the label embeddings of the full vocabulary) or outputs
      the context embeddings of the batch examples and the label embeddings of
      the full vocabulary. If false, the model computes the batch similarity (
      the similarities between the context and label embeddings of the batch
      examples) or outputs the context and label embeddings of the batch
      examples. `item_vocab_size` is ignored if `use_global_similarity` is
      False.
    item_vocab_size: Size of the vocabulary. It doesn't include pad ID,
      which is assumed to be 0. The non-pad item IDs are expected to be
      in the range [1, item_vocab_size]. Must pass in this argument if
      `use_global_similarity` is True.

  Returns:
    A Keras model for dual encoder.

  Raises:
    ValueError: If the last dimension of the outputs of context encoder and
      label encoder are not equal.
  """

  # Context tower.
  context_encoder_input = tf.keras.layers.Input(
      shape=context_input_shape, name='ContextInput')
  context_embedding = context_encoder(context_encoder_input)

  # Label tower.
  label_input_layer = tf.keras.layers.Input(
      shape=label_input_shape, name='LabelInput')
  if use_global_similarity:
    # TODO(b/174267080) This full_vocab_layer_input only works for the ID based
    # dual encoder model for the MovieLens data. Can make it general by passing
    # as an input arg in the future.
    embedding_vocab_size = item_vocab_size + 1
    full_vocab_layer_input = tf.expand_dims(tf.range(embedding_vocab_size), -1)
    label_encoder_input = tf.keras.layers.Lambda(
        lambda x: full_vocab_layer_input)(
            label_input_layer)
  else:
    label_encoder_input = label_input_layer

  label_embedding = label_encoder(label_encoder_input)

  if context_embedding.shape.dims[-1] != label_embedding.shape.dims[-1]:
    raise ValueError('The last dimension of `context_embedding` and '
                     '`label_embedding` must be same, but are '
                     f'{context_embedding.shape.dims[-1]} and '
                     f'{label_embedding.shape.dims[-1]}.')

  if output_embeddings:
    output = tf.keras.backend.concatenate(
        (context_embedding, label_embedding), axis=0)
  else:
    # output[i][j] is the dot product of the ith context embedding and the jth
    # label embedding in a batch.
    output = utils.Similarities(
        normalization_fn=normalization_fn, name='Similarities')(
            [context_embedding, label_embedding])

  keras_inputs = {'context': context_encoder_input, 'label': label_input_layer}
  model = tf.keras.Model(inputs=keras_inputs, outputs=output)

  return model
