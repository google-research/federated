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
"""Transformer model for next word prediction.

Reference
Attention Is All You Need, 2017 (https://arxiv.org/abs/1706.03762)
TF transformer tutorial (https://www.tensorflow.org/tutorials/text/transformer)
"""
from typing import Optional, Tuple

import numpy as np

import tensorflow as tf
from utils.models.stackoverflow_models import TransposableEmbedding

DEFAULT_LARGE_NEGATIVE = -1e9
DEFAULT_POSITIONAL_BASE = 10000


def scaled_dot_product_attention(
    query: tf.Tensor, key: tf.Tensor, value: tf.Tensor,
    mask: Optional[tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
  """Apply the scaled attention weights.

  q (query), k (key), v (value) must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    query: Query feature vectors, shape == (..., seq_len_q, depth).
    key: Key feature vectors, shape == (..., seq_len_k, depth).
    value: Value feature vectors, shape == (..., seq_len_v, depth_v).
    mask: Float tensor with shape broadcastable to (..., seq_len_q, seq_len_k).

  Returns:
    The output attention vectors.
  """

  matmul_qk = tf.matmul(query, key, transpose_b=True)

  # Scale matmul_qk so that the softmax does not vanish when feature dimension
  # is large.
  ftr_dim = tf.cast(tf.shape(key)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(ftr_dim)

  if mask is not None:
    scaled_attention_logits += (mask * DEFAULT_LARGE_NEGATIVE)

  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

  output = tf.matmul(attention_weights, value)

  return output


class MultiHeadAttention(tf.keras.layers.Layer):
  """Multi-head attention layer for transformer.

  Attributes:
    num_heads: An integer of the number of heads.
    d_model: An integer of the total dimension of the multi-head layer. Must be
      divisible by num_heads. Each head will apply the scaled attention weights.
  """

  def __init__(self, d_model: int, num_heads: int):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    if d_model % self.num_heads != 0:
      raise ValueError(
          'Feature dimension should be divisible by number of heads! Got {}/{}'
          .format(d_model, num_heads))

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

  def _split_heads(self, x: tf.Tensor, batch_size: int) -> tf.Tensor:
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    # Transpose the result such that the shape is (batch_size, num_heads,
    # seq_len, depth)
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v: tf.Tensor, k: tf.Tensor, q: tf.Tensor,
           mask: tf.Tensor) -> tf.Tensor:
    batch_size = tf.shape(q)[0]

    q = self.wq(q)
    k = self.wk(k)
    v = self.wv(v)

    q = self._split_heads(q, batch_size)
    k = self._split_heads(k, batch_size)
    v = self._split_heads(v, batch_size)

    scaled_attention = scaled_dot_product_attention(q, k, v, mask)

    # Reshape scaled_attention from (batch_size, num_heads, seq_len_q, depth)
    # to (batch_size, seq_len_q, num_heads, depth).
    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))

    # The output is of shape (batch_size, seq_len_q, d_model).
    output = self.dense(concat_attention)

    return output


def point_wise_feed_forward_network(d_model: int,
                                    d_hidden: int) -> tf.keras.Model:
  """Returns all the possible positional encodings.

  Args:
    d_model: Dimension of the input feature.
    d_hidden: Dimension of the hidden layer.

  Returns:
    A one-hidden-layer MLP.
  """
  return tf.keras.Sequential([
      tf.keras.layers.Dense(d_hidden, activation='relu'),
      tf.keras.layers.Dense(d_model)
  ])


class EncoderLayer(tf.keras.layers.Layer):
  """Encoder of transformer."""

  def __init__(self, d_model, num_heads, d_hidden, dropout_rate):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, d_hidden)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
    self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x: tf.Tensor, training: Optional[bool],
           mask: tf.Tensor) -> tf.Tensor:
    attn_output = self.mha(x, x, x, mask)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)

    ffn_output = self.ffn(out1)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)

    return out2


def positional_encoding(max_positions: int, d_model: int) -> tf.Tensor:
  """Returns all the possible positional encodings.

  Args:
    max_positions: Maximum number of positions.
    d_model: Dimension of features of MultiHeadAttention layers.

  Returns:
    The position encodings of the input sequence.
  """

  def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(DEFAULT_POSITIONAL_BASE,
                               (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates

  angle_rads = get_angles(
      np.arange(max_positions)[:, np.newaxis],
      np.arange(d_model)[np.newaxis, :], d_model)

  # Apply sin to even indices in the array; 2i.
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # Apply cos to odd indices in the array; 2i+1.
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)


class TransformerLM(tf.keras.layers.Layer):
  """Transformer for next word prediction."""

  def __init__(self, num_layers: int, d_embed: int, d_model: int,
               num_heads: int, d_hidden: int, input_vocab_size: int,
               maximum_position_encoding: int, dropout_rate: float):
    super(TransformerLM, self).__init__()

    self.d_model = d_model
    self.d_embed = d_embed
    self.num_layers = num_layers

    # Set mask_zero to True to be consistent with the LSTM model
    # for StackOverflow in TFF.
    self.embedding = tf.keras.layers.Embedding(
        input_vocab_size, d_embed, mask_zero=True)

    self.pos_encoding = positional_encoding(maximum_position_encoding,
                                            self.d_embed)
    self.embedding_proj = tf.keras.layers.Dense(d_model)

    self.enc_layers = [
        EncoderLayer(d_model, num_heads, d_hidden, dropout_rate)
        for _ in range(num_layers)
    ]

    self.embedding_out_proj = tf.keras.layers.Dense(d_embed)
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
    seq_len = tf.shape(x)[1]

    mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)

    x = self.embedding(x)
    x *= tf.math.sqrt(tf.cast(self.d_embed, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]
    x = self.embedding_proj(x)

    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)

    x = self.embedding_out_proj(x)

    # Shape of output: (batch_size, input_seq_len, d_embed)
    return x


def create_transformer_lm(vocab_size: int = 10000,
                          num_oov_buckets: int = 1,
                          dim_embed: int = 96,
                          dim_model: int = 512,
                          dim_hidden: int = 2048,
                          num_heads: int = 8,
                          num_layers: int = 1,
                          max_position_encoding: int = 1000,
                          dropout: float = 0.1,
                          name='transformer_lm') -> tf.keras.Model:
  """Create the transformer-based language model for next-token prediction.

  Args:
    vocab_size: Vocab size for normal tokens.
    num_oov_buckets: Number of out of vocabulary buckets.
    dim_embed: Dimension of the token embeddings.
    dim_model: Dimension of features of MultiHeadAttention layers.
    dim_hidden: Dimension of hidden layers of the FFN.
    num_heads: Number of attention heads.
    num_layers: Number of Transformer blocks.
    max_position_encoding: Maximum number of positions for position embeddings.
    dropout: Dropout rate.
    name: Name of the model.

  Returns:
    A transformer model.
  """
  if max_position_encoding > DEFAULT_POSITIONAL_BASE:
    raise ValueError(
        'The maximum position cannot exceed the default positional base {}'
        .format(DEFAULT_POSITIONAL_BASE))

  extended_vocab_size = vocab_size + 3 + num_oov_buckets  # For pad/bos/eos/oov.
  inputs = tf.keras.layers.Input(shape=(None,))
  transformer = TransformerLM(
      num_layers,
      dim_embed,
      dim_model,
      num_heads,
      dim_hidden,
      extended_vocab_size,
      max_position_encoding,
      dropout_rate=dropout)
  features = transformer(inputs)

  # Use shared embedding by default. Put it outside TransformerLM because of
  # the initialization of transformer.embedding.embeddings.
  transpose_embedding = TransposableEmbedding(transformer.embedding)
  logits = transpose_embedding(features)

  return tf.keras.Model(inputs=inputs, outputs=logits, name=name)
