# Copyright 2020, Google LLC.
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
"""The Transformer model for research baselines. Reference: https://arxiv.org/abs/1706.03762"""

import tensorflow as tf
import numpy as np
from utils.models.stackoverflow_models import TransposableEmbedding

DEFAULT_LARGE_NEGATIVE = -1e9


def scaled_dot_product_attention(query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, mask: tf.Tensor):
  """Calculate the attention weights.

  q (query), k (key), v (value) must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    query: Query feature vectors, shape == (..., seq_len_q, depth).
    key: Key feature vectors, shape == (..., seq_len_k, depth).
    value: Value feature vectors, shape == (..., seq_len_v, depth_v).
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.
  Returns:
    output: The output attention vectors.
    attention_weights: The attention weights.
  """

  matmul_qk = tf.matmul(query, key, transpose_b=True)

  # scale matmul_qk by 1/sqrt(ftr_dim), so that the softmax does not vanish when feature dimension becoming higher.
  ftr_dim = tf.cast(tf.shape(key)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(ftr_dim)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * DEFAULT_LARGE_NEGATIVE)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, value)  # (..., seq_len_q, depth_v)

  return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
  """The multihead attention layer.
  """
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model  #scalar value of model dimension

    if d_model % self.num_heads != 0:
      raise ValueError('Feature dimension should be divisible by number of heads! Got {}/{}'.format(d_model, num_heads))

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
      q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
  """Returns all the possible positional encodings.
  Args:
      d_model: Dimension of the input feature.
      dff: Dimension of the hidden layer.
  Returns:
      `tf.keras.Sequential`: A one-hidden-layer MLP.
  """
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])


class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask):
    attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

    return out2


def positional_encoding(position, d_model):
  """Returns all the possible positional encodings. 
  Args:
        position: Maximum number of positions.
        d_model: Dimension of features of MultiHeadAttention layers.
  Returns:
      `tf.Tensor`: The position encodings of the input sequence.
  """
  def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(position, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)    # Return a constant tensor, so should work in tf.function


class TransformerLM(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_embed, d_model, num_heads, dff, input_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(TransformerLM, self).__init__()

    self.d_model = d_model
    self.d_embed = d_embed
    self.num_layers = num_layers

    # mask_zero set to True to be consistent with the default setting of LSTM
    self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_embed, mask_zero=True)

    self.pos_encoding = positional_encoding(maximum_position_encoding,
                                            self.d_embed)
    self.embedding_proj = tf.keras.layers.Dense(d_model)

    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                       for _ in range(num_layers)]

    self.embedding_out_proj = tf.keras.layers.Dense(d_embed)
    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, training=None):
    seq_len = tf.shape(x)[1]

    mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)

    # adding embedding and position encoding.
    x = self.embedding(x)  # (batch_size, input_seq_len, d_embed)
    x *= tf.math.sqrt(tf.cast(self.d_embed, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]
    x = self.embedding_proj(x)

    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)

    x = self.embedding_out_proj(x)
    return x  # (batch_size, input_seq_len, d_embed)


def create_transformer_lm(vocab_size=10000,
                          num_oov_buckets=1,
                          d_embed=96,
                          d_model=512,
                          dff=2048,
                          num_heads=8,
                          num_layers=1,
                          max_position_encoding=10000,
                          dropout=0.1,
                          name='transformer_lm'):
  """Create the transformer-based language model for next-token prediction.
  Args:
      vocab_size: Vocab size for normal tokens.
      num_oov_buckets: Number of out of vocabulary buckets.
      d_embed: Dimension of the token embeddings.
      d_model: Dimension of features of MultiHeadAttention layers.
      dff: Dimension of hidden layers of the FFN.
      num_heads: Number of attention heads.
      num_layers: Number of Transformer blocks.
      max_position_encoding: Maximum number of positions for position embeddings.
      dropout: Dropout rate.
      name: Name of the model.
  Returns:
    `tf.keras.Model`.
  """
  extended_vocab_size = vocab_size + 3 + num_oov_buckets  # For pad/bos/eos/oov.
  inputs = tf.keras.layers.Input(shape=(None,))
  transformer = TransformerLM(
    num_layers, d_embed, d_model, num_heads, dff, extended_vocab_size,
    max_position_encoding, rate=dropout)
  features = transformer(inputs)

  # use shared embedding by default. Have to put it here
  # due to initialization of transformer.embedding.embeddings
  transpose_embedding = TransposableEmbedding(transformer.embedding)
  logits = transpose_embedding(features)

  return tf.keras.Model(inputs=inputs, outputs=logits, name=name)
