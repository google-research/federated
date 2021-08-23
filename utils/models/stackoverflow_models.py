# Copyright 2019, Google LLC.
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
"""Sequence model functions for research baselines."""

import functools
from typing import Optional

import tensorflow as tf


class TransposableEmbedding(tf.keras.layers.Layer):
  """A Keras layer implements a transposed projection for output."""

  def __init__(self, embedding_layer: tf.keras.layers.Embedding):
    super().__init__()
    self.embeddings = embedding_layer.embeddings

  # Placing `tf.matmul` under the `call` method is important for backpropagating
  # the gradients of `self.embeddings` in graph mode.
  def call(self, inputs):
    return tf.matmul(inputs, self.embeddings, transpose_b=True)


def create_recurrent_model(vocab_size: int = 10000,
                           num_oov_buckets: int = 1,
                           embedding_size: int = 96,
                           latent_size: int = 670,
                           num_layers: int = 1,
                           name: str = 'rnn',
                           shared_embedding: bool = False,
                           seed: Optional[int] = None):
  """Constructs zero-padded keras model with the given parameters and cell.

  Args:
    vocab_size: Size of vocabulary to use.
    num_oov_buckets: Number of out of vocabulary buckets.
    embedding_size: The size of the embedding.
    latent_size: The size of the recurrent state.
    num_layers: The number of layers.
    name: (Optional) string to name the returned `tf.keras.Model`.
    shared_embedding: (Optional) Whether to tie the input and output
      embeddings.
    seed: A random seed governing the model initialization and layer randomness.
      If not `None`, then the global random seed will be set before constructing
      the tensor initializer, in order to guarantee the same model is produced.

  Returns:
    `tf.keras.Model`.
  """
  if seed is not None:
    tf.random.set_seed(seed)
  extended_vocab_size = vocab_size + 3 + num_oov_buckets  # For pad/bos/eos/oov.
  inputs = tf.keras.layers.Input(shape=(None,))
  input_embedding = tf.keras.layers.Embedding(
      input_dim=extended_vocab_size,
      output_dim=embedding_size,
      mask_zero=True,
      embeddings_initializer=tf.keras.initializers.RandomUniform(seed=seed),
  )
  embedded = input_embedding(inputs)
  projected = embedded

  lstm_layer_builder = functools.partial(
      tf.keras.layers.LSTM,
      units=latent_size,
      return_sequences=True,
      recurrent_initializer=tf.keras.initializers.Orthogonal(seed=seed),
      kernel_initializer=tf.keras.initializers.HeNormal(seed=seed))

  dense_layer_builder = functools.partial(
      tf.keras.layers.Dense,
      kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed))

  for _ in range(num_layers):
    layer = lstm_layer_builder()
    processed = layer(projected)
    # A projection changes dimension from rnn_layer_size to input_embedding_size
    dense_layer = dense_layer_builder(units=embedding_size)
    projected = dense_layer(processed)

  if shared_embedding:
    transposed_embedding = TransposableEmbedding(input_embedding)
    logits = transposed_embedding(projected)
  else:
    final_dense_layer = dense_layer_builder(
        units=extended_vocab_size, activation=None)
    logits = final_dense_layer(projected)

  return tf.keras.Model(inputs=inputs, outputs=logits, name=name)
