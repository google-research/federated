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
"""Tests for models."""

from absl.testing import absltest
import tensorflow as tf

from reconstruction.stackoverflow import models


class PassMask(tf.keras.layers.Layer):
  """Keras layer returning its mask.

  This class is used to test the mask propagation after summing the local and
  global embeddings.
  """

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.supports_masking = True

  def call(self, inputs, mask=None):
    return mask


class ModelsTest(tf.test.TestCase):

  def test_global_embedding_input_in_range(self):

    vocab_size = 2
    total_vocab_size = vocab_size + 3  # +3 for pad/bos/eos.
    embedding_size = 3
    num_oov_buckets = 2

    inputs = tf.constant([[1], [2]], dtype=tf.int64)

    global_embedding = models.GlobalEmbedding(
        total_vocab_size=total_vocab_size,
        embedding_dim=embedding_size,
        mask_zero=True,
        name='global_embedding_layer')

    local_embedding = models.LocalEmbedding(
        input_dim=num_oov_buckets,
        embedding_dim=embedding_size,
        total_vocab_size=total_vocab_size,
        mask_zero=True,
        name='local_embedding_layer')

    global_output = global_embedding(inputs)
    local_output = local_embedding(inputs)

    self.assertNotAllEqual(global_output, tf.zeros_like(input=global_output))
    self.assertAllEqual(local_output, tf.zeros_like(input=local_output))

  def test_global_embedding_input_out_of_range(self):

    vocab_size = 2
    total_vocab_size = vocab_size + 3  # +3 for pad/bos/eos.
    embedding_size = 3
    num_oov_buckets = 2

    inputs = tf.constant([[5], [6]], dtype=tf.int64)

    global_embedding = models.GlobalEmbedding(
        total_vocab_size=total_vocab_size,
        embedding_dim=embedding_size,
        mask_zero=True,
        name='global_embedding_layer')

    local_embedding = models.LocalEmbedding(
        input_dim=num_oov_buckets,
        embedding_dim=embedding_size,
        total_vocab_size=total_vocab_size,
        mask_zero=True,
        name='local_embedding_layer')

    global_output = global_embedding(inputs)
    local_output = local_embedding(inputs)

    self.assertAllEqual(global_output, tf.zeros_like(input=global_output))
    self.assertNotAllEqual(local_output, tf.zeros_like(input=local_output))

  def test_create_recurrent_model_with_oov(self):
    reconstruction_model = models.create_recurrent_reconstruction_model(
        vocab_size=100,
        num_oov_buckets=10,
        embedding_size=96,
        latent_size=67,
        num_layers=1,
        name='rnn_recon_embeddings_with_oov')
    # Check global/local trainable variables.
    local_trainable_variable_names = [
        var.name for var in reconstruction_model.local_trainable_variables
    ]
    global_trainable_variable_names = [
        var.name for var in reconstruction_model.global_trainable_variables
    ]

    self.assertEmpty(
        reconstruction_model.local_non_trainable_variables,
        msg='Expected local_non_trainable_variables to be empty.')
    self.assertEmpty(
        reconstruction_model.global_non_trainable_variables,
        msg='Expected global_non_trainable_variables to be empty.')

    expected_global_variable_names = [
        'global_embedding_layer/global_embedding:0',
        'lstm_0/lstm_cell/kernel:0', 'lstm_0/lstm_cell/recurrent_kernel:0',
        'lstm_0/lstm_cell/bias:0', 'projection_0/kernel:0',
        'projection_0/bias:0', 'last_layer/kernel:0', 'last_layer/bias:0'
    ]
    self.assertSequenceEqual(global_trainable_variable_names,
                             expected_global_variable_names)

    expected_local_variable_names = ['local_embedding_layer/local_embedding:0']
    self.assertSequenceEqual(local_trainable_variable_names,
                             expected_local_variable_names)

  def test_create_recurrent_model_with_oov_all_global(self):
    reconstruction_model = models.create_recurrent_reconstruction_model(
        vocab_size=100,
        num_oov_buckets=10,
        embedding_size=96,
        latent_size=67,
        num_layers=1,
        global_variables_only=True,
        name='rnn_recon_embeddings_with_oov')
    # Check global/local trainable variables.
    local_trainable_variable_names = [
        var.name for var in reconstruction_model.local_trainable_variables
    ]
    global_trainable_variable_names = [
        var.name for var in reconstruction_model.global_trainable_variables
    ]

    self.assertEmpty(
        reconstruction_model.local_non_trainable_variables,
        msg='Expected local_non_trainable_variables to be empty.')
    self.assertEmpty(
        reconstruction_model.global_non_trainable_variables,
        msg='Expected global_non_trainable_variables to be empty.')

    expected_global_variable_names = [
        'global_embedding_layer/global_embedding:0',
        'lstm_0/lstm_cell/kernel:0', 'lstm_0/lstm_cell/recurrent_kernel:0',
        'lstm_0/lstm_cell/bias:0', 'projection_0/kernel:0',
        'projection_0/bias:0', 'last_layer/kernel:0', 'last_layer/bias:0',
        'local_embedding_layer/local_embedding:0'
    ]
    self.assertSequenceEqual(global_trainable_variable_names,
                             expected_global_variable_names)
    self.assertEmpty(
        local_trainable_variable_names,
        msg='Expected local_trainable_variables to be empty.')

  def test_negative_oov_raises_exception(self):
    with self.assertRaisesRegex(ValueError, 'out of vocabulary buckets'):
      models.create_recurrent_reconstruction_model(
          vocab_size=10,
          num_oov_buckets=-2,
          embedding_size=96,
          latent_size=67,
          num_layers=1,
          name='rnn_recon_embeddings_with_oov')

  def test_negative_vocab_size_raises_exception(self):
    with self.assertRaisesRegex(ValueError, 'vocab_size'):
      models.create_recurrent_reconstruction_model(
          vocab_size=-5,
          num_oov_buckets=20,
          embedding_size=96,
          latent_size=67,
          num_layers=1,
          name='rnn_recon_embeddings_with_oov')

  def test_embedding_mask(self):
    vocab_size = 2
    total_vocab_size = vocab_size + 3  # +3 for pad/bos/eos.
    embedding_size = 3
    num_oov_buckets = 2

    inputs = tf.constant([[1], [2], [0], [0]], dtype=tf.int64)

    global_embedding = models.GlobalEmbedding(
        total_vocab_size=total_vocab_size,
        embedding_dim=embedding_size,
        mask_zero=True,
        name='global_embedding_layer')

    local_embedding = models.LocalEmbedding(
        input_dim=num_oov_buckets,
        embedding_dim=embedding_size,
        total_vocab_size=total_vocab_size,
        mask_zero=True,
        name='local_embedding_layer')

    self.assertAllEqual(
        global_embedding.compute_mask(inputs),
        [[True], [True], [False], [False]])
    self.assertAllEqual(
        local_embedding.compute_mask(inputs),
        [[True], [True], [False], [False]])

  def test_add_embeddings(self):
    vocab_size = 2
    total_vocab_size = vocab_size + 3  # +3 for pad/bos/eos.
    embedding_size = 3
    num_oov_buckets = 2

    inputs = tf.constant([[1], [0], [5], [0]], dtype=tf.int64)

    global_embedding = models.GlobalEmbedding(
        total_vocab_size=total_vocab_size,
        embedding_dim=embedding_size,
        mask_zero=True,
        name='global_embedding_layer')

    local_embedding = models.LocalEmbedding(
        input_dim=num_oov_buckets,
        embedding_dim=embedding_size,
        total_vocab_size=total_vocab_size,
        mask_zero=True,
        name='local_embedding_layer')

    projected = tf.keras.layers.Add()(
        [global_embedding(inputs),
         local_embedding(inputs)])

    projected = PassMask()(projected)

    self.assertAllEqual(projected, [[True], [False], [True], [False]])

  def test_project_on_known_embeddings(self):
    vocab_size = 2
    total_vocab_size = vocab_size + 3  # +3 for pad/bos/eos.
    embedding_size = 3
    num_oov_buckets = 1

    inputs = tf.constant([[1], [0], [2], [0], [5]], dtype=tf.int64)

    global_embedding = models.GlobalEmbedding(
        total_vocab_size=total_vocab_size,
        embedding_dim=embedding_size,
        mask_zero=True,
        initializer=tf.keras.initializers.Constant([[1, 1, 1], [2, 2, 2],
                                                    [3, 3, 3], [4, 4, 4],
                                                    [5, 5, 5]]),
        name='global_embedding_layer')

    local_embedding = models.LocalEmbedding(
        input_dim=num_oov_buckets,
        embedding_dim=embedding_size,
        total_vocab_size=total_vocab_size,
        mask_zero=True,
        initializer=tf.keras.initializers.Constant([[6, 6, 6]]),
        name='local_embedding_layer')

    global_output = global_embedding(inputs)
    local_output = local_embedding(inputs)

    self.assertAllEqual(
        global_output,
        [[[2, 2, 2]], [[1, 1, 1]], [[3, 3, 3]], [[1, 1, 1]], [[0, 0, 0]]])
    self.assertAllEqual(
        local_output,
        [[[0, 0, 0]], [[0, 0, 0]], [[0, 0, 0]], [[0, 0, 0]], [[6, 6, 6]]])


if __name__ == '__main__':
  absltest.main()
