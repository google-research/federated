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

import tensorflow as tf

from dual_encoder import encoders


class EncodersTest(tf.test.TestCase):

  def count_encoder_variables(self, encoder):
    """Counts an encoder's trainable variables."""
    layers = list(encoder.global_layers) + list(encoder.local_layers)

    trainable_variables = 0
    for layer in layers:
      trainable_variables += len(layer.trainable_variables)
    return trainable_variables

  def setUp(self):
    super().setUp()
    self.item_embedding_layer = tf.keras.layers.Embedding(
        10,
        5,
        mask_zero=True,
        embeddings_initializer='ones',
        name='ItemEmbeddings')
    self.test_sequence_batch = tf.constant([[2, 3], [4, 5]])
    self.test_item_batch = tf.constant([[0], [1]])

  def test_embedding_bow_encoder_call_no_hidden_layers(self):
    """Ensures encoder output matches expectations."""
    encoder = encoders.EmbeddingBOWEncoder(
        item_embedding_layer=self.item_embedding_layer,
        hidden_dims=None,
        hidden_activations=None,
        layer_name_prefix='Context')

    encoder_output = encoder.call(self.test_sequence_batch)
    self.assertEqual(encoder_output.dtype, tf.float32)
    self.assertEqual(encoder_output.shape, [2, 5])

  def test_embedding_bow_encoder_call_hidden_layers(self):
    """Ensures encoder output matches expectations."""
    encoder = encoders.EmbeddingBOWEncoder(
        item_embedding_layer=self.item_embedding_layer,
        hidden_dims=[1, 2, 3, 10],
        hidden_activations=['relu', None, 'relu', 'relu'],
        layer_name_prefix='Context')

    encoder_output = encoder.call(self.test_sequence_batch)
    self.assertEqual(encoder_output.dtype, tf.float32)
    self.assertEqual(encoder_output.shape, [2, 10])

  def test_embedding_bow_encoder_init_fails_mask_zero(self):
    """Ensures encoder init fails as expected."""
    embedding_layer = tf.keras.layers.Embedding(
        10, 5, mask_zero=False, name='ItemEmbeddings')
    with self.assertRaisesRegex(ValueError, 'mask_zero'):
      encoders.EmbeddingBOWEncoder(
          item_embedding_layer=embedding_layer,
          hidden_dims=None,
          hidden_activations=None,
          layer_name_prefix='Context')

  def test_embedding_bow_encoder_init_fails_hidden_dims_None_mismatch(self):
    """Ensures encoder init fails as expected."""
    with self.assertRaisesRegex(ValueError, 'length'):
      encoders.EmbeddingBOWEncoder(
          item_embedding_layer=self.item_embedding_layer,
          hidden_dims=None,
          hidden_activations=['relu'],
          layer_name_prefix='Context')

  def test_embedding_bow_encoder_init_fails_hidden_dims_length_mismatch(self):
    """Ensures encoder init fails as expected."""
    with self.assertRaisesRegex(ValueError, 'length'):
      encoders.EmbeddingBOWEncoder(
          item_embedding_layer=self.item_embedding_layer,
          hidden_dims=[1, 2, 3],
          hidden_activations=['relu'],
          layer_name_prefix='Context')

  def test_embedding_bow_encoder_global_local_layers(self):
    """Ensure global and local layers are as expected."""
    encoder = encoders.EmbeddingBOWEncoder(
        item_embedding_layer=self.item_embedding_layer,
        hidden_dims=[1, 2, 3, 10],
        hidden_activations=['relu', None, 'relu', 'relu'],
        layer_name_prefix='Context')

    # Layer variables are lazily initialized during `call`.
    encoder.call(self.test_sequence_batch)

    global_layers = list(encoder.global_layers)
    local_layers = list(encoder.local_layers)

    self.assertLen(global_layers, 5)
    self.assertEmpty(local_layers)

    variable_count = self.count_encoder_variables(encoder)
    # 2 variables for each dense layer.
    self.assertEqual(variable_count, 8)

  def test_embedding_bow_encoder_layer_names(self):
    """Ensures layer name prefix is applied correctly."""
    encoder = encoders.EmbeddingBOWEncoder(
        item_embedding_layer=self.item_embedding_layer,
        hidden_dims=[1, 2, 3, 10],
        hidden_activations=['relu', None, 'relu', 'relu'],
        layer_name_prefix='Test')

    # Layer variables are lazily initialized during `call`.
    encoder.call(self.test_sequence_batch)

    global_layers = list(encoder.global_layers)
    local_layers = list(encoder.local_layers)
    layers = global_layers + local_layers

    for layer in layers:
      self.assertStartsWith(layer.name, 'Test', msg=layer.name)

  def test_embedding_encoder_call_no_hidden_layer(self):
    """Ensures encoder output matches expectations."""
    encoder = encoders.EmbeddingEncoder(
        item_embedding_layer=self.item_embedding_layer,
        hidden_dims=None,
        hidden_activations=None,
        layer_name_prefix='Context')

    encoder_output = encoder.call(self.test_item_batch)
    self.assertEqual(encoder_output.dtype, tf.float32)
    self.assertEqual(encoder_output.shape, [2, 5])

  def test_embedding_encoder_call_hidden_layers(self):
    """Ensures encoder output matches expectations."""
    encoder = encoders.EmbeddingEncoder(
        item_embedding_layer=self.item_embedding_layer,
        hidden_dims=[1, 2, 3, 10],
        hidden_activations=['relu', None, 'relu', 'relu'],
        layer_name_prefix='Context')

    encoder_output = encoder.call(self.test_item_batch)
    self.assertEqual(encoder_output.dtype, tf.float32)
    self.assertEqual(encoder_output.shape, [2, 10])

  def test_embedding_encoder_init_fails_hidden_dims_None_mismatch(self):
    """Ensures encoder init fails as expected."""
    with self.assertRaisesRegex(ValueError, 'length'):
      encoders.EmbeddingEncoder(
          item_embedding_layer=self.item_embedding_layer,
          hidden_dims=None,
          hidden_activations=['relu'],
          layer_name_prefix='Context')

  def test_embedding_encoder_init_fails_hidden_dims_length_mismatch(self):
    """Ensures encoder init fails as expected."""
    with self.assertRaisesRegex(ValueError, 'length'):
      encoders.EmbeddingEncoder(
          item_embedding_layer=self.item_embedding_layer,
          hidden_dims=[1, 2, 3],
          hidden_activations=['relu'],
          layer_name_prefix='Context')

  def test_embedding_encoder_global_local_layers(self):
    """Ensure global and local layers are as expected."""
    encoder = encoders.EmbeddingEncoder(
        item_embedding_layer=self.item_embedding_layer,
        hidden_dims=[1, 2, 3, 10],
        hidden_activations=['relu', None, 'relu', 'relu'],
        layer_name_prefix='Context')

    # Layer variables are lazily initialized during `call`.
    encoder.call(self.test_item_batch)

    global_layers = list(encoder.global_layers)
    local_layers = list(encoder.local_layers)

    self.assertLen(global_layers, 5)
    self.assertEmpty(local_layers)

    variable_count = self.count_encoder_variables(encoder)
    # 2 variables for each dense layer.
    self.assertEqual(variable_count, 8)

  def test_embedding_encoder_layer_names(self):
    """Ensures layer name prefix is applied correctly."""
    encoder = encoders.EmbeddingEncoder(
        item_embedding_layer=self.item_embedding_layer,
        hidden_dims=[1, 2, 3, 10],
        hidden_activations=['relu', None, 'relu', 'relu'],
        layer_name_prefix='Test')

    # Layer variables are lazily initialized during `call`.
    encoder.call(self.test_item_batch)

    global_layers = list(encoder.global_layers)
    local_layers = list(encoder.local_layers)
    layers = global_layers + local_layers

    for layer in layers:
      self.assertStartsWith(layer.name, 'Test', msg=layer.name)

  def test_init_dense_layers(self):
    layers = {'my_fake_layer': tf.keras.layers.Dense(10)}
    num_hidden_layers = encoders.init_dense_layers(
        layers, [20, 5, 2],
        hidden_activations=['relu', None, None],
        layer_name_prefix='Test')

    self.assertEqual(num_hidden_layers, 3)
    # 1 original layer and 3 new ones.
    self.assertLen(layers, 4)

    self.assertIsInstance(layers['my_fake_layer'], tf.keras.layers.Dense)
    self.assertIsInstance(layers['TestHidden1'], tf.keras.layers.Dense)
    self.assertIsInstance(layers['TestHidden2'], tf.keras.layers.Dense)
    self.assertIsInstance(layers['TestHidden3'], tf.keras.layers.Dense)

  def test_init_dense_layers_raises_error_mismatch(self):
    layers = {'my_fake_layer': tf.keras.layers.Dense(10)}
    with self.assertRaisesRegex(ValueError, 'same length'):
      encoders.init_dense_layers(
          layers, [20, 5],
          hidden_activations=['relu'],
          layer_name_prefix='Test')

  def test_apply_dense_layers(self):
    layers = {'my_fake_layer': tf.keras.layers.Dense(10)}
    num_hidden_layers = encoders.init_dense_layers(
        layers, [20, 5, 2],
        hidden_activations=['relu', None, None],
        layer_name_prefix='Test')
    output = encoders.apply_dense_layers(
        tf.constant([[2.0, 3.0], [1.0, 2.0]], tf.float32), layers,
        num_hidden_layers, 'Test')

    self.assertEqual(output.dtype, tf.float32)
    self.assertEqual(output.shape, [2, 2])


if __name__ == '__main__':
  tf.test.main()
