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

from absl.testing import absltest
import tensorflow as tf

from dual_encoder import keras_layers

l2_normalize_fn = lambda x: tf.keras.backend.l2_normalize(x, axis=-1)


class KerasLayersTest(absltest.TestCase):

  def test_masked_average_3d(self):
    masked_average_layer = keras_layers.MaskedAverage(1)

    inputs = tf.constant([
        [[0.5, 0.3], [0.4, 0.1], [0.4, 0.1]],
        [[0.6, 0.8], [0.5, 0.4], [0.4, 0.1]],
        [[0.9, 0.4], [0.4, 0.1], [0.4, 0.1]],
        [[0.9, 0.4], [0.4, 0.1], [0.4, 0.1]],
    ])
    mask = tf.constant([[True, True, True],
                        [False, False, True],
                        [True, False, False],
                        [False, False, False]])
    output_average = masked_average_layer.call(inputs, mask=mask)
    output_mask = masked_average_layer.compute_mask(inputs, mask=mask)

    expected_average = tf.constant([
        [1.3 / 3, 0.5 / 3],
        [0.4, 0.1],
        [0.9, 0.4],
        [0.0, 0.0]
    ])
    expected_mask = None

    tf.debugging.assert_near(expected_average, output_average)
    self.assertEqual(expected_mask, output_mask)

  def test_masked_average_4d(self):
    masked_average_layer = keras_layers.MaskedAverage(2)

    inputs = tf.constant([
        [[[0.5, 0.3], [0.4, 0.1], [0.4, 0.1]],
         [[0.6, 0.8], [0.5, 0.4], [0.4, 0.1]]],
        [[[0.6, 0.8], [0.5, 0.4], [0.4, 0.1]],
         [[0.6, 0.8], [0.5, 0.4], [0.4, 0.1]]],
        [[[0.9, 0.4], [0.4, 0.1], [0.4, 0.1]],
         [[0.6, 0.8], [0.5, 0.4], [0.4, 0.1]]],
        [[[0.9, 0.4], [0.4, 0.1], [0.4, 0.1]],
         [[0.6, 0.8], [0.5, 0.4], [0.4, 0.1]]],
    ])
    mask = tf.constant([[[True, True, True], [True, False, True]],
                        [[False, False, True], [False, False, False]],
                        [[True, False, False], [True, True, True]],
                        [[False, False, False], [True, False, False]]])
    output_average = masked_average_layer.call(inputs, mask=mask)
    output_mask = masked_average_layer.compute_mask(inputs, mask=mask)

    expected_average = tf.constant([
        [[1.3 / 3, 0.5 / 3], [0.5, 0.45]],
        [[0.4, 0.1], [0.0, 0.0]],
        [[0.9, 0.4], [0.5, 1.3 / 3]],
        [[0.0, 0.0], [0.6, 0.8]],
    ])
    expected_mask = tf.constant([[True, True],
                                 [True, False],
                                 [True, True],
                                 [False, True]])

    tf.debugging.assert_near(expected_average, output_average)
    tf.debugging.assert_equal(expected_mask, output_mask)

  def test_masked_average_raises_error(self):
    masked_average_layer = keras_layers.MaskedAverage(1)

    inputs = tf.constant([
        [[0.5, 0.3], [0.4, 0.1], [0.4, 0.1]],
        [[0.6, 0.8], [0.5, 0.4], [0.4, 0.1]],
        [[0.9, 0.4], [0.4, 0.1], [0.4, 0.1]],
    ])
    mask = None

    with self.assertRaises(ValueError):
      masked_average_layer.call(inputs, mask=mask)

    with self.assertRaises(ValueError):
      masked_average_layer.compute_mask(inputs, mask=mask)

  def test_masked_average_get_config(self):
    masked_average_layer = keras_layers.MaskedAverage(1)
    config = masked_average_layer.get_config()
    self.assertEqual(config['axis'], 1)

  def test_masked_reshape(self):
    masked_reshape_layer = keras_layers.MaskedReshape((4, 4, 2, 1), (4, 4, 2))

    inputs = tf.constant([
        [[1.0], [2.0], [0.5], [0.4], [0.4], [0.1], [0.0], [0.0]],
        [[0.4], [0.1], [0.0], [0.0], [0.0], [0.0], [0.6], [0.8]],
        [[0.9], [0.4], [0.5], [3.0], [0.9], [0.4], [0.5], [3.0]],
        [[0.0], [0.0], [0.6], [0.8], [0.4], [0.1], [0.0], [0.0]],
    ])
    mask = tf.constant(
        [[True, False, True, True, True, False, False, False],
         [True, False, True, True, True, True, False, True],
         [False, True, True, False, True, True, True, True],
         [False, True, True, True, True, False, False, True]])

    output = masked_reshape_layer.call(inputs, mask=mask)
    output_mask = masked_reshape_layer.compute_mask(inputs, mask=mask)

    expected_output = tf.constant([
        [[[1.0], [2.0]], [[0.5], [0.4]], [[0.4], [0.1]], [[0.0], [0.0]]],
        [[[0.4], [0.1]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.6], [0.8]]],
        [[[0.9], [0.4]], [[0.5], [3.0]], [[0.9], [0.4]], [[0.5], [3.0]]],
        [[[0.0], [0.0]], [[0.6], [0.8]], [[0.4], [0.1]], [[0.0], [0.0]]],
    ])
    expected_mask = tf.constant(
        [[[True, False], [True, True], [True, False], [False, False]],
         [[True, False], [True, True], [True, True], [False, True]],
         [[False, True], [True, False], [True, True], [True, True]],
         [[False, True], [True, True], [True, False], [False, True]]])

    tf.debugging.assert_near(expected_output, output)
    tf.debugging.assert_equal(expected_mask, output_mask)

  def test_masked_reshape_unknown_batch_size(self):
    masked_reshape_layer = keras_layers.MaskedReshape((-1, 4, 2, 1), (-1, 4, 2))

    inputs = tf.constant([
        [[1.0], [2.0], [0.5], [0.4], [0.4], [0.1], [0.0], [0.0]],
        [[0.4], [0.1], [0.0], [0.0], [0.0], [0.0], [0.6], [0.8]],
        [[0.9], [0.4], [0.5], [3.0], [0.9], [0.4], [0.5], [3.0]],
        [[0.0], [0.0], [0.6], [0.8], [0.4], [0.1], [0.0], [0.0]],
    ])
    mask = tf.constant(
        [[True, False, True, True, True, False, False, False],
         [True, False, True, True, True, True, False, True],
         [False, True, True, False, True, True, True, True],
         [False, True, True, True, True, False, False, True]])

    output = masked_reshape_layer.call(inputs, mask=mask)
    output_mask = masked_reshape_layer.compute_mask(inputs, mask=mask)

    expected_output = tf.constant([
        [[[1.0], [2.0]], [[0.5], [0.4]], [[0.4], [0.1]], [[0.0], [0.0]]],
        [[[0.4], [0.1]], [[0.0], [0.0]], [[0.0], [0.0]], [[0.6], [0.8]]],
        [[[0.9], [0.4]], [[0.5], [3.0]], [[0.9], [0.4]], [[0.5], [3.0]]],
        [[[0.0], [0.0]], [[0.6], [0.8]], [[0.4], [0.1]], [[0.0], [0.0]]],
    ])
    expected_mask = tf.constant(
        [[[True, False], [True, True], [True, False], [False, False]],
         [[True, False], [True, True], [True, True], [False, True]],
         [[False, True], [True, False], [True, True], [True, True]],
         [[False, True], [True, True], [True, False], [False, True]]])

    tf.debugging.assert_near(expected_output, output)
    tf.debugging.assert_equal(expected_mask, output_mask)

  def test_masked_reshape_raises_error(self):
    masked_reshape_layer = keras_layers.MaskedReshape((-1, 4, 2, 1), (-1, 4, 2))

    inputs = tf.constant([
        [[1.0], [2.0], [0.5], [0.4], [0.4], [0.1], [0.0], [0.0]],
        [[0.4], [0.1], [0.0], [0.0], [0.0], [0.0], [0.6], [0.8]],
        [[0.9], [0.4], [0.5], [3.0], [0.9], [0.4], [0.5], [3.0]],
        [[0.0], [0.0], [0.6], [0.8], [0.4], [0.1], [0.0], [0.0]],
    ])
    mask = None

    with self.assertRaises(ValueError):
      masked_reshape_layer.call(inputs, mask=mask)

    with self.assertRaises(ValueError):
      masked_reshape_layer.compute_mask(inputs, mask=mask)

  def test_masked_reshape_get_config(self):
    masked_reshape_layer = keras_layers.MaskedReshape((-1, 4, 2, 1), (-1, 4, 2))
    config = masked_reshape_layer.get_config()
    self.assertEqual(config['new_inputs_shape'], (-1, 4, 2, 1))
    self.assertEqual(config['new_mask_shape'], (-1, 4, 2))

  def test_embedding_spreadout_regularizer_dot_product(self):
    weights = tf.constant(
        [[1.0, 0.0, 0.0],
         [2.0, 2.0, 2.0],
         [0.1, 0.2, 0.3],
         [0.3, 0.2, 0.1],
         [0.0, 1.0, 0.0]])

    regularizer = keras_layers.EmbeddingSpreadoutRegularizer(
        spreadout_lambda=0.1,
        normalization_fn=None,
        l2_regularization=0.0)

    # Similarities without diagonal looks like:
    # 0.0 2.0 0.1 0.3 0.0
    # 2.0 0.0 1.2 1.2 2.0
    # 0.1 1.2 0.0 0.1 0.2
    # 0.3 1.2 0.1 0.0 0.2
    # 0.0 2.0 0.2 0.2 0.0

    loss = regularizer(weights)
    # L2 norm of above similarities.
    expected_loss = 0.47053161424

    tf.debugging.assert_near(expected_loss, loss)

    regularizer = keras_layers.EmbeddingSpreadoutRegularizer(
        spreadout_lambda=0.1,
        normalization_fn=None,
        l2_regularization=1.0)
    l2_regularizer = tf.keras.regularizers.l2(1.0)

    loss = regularizer(weights)
    expected_loss = 0.47053161424 + l2_regularizer(weights)

    tf.debugging.assert_near(expected_loss, loss)

  def test_embedding_spreadout_regularizer_cosine_similarity(self):
    weights = tf.constant(
        [[1.0, 0.0, 0.0],
         [2.0, 2.0, 2.0],
         [0.1, 0.2, 0.3],
         [0.3, 0.2, 0.1],
         [0.0, 1.0, 0.0]])

    regularizer = keras_layers.EmbeddingSpreadoutRegularizer(
        spreadout_lambda=0.1,
        normalization_fn=l2_normalize_fn,
        l2_regularization=0.0)

    loss = regularizer(weights)
    # L2 norm of above similarities.
    expected_loss = 0.2890284

    tf.debugging.assert_near(expected_loss, loss)

    regularizer = keras_layers.EmbeddingSpreadoutRegularizer(
        spreadout_lambda=0.1,
        normalization_fn=l2_normalize_fn,
        l2_regularization=1.0)
    l2_regularizer = tf.keras.regularizers.l2(1.0)

    loss = regularizer(weights)
    expected_loss = 0.2890284 + l2_regularizer(weights)

    tf.debugging.assert_near(expected_loss, loss)

  def test_embedding_spreadout_regularizer_no_spreadout(self):
    weights = tf.constant(
        [[1.0, 0.0, 0.0],
         [2.0, 2.0, 2.0],
         [0.1, 0.2, 0.3],
         [0.3, 0.2, 0.1],
         [0.0, 1.0, 0.0]])

    regularizer = keras_layers.EmbeddingSpreadoutRegularizer(
        spreadout_lambda=0.0,
        normalization_fn=None,
        l2_regularization=0.0)

    loss = regularizer(weights)
    expected_loss = 0.0

    tf.debugging.assert_near(expected_loss, loss)

    # Test that L2 normalization behaves normally.
    regularizer = keras_layers.EmbeddingSpreadoutRegularizer(
        spreadout_lambda=0.0,
        normalization_fn=None,
        l2_regularization=0.1)
    l2_regularizer = tf.keras.regularizers.l2(0.1)

    loss = regularizer(weights)
    l2_loss = l2_regularizer(weights)

    tf.debugging.assert_near(l2_loss, loss)

    # Test that normalization_fn has no effect.
    regularizer = keras_layers.EmbeddingSpreadoutRegularizer(
        spreadout_lambda=0.0,
        normalization_fn=l2_normalize_fn,
        l2_regularization=0.1)
    l2_regularizer = tf.keras.regularizers.l2(0.1)

    loss = regularizer(weights)
    l2_loss = l2_regularizer(weights)

    tf.debugging.assert_near(l2_loss, loss)

  def test_embedding_spreadout_regularizer_get_config(self):
    weights = tf.constant(
        [[1.0, 0.0, 0.0],
         [2.0, 2.0, 2.0],
         [0.1, 0.2, 0.3],
         [0.3, 0.2, 0.1],
         [0.0, 1.0, 0.0]])
    regularizer = keras_layers.EmbeddingSpreadoutRegularizer(
        spreadout_lambda=0.0,
        normalization_fn=l2_normalize_fn,
        l2_regularization=0.1)

    config = regularizer.get_config()
    expected_config = {
        'spreadout_lambda': 0.0,
        'normalization_fn': l2_normalize_fn,
        'l2_regularization': 0.1
    }

    new_regularizer = (
        keras_layers.EmbeddingSpreadoutRegularizer.from_config(config))
    l2_regularizer = tf.keras.regularizers.l2(0.1)

    loss = new_regularizer(weights)
    l2_loss = l2_regularizer(weights)

    self.assertEqual(config, expected_config)
    tf.debugging.assert_near(l2_loss, loss)

if __name__ == '__main__':
  absltest.main()
