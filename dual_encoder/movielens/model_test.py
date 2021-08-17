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

from dual_encoder import encoders
from dual_encoder import model_utils as utils
from dual_encoder.movielens import model as model_lib


class ModelLauncherUtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self.item_embedding_dim = 8
    self.final_embedding_dim = 10
    self.item_vocab_size = 8
    self.embedding_vocab_size = self.item_vocab_size + 1
    self.normalization_fn = utils.l2_normalize_fn
    self.batch_size = 16
    self.input_context = tf.keras.layers.Input(
        shape=(None,), batch_size=self.batch_size, name='InputContext')
    self.input_label = tf.keras.layers.Input(
        shape=(1,), batch_size=self.batch_size, name='InputLabel')
    self.inputs = {'context': self.input_context, 'label': self.input_label}
    self.context_input_shape = (None,)
    self.label_input_shape = (1,)
    self.item_embedding_layer = tf.keras.layers.Embedding(
        self.embedding_vocab_size,
        self.item_embedding_dim,
        mask_zero=True,
        embeddings_initializer='ones',
        name='ItemEmbeddings')
    self.context_encoder = encoders.EmbeddingBOWEncoder(
        item_embedding_layer=self.item_embedding_layer,
        hidden_dims=[self.final_embedding_dim],
        hidden_activations=['relu'],
        layer_name_prefix='Context')
    self.label_encoder = encoders.EmbeddingEncoder(
        item_embedding_layer=self.item_embedding_layer,
        hidden_dims=[self.final_embedding_dim],
        hidden_activations=['relu'],
        layer_name_prefix='Label')
    self.test_sequence_batch = tf.constant([[2, 3], [4, 5]])
    self.test_item_batch = tf.constant([[0], [1]])

  def test_build_encoder_flatten(self):
    test_encoder = model_lib.build_encoder(
        encoder_type='flatten',
        item_embedding_layer=self.item_embedding_layer,
        hidden_dims=[self.final_embedding_dim],
        hidden_activations=['relu'],
        layer_name_prefix='Label')

    test_encoder_output = test_encoder.call(self.test_item_batch)
    self.assertEqual(test_encoder_output.dtype, tf.float32)
    self.assertEqual(test_encoder_output.shape, [2, self.final_embedding_dim])

  def test_build_encoder_flatten_with_no_dense_layer(self):
    test_encoder = model_lib.build_encoder(
        encoder_type='flatten',
        item_embedding_layer=self.item_embedding_layer,
        hidden_dims=None,
        hidden_activations=None,
        layer_name_prefix='Label')

    test_encoder_output = test_encoder.call(self.test_item_batch)
    self.assertEqual(test_encoder_output.dtype, tf.float32)
    self.assertEqual(test_encoder_output.shape, [2, self.item_embedding_dim])

  def test_build_encoder_bow(self):
    test_encoder = model_lib.build_encoder(
        encoder_type='bow',
        item_embedding_layer=self.item_embedding_layer,
        hidden_dims=[self.final_embedding_dim],
        hidden_activations=['relu'],
        layer_name_prefix='Context')

    test_encoder_output = test_encoder.call(self.test_sequence_batch)
    self.assertEqual(test_encoder_output.dtype, tf.float32)
    self.assertEqual(test_encoder_output.shape, [2, self.final_embedding_dim])

  def test_build_encoder_bow_with_no_dense_layer(self):
    test_encoder = model_lib.build_encoder(
        encoder_type='bow',
        item_embedding_layer=self.item_embedding_layer,
        hidden_dims=None,
        hidden_activations=None,
        layer_name_prefix='Label')

    test_encoder_output = test_encoder.call(self.test_item_batch)
    self.assertEqual(test_encoder_output.dtype, tf.float32)
    self.assertEqual(test_encoder_output.shape, [2, self.item_embedding_dim])

  def test_build_encoder_invalid_encoder_type(self):
    with self.assertRaisesRegex(ValueError, 'unexpected encoder type'):
      model_lib.build_encoder(
          encoder_type='test',
          item_embedding_layer=self.item_embedding_layer,
          hidden_dims=[self.final_embedding_dim],
          hidden_activations=['relu'],
          layer_name_prefix='Context')

  def test_get_loss_batch_softmax(self):
    loss = model_lib.get_loss('batch_softmax', expect_embeddings=True)

    y_pred = tf.constant(
        [[1, 2, 3],
         [4.0, 5.0, 6.0],
         [1, 1, 1],
         [1, 1, 1],
         [1, 2, 3],
         [4.0, 5.0, 6.0],
         [1, 1, 1],
         [1, 1, 1]]
    )
    y_true = tf.constant([1.0, 1.0, 1.0, 1.0])

    loss_value = loss(y_true, y_pred)
    expected_loss_value = 1.3616819

    tf.debugging.assert_near(expected_loss_value, loss_value)

  def test_get_loss_hinge(self):
    loss = model_lib.get_loss('hinge', expect_embeddings=True)

    y_pred = tf.constant(
        [[1, 0],
         [0.0, 1.0],
         [1, 1],
         [0, 1],
         [1, 0],
         [1, 0]]
    )
    y_true = tf.constant([1.0, 1.0, 1.0])

    loss_value = loss(y_true, y_pred)
    expected_loss_value = 0.552403

    tf.debugging.assert_near(expected_loss_value, loss_value)

  def test_get_loss_invalid_loss_function(self):
    with self.assertRaisesRegex(ValueError, 'unexpected loss function'):
      model_lib.get_loss('test', expect_embeddings=True)

  def test_get_metrics(self):
    metrics_list = model_lib.get_metrics(eval_top_k=[1, 5, 10])
    self.assertLen(metrics_list, 6)

  def test_build_id_based_keras_model_has_dense_layer_in_both_towers(self):
    test_model = model_lib.build_keras_model(
        item_vocab_size=self.item_vocab_size,
        item_embedding_dim=self.item_embedding_dim,
        context_hidden_dims=[self.final_embedding_dim],
        context_hidden_activations=['relu'],
        label_hidden_dims=[self.final_embedding_dim],
        label_hidden_activations=['relu'])

    logits = test_model(self.inputs)
    self.assertEqual([self.batch_size, self.batch_size],
                     logits.shape.as_list())

  def test_build_id_based_keras_model_no_dense_layer_in_both_tower(self):
    test_model = model_lib.build_keras_model(
        item_vocab_size=self.item_vocab_size,
        item_embedding_dim=self.item_embedding_dim,
        context_hidden_dims=None,
        context_hidden_activations=None,
        label_hidden_dims=None,
        label_hidden_activations=None,
    )

    logits = test_model(self.inputs)
    self.assertEqual([self.batch_size, self.batch_size], logits.shape.as_list())

  def test_build_id_based_keras_model_no_dense_layer_in_label_tower(self):
    test_model = model_lib.build_keras_model(
        item_vocab_size=self.item_vocab_size,
        item_embedding_dim=self.item_embedding_dim,
        context_hidden_dims=[self.item_embedding_dim],
        context_hidden_activations=['relu'],
        label_hidden_dims=None,
        label_hidden_activations=None,
    )

    logits = test_model(self.inputs)
    self.assertEqual([self.batch_size, self.batch_size], logits.shape.as_list())

  def test_build_id_based_dual_encoder_model(self):
    test_model = model_lib.build_id_based_dual_encoder_model(
        context_input_shape=self.context_input_shape,
        label_input_shape=self.label_input_shape,
        context_encoder=self.context_encoder,
        label_encoder=self.label_encoder,
        normalization_fn=self.normalization_fn)
    logits = test_model(self.inputs)
    self.assertEqual([self.batch_size, self.batch_size],
                     logits.shape.as_list())

  def test_build_id_based_dual_encoder_model_no_dense_layer_in_both_towers(self):  # pylint: disable=line-too-long
    context_encoder = encoders.EmbeddingEncoder(
        item_embedding_layer=self.item_embedding_layer,
        hidden_dims=None,
        hidden_activations=None,
        layer_name_prefix='Context')
    label_encoder = encoders.EmbeddingEncoder(
        item_embedding_layer=self.item_embedding_layer,
        hidden_dims=None,
        hidden_activations=None,
        layer_name_prefix='Label')
    test_model = model_lib.build_id_based_dual_encoder_model(
        context_input_shape=self.context_input_shape,
        label_input_shape=self.label_input_shape,
        context_encoder=context_encoder,
        label_encoder=label_encoder,
        normalization_fn=self.normalization_fn)
    logits = test_model(self.inputs)
    self.assertEqual([self.batch_size, self.batch_size], logits.shape.as_list())

  def test_build_id_based_dual_encoder_model_mismatch_output_embedding_dims(self):  # pylint: disable=line-too-long
    context_encoder = encoders.EmbeddingEncoder(
        item_embedding_layer=self.item_embedding_layer,
        hidden_dims=[6],
        hidden_activations=[None],
        layer_name_prefix='Context')
    label_encoder = encoders.EmbeddingEncoder(
        item_embedding_layer=self.item_embedding_layer,
        hidden_dims=[8],
        hidden_activations=[None],
        layer_name_prefix='Label')
    with self.assertRaisesRegex(ValueError, 'last dimension'):
      model_lib.build_id_based_dual_encoder_model(
          context_input_shape=self.context_input_shape,
          label_input_shape=self.label_input_shape,
          context_encoder=context_encoder,
          label_encoder=label_encoder,
          normalization_fn=self.normalization_fn)

  def test_build_id_based_dual_encoder_model_with_output_embeddings(self):
    test_model = model_lib.build_id_based_dual_encoder_model(
        context_input_shape=self.context_input_shape,
        label_input_shape=self.label_input_shape,
        context_encoder=self.context_encoder,
        label_encoder=self.label_encoder,
        normalization_fn=self.normalization_fn,
        output_embeddings=True)
    logits = test_model(self.inputs)
    self.assertEqual(
        [self.batch_size + self.batch_size, self.final_embedding_dim],
        logits.shape.as_list())

  def test_build_id_based_dual_encoder_model_use_global_similarity(self):
    test_model = model_lib.build_id_based_dual_encoder_model(
        context_input_shape=self.context_input_shape,
        label_input_shape=self.label_input_shape,
        context_encoder=self.context_encoder,
        label_encoder=self.label_encoder,
        normalization_fn=self.normalization_fn,
        output_embeddings=False,
        use_global_similarity=True,
        item_vocab_size=self.item_vocab_size)
    logits = test_model(self.inputs)
    self.assertEqual([self.batch_size, self.embedding_vocab_size],
                     logits.shape.as_list())

  def test_build_id_based_dual_encoder_model_use_global_similarity_with_output_embeddings(self):  # pylint: disable=line-too-long
    test_model = model_lib.build_id_based_dual_encoder_model(
        context_input_shape=self.context_input_shape,
        label_input_shape=self.label_input_shape,
        context_encoder=self.context_encoder,
        label_encoder=self.label_encoder,
        normalization_fn=self.normalization_fn,
        output_embeddings=True,
        use_global_similarity=True,
        item_vocab_size=self.item_vocab_size)
    logits = test_model(self.inputs)
    self.assertEqual(
        [self.batch_size + self.embedding_vocab_size, self.final_embedding_dim],
        logits.shape.as_list())


if __name__ == '__main__':
  absltest.main()
