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
"""Tests for keras_utils."""
import collections
from typing import Any
from absl.testing import parameterized

import tensorflow as tf
import tensorflow_federated as tff

from dp_visual_embeddings.models import embedding_model as model_lib
from dp_visual_embeddings.models import keras_utils

_DATA_DIM = 3
_LABEL_SIZE = 2
_DATA_ELEMENT_SPEC = (tf.TensorSpec([None, _DATA_DIM], dtype=tf.float32),
                      tf.TensorSpec([None], dtype=tf.int32))


def _model_fn(input_spec=_DATA_ELEMENT_SPEC):
  global_layer = tf.keras.layers.Dense(5)
  client_layer = tf.keras.layers.Dense(_LABEL_SIZE)
  inputs = tf.keras.Input(shape=(_DATA_DIM,))
  x = global_layer(inputs)
  outputs = client_layer(x)
  keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)

  global_variables = tff.learning.ModelWeights(
      trainable=global_layer.trainable_variables,
      non_trainable=global_layer.non_trainable_variables)
  client_variables = tff.learning.ModelWeights(
      trainable=client_layer.trainable_variables,
      non_trainable=client_layer.non_trainable_variables)
  tff_model = keras_utils.from_keras_model(
      keras_model,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
      input_spec=input_spec,
      global_variables=global_variables,
      client_variables=client_variables)
  return tff_model


def _get_synthetic_data_batch(batch_size: int = 2) -> dict[str, Any]:
  return collections.OrderedDict(
      x=tf.random.uniform(shape=[batch_size, _DATA_DIM]),
      y=tf.random.uniform(
          shape=[batch_size], maxval=_LABEL_SIZE, minval=0, dtype=tf.int32))


class KerasUtilsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('always_norm', True), ('infer_norm', False))
  def test_inference_normalize(self, always_normalize):
    norm_layer = keras_utils.EmbedNormLayer(always_normalize=always_normalize)
    batch_size, embed_size = 5, 32
    embeddings = tf.random.stateless_uniform(
        shape=[batch_size, embed_size], seed=[1, 2])
    normalize_embeddings = norm_layer(embeddings, training=False)
    norms = tf.norm(normalize_embeddings, ord='euclidean', axis=1)
    self.assertAllClose(norms, tf.ones([batch_size]))

  def test_training_no_normalize(self):
    norm_layer = keras_utils.EmbedNormLayer(always_normalize=False)
    batch_size, embed_size = 5, 32
    embeddings = tf.random.stateless_uniform(
        shape=[batch_size, embed_size], seed=[1, 2])
    normalize_embeddings = norm_layer(embeddings, training=True)
    self.assertAllClose(normalize_embeddings, embeddings)

  def test_training_normalize(self):
    norm_layer = keras_utils.EmbedNormLayer(always_normalize=True)
    batch_size, embed_size = 5, 32
    embeddings = tf.random.stateless_uniform(
        shape=[batch_size, embed_size], seed=[1, 2])
    normalize_embeddings = norm_layer(embeddings, training=True)
    norms = tf.norm(normalize_embeddings, ord='euclidean', axis=1)
    self.assertAllClose(norms, tf.ones([batch_size]))

  @parameterized.named_parameters(('train_no_norm', True),
                                  ('inference_norm', False))
  def test_zero_embedding(self, training):
    norm_layer = keras_utils.EmbedNormLayer(always_normalize=False)
    batch_size, embed_size = 5, 32
    embeddings = tf.zeros(shape=[batch_size, embed_size])
    normalize_embeddings = norm_layer(embeddings, training=training)
    norms = tf.norm(normalize_embeddings, ord='euclidean', axis=1)
    self.assertAllClose(norms, tf.zeros([batch_size]))
    self.assertAllClose(normalize_embeddings, embeddings)

  def test_mix_zero_nonzero_normalize(self):
    norm_layer = keras_utils.EmbedNormLayer(always_normalize=False)
    batch_size, embed_size = 5, 32
    nonzero_embeddings = tf.random.stateless_uniform(
        shape=[batch_size, embed_size], seed=[1, 2])
    zero_embeddings = tf.zeros(shape=[batch_size, embed_size])
    embeddings = tf.concat([nonzero_embeddings, zero_embeddings], axis=0)
    normalize_embeddings = norm_layer(embeddings, training=False)
    norms = tf.norm(normalize_embeddings, ord='euclidean', axis=1)
    self.assertAllClose(norms, [1] * batch_size + [0] * batch_size)

  def test_dense_norm_layer(self):
    batch_size, embed_size, output_size = 5, 32, 7
    layer = keras_utils.DenseNormLayer(
        output_size, activation=None, use_bias=False)
    norm_layer = keras_utils.EmbedNormLayer(always_normalize=True)
    embeddings = norm_layer(
        tf.random.stateless_uniform(
            shape=[batch_size, embed_size], seed=[1, 2]))
    outputs = layer(embeddings)
    self.assertAllEqual(tf.shape(outputs), [batch_size, output_size])
    self.assertAllGreaterEqual(outputs, -1)
    self.assertAllLessEqual(outputs, 1)
    embeddings = norm_layer(tf.zeros(shape=[batch_size, embed_size]))
    outputs = layer(embeddings)
    self.assertAllEqual(outputs, tf.zeros(shape=[batch_size, output_size]))
    embeddings = norm_layer(tf.transpose(layer.kernel[:, :batch_size]))
    outputs = layer(embeddings)
    self.assertAllClose(outputs[:, :batch_size],
                        tf.matmul(embeddings, embeddings, transpose_b=True))
    scale = 3.
    layer.global_scale.assign(scale)
    outputs = layer(embeddings)
    self.assertAllClose(
        outputs[:, :batch_size],
        scale * tf.matmul(embeddings, embeddings, transpose_b=True))


class ModelTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    tf.keras.backend.clear_session()
    super().setUp()

  # Test class for batches using namedtuple.
  _make_test_batch = collections.namedtuple('TestBatch', ['x', 'y'])

  @parameterized.named_parameters(
      ('container',
       collections.OrderedDict(
           x=tf.TensorSpec(shape=[None, _DATA_DIM], dtype=tf.float32),
           y=tf.TensorSpec(shape=[None, _DATA_DIM], dtype=tf.float32))),
      ('container_fn',
       _make_test_batch(
           x=tf.TensorSpec(shape=[1, _DATA_DIM], dtype=tf.float32),
           y=tf.TensorSpec(shape=[None, _DATA_DIM], dtype=tf.float32))))
  def test_input_spec_python_container(self, input_spec):
    tff_model = _model_fn(input_spec)
    self.assertIsInstance(tff_model, model_lib.Model)
    self.assertIsInstance(tff_model, tff.learning.Model)
    tf.nest.map_structure(lambda x: self.assertIsInstance(x, tf.TensorSpec),
                          tff_model.input_spec)

  @parameterized.named_parameters(
      ('more_than_two_elements', [
          tf.TensorSpec(shape=[None, _DATA_DIM], dtype=tf.float32),
          tf.TensorSpec(shape=[None, _DATA_DIM], dtype=tf.float32),
          tf.TensorSpec(shape=[None, _DATA_DIM], dtype=tf.float32)
      ]),
      ('dict_with_key_not_named_x',
       collections.OrderedDict(
           foo=tf.TensorSpec(shape=[None, _DATA_DIM], dtype=tf.float32),
           y=tf.TensorSpec(shape=[None, _DATA_DIM], dtype=tf.float32))),
      ('dict_with_key_not_named_y',
       collections.OrderedDict(
           x=tf.TensorSpec(shape=[None, _DATA_DIM], dtype=tf.float32),
           bar=tf.TensorSpec(shape=[None, _DATA_DIM], dtype=tf.float32))),
  )
  def test_input_spec_batch_types_value_errors(self, input_spec):
    with self.assertRaises(ValueError):
      _model_fn(input_spec)

  def test_build_model(self):
    batch_size = 4
    tff_model = _model_fn()
    self.assertAllClose(
        tff_model._keras_model.trainable_variables,
        tff_model.trainable_variables + tff_model.client_trainable_variables)
    self.assertAllClose(
        tff_model._keras_model.non_trainable_variables,
        tff_model.non_trainable_variables +
        tff_model.client_non_trainable_variables)
    batch_data = _get_synthetic_data_batch(batch_size)
    logits = tff_model.predict_on_batch(batch_data['x'])
    self.assertAllEqual(tf.shape(logits), [batch_size, _LABEL_SIZE])
    batch_output = tff_model.forward_pass(batch_data)
    self.assertIsInstance(batch_output, tff.learning.BatchOutput)
    self.assertAllEqual(batch_output.predictions, logits)
    self.assertEqual(batch_output.num_examples, batch_size)
    metrics = tff_model.report_local_unfinalized_metrics()
    self.assertEqual(metrics['num_examples'][0], batch_size)
    self.assertEqual(metrics['num_batches'][0], 1)
    self.assertLen(metrics['sparse_categorical_accuracy'], 2)
    self.assertLessEqual(
        metrics['sparse_categorical_accuracy'][0] /
        metrics['sparse_categorical_accuracy'][1], 1.)


if __name__ == '__main__':
  tf.test.main()
