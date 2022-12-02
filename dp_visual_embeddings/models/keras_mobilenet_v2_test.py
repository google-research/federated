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
import functools
from absl.testing import parameterized

import tensorflow as tf

from dp_visual_embeddings.models import keras_mobilenet_v2 as mobilenet_v2


TEST_INPUT_SHAPE = (224, 224, 3)
TEST_SMALL_MODEL_INPUT_SHAPE = (64, 64, 3)
TEST_NUM_IDENTITIES = 1000
TEST_EMBEDDING_DIM_SIZE = 128


# Helper functions to return only the model.
def _create_mobilenet_v2_for_backbone_training(*args, **kwargs):
  return mobilenet_v2.create_mobilenet_v2_for_backbone_training(
      *args, **kwargs).model


def _create_small_mobilenet_v2_for_backbone_training(*args, **kwargs):
  return mobilenet_v2.create_small_mobilenet_v2_for_backbone_training(
      *args, **kwargs).model


class MobileNetModelTest(tf.test.TestCase, parameterized.TestCase):

  MOBILENET_MODEL_FN_FOR_EMBED_PRED = functools.partial(
      mobilenet_v2.create_mobilenet_v2_for_embedding_prediction,
      input_shape=TEST_INPUT_SHAPE,
      embedding_dim_size=TEST_EMBEDDING_DIM_SIZE)
  MOBILENET_MODEL_FN_FOR_BACKBONE = functools.partial(
      _create_mobilenet_v2_for_backbone_training,
      input_shape=TEST_INPUT_SHAPE,
      num_identities=TEST_NUM_IDENTITIES,
      embedding_dim_size=TEST_EMBEDDING_DIM_SIZE)
  SMALL_MOBILENET_MODEL_FN_FOR_EMBED_PRED = functools.partial(
      mobilenet_v2.create_small_mobilenet_v2_for_embedding_prediction,
      input_shape=TEST_SMALL_MODEL_INPUT_SHAPE,
      embedding_dim_size=TEST_EMBEDDING_DIM_SIZE)
  SMALL_MOBILENET_MODEL_FN_FOR_BACKBONE = functools.partial(
      _create_small_mobilenet_v2_for_backbone_training,
      input_shape=TEST_SMALL_MODEL_INPUT_SHAPE,
      num_identities=TEST_NUM_IDENTITIES,
      embedding_dim_size=TEST_EMBEDDING_DIM_SIZE)
  MOBILENET_MODEL_VARIABLES_FN_FOR_BACKBONE = functools.partial(
      mobilenet_v2.create_mobilenet_v2_for_backbone_training,
      input_shape=TEST_INPUT_SHAPE,
      num_identities=TEST_NUM_IDENTITIES,
      embedding_dim_size=TEST_EMBEDDING_DIM_SIZE)
  SMALL_MOBILENET_MODEL_VARIABLES_FN_FOR_BACKBONE = functools.partial(
      mobilenet_v2.create_small_mobilenet_v2_for_backbone_training,
      input_shape=TEST_SMALL_MODEL_INPUT_SHAPE,
      num_identities=TEST_NUM_IDENTITIES,
      embedding_dim_size=TEST_EMBEDDING_DIM_SIZE)

  @parameterized.named_parameters(
      ('w_mobilenet_v2_for_embed_pred', MOBILENET_MODEL_FN_FOR_EMBED_PRED),
      ('w_mobilenet_v2_for_backbone_train', MOBILENET_MODEL_FN_FOR_BACKBONE),
      ('w_small_mobilenet_v2_for_embed_pred',
       SMALL_MOBILENET_MODEL_FN_FOR_EMBED_PRED),
      ('w_small_mobilenet_v2_for_backbone_train',
       SMALL_MOBILENET_MODEL_FN_FOR_BACKBONE))
  def test_alpha_changes_number_parameters(self, create_model_fn):
    model1 = create_model_fn()
    model2 = create_model_fn(alpha=0.5)
    model3 = create_model_fn(alpha=2.0)
    self.assertIsInstance(model1, tf.keras.Model)
    self.assertIsInstance(model2, tf.keras.Model)
    self.assertIsInstance(model3, tf.keras.Model)
    self.assertLess(model2.count_params(), model1.count_params())
    self.assertLess(model1.count_params(), model3.count_params())

  @parameterized.named_parameters(
      ('w_mobilenet_v2_for_embed_pred', MOBILENET_MODEL_FN_FOR_EMBED_PRED),
      ('w_mobilenet_v2_for_backbone_train', MOBILENET_MODEL_FN_FOR_BACKBONE),
      ('w_small_mobilenet_v2_for_embed_pred',
       SMALL_MOBILENET_MODEL_FN_FOR_EMBED_PRED),
      ('w_small_mobilenet_v2_for_backbone_train',
       SMALL_MOBILENET_MODEL_FN_FOR_BACKBONE))
  def test_num_groups(self, create_model_fn):
    model1 = create_model_fn()
    model2 = create_model_fn(num_groups=4)
    self.assertIsInstance(model1, tf.keras.Model)
    self.assertIsInstance(model2, tf.keras.Model)
    self.assertEqual(model1.count_params(), model2.count_params())

  @parameterized.named_parameters(
      ('w_mobilenet_v2_for_embed_pred', MOBILENET_MODEL_FN_FOR_EMBED_PRED),
      ('w_mobilenet_v2_for_backbone_train', MOBILENET_MODEL_FN_FOR_BACKBONE),
      ('w_small_mobilenet_v2_for_embed_pred',
       SMALL_MOBILENET_MODEL_FN_FOR_EMBED_PRED),
      ('w_small_mobilenet_v2_for_backbone_train',
       SMALL_MOBILENET_MODEL_FN_FOR_BACKBONE))
  def test_pooling_method(self, create_model_fn):
    model1 = create_model_fn(pooling='avg')
    model2 = create_model_fn(pooling='max')
    self.assertIsInstance(model1, tf.keras.Model)
    self.assertIsInstance(model2, tf.keras.Model)
    self.assertEqual(model1.count_params(), model2.count_params())

  @parameterized.named_parameters(
      ('w_mobilenet_v2',
       functools.partial(
           mobilenet_v2.create_mobilenet_v2_for_embedding_prediction,
           input_shape=TEST_INPUT_SHAPE,
           embedding_dim_size=TEST_EMBEDDING_DIM_SIZE)),
      ('w_small_mobilenet_v2',
       functools.partial(
           mobilenet_v2.create_small_mobilenet_v2_for_embedding_prediction,
           input_shape=TEST_SMALL_MODEL_INPUT_SHAPE,
           embedding_dim_size=TEST_EMBEDDING_DIM_SIZE)))
  def test_dropout(self, create_model_fn):
    model1 = create_model_fn(dropout_prob=0.5)
    model2 = create_model_fn(dropout_prob=0.2)
    model3 = create_model_fn(dropout_prob=None)
    self.assertEqual(len(model1.layers), len(model2.layers))
    self.assertGreater(len(model1.layers), len(model3.layers))

  @parameterized.named_parameters(
      ('w_mobilenet_v2_for_embed_pred', MOBILENET_MODEL_FN_FOR_EMBED_PRED,
       TEST_INPUT_SHAPE),
      ('w_small_mobilenet_v2_for_embed_pred',
       SMALL_MOBILENET_MODEL_FN_FOR_EMBED_PRED, TEST_SMALL_MODEL_INPUT_SHAPE),
  )
  def test_normalize_embedding(self, create_model_fn, input_shape):
    model = create_model_fn(pooling='avg')
    batch_size = 2
    synthetic_batch = tf.random.uniform(
        tf.TensorShape((batch_size,) + input_shape), dtype=tf.float32)
    embeddings = model(synthetic_batch, training=False)
    norms = tf.norm(embeddings, ord='euclidean', axis=1)
    self.assertAllClose(norms, tf.ones([batch_size]))

  @parameterized.named_parameters(
      ('w_mobilenet_v2_for_embed_pred', MOBILENET_MODEL_FN_FOR_EMBED_PRED,
       TEST_INPUT_SHAPE),
      ('w_small_mobilenet_v2_for_embed_pred',
       SMALL_MOBILENET_MODEL_FN_FOR_EMBED_PRED, TEST_SMALL_MODEL_INPUT_SHAPE),
  )
  def test_unnormalize_embedding(self, create_model_fn, input_shape):
    model = create_model_fn(pooling='avg')
    batch_size = 5
    synthetic_batch = tf.random.uniform(
        tf.TensorShape((batch_size,) + input_shape), dtype=tf.float32)
    embeddings = model(synthetic_batch, training=True)
    norms = tf.norm(embeddings, ord='euclidean', axis=1)
    self.assertNotAllClose(norms, tf.ones([batch_size]))

  @parameterized.named_parameters(
      ('w_mobilenet_v2', MOBILENET_MODEL_FN_FOR_BACKBONE, TEST_INPUT_SHAPE),
      ('w_small_mobilenet', SMALL_MOBILENET_MODEL_FN_FOR_BACKBONE,
       TEST_SMALL_MODEL_INPUT_SHAPE),
  )
  def test_training_output(self, create_model_fn, input_shape):
    model = create_model_fn(pooling='avg')
    batch_size = 5
    synthetic_batch = tf.random.uniform(
        tf.TensorShape((batch_size,) + input_shape), dtype=tf.float32)
    preds = model(synthetic_batch, training=True)
    self.assertAllEqual([batch_size, TEST_NUM_IDENTITIES], preds[0].shape)
    self.assertAllEqual([batch_size, TEST_EMBEDDING_DIM_SIZE], preds[1].shape)

  @parameterized.named_parameters(
      ('w_mobilenet_v2', MOBILENET_MODEL_VARIABLES_FN_FOR_BACKBONE,
       MOBILENET_MODEL_FN_FOR_EMBED_PRED),
      ('w_small_mobilenet', SMALL_MOBILENET_MODEL_VARIABLES_FN_FOR_BACKBONE,
       SMALL_MOBILENET_MODEL_FN_FOR_EMBED_PRED),
  )
  def test_global_and_client_variables(self, model_variables_fn,
                                       inference_model_fn):
    train_model = model_variables_fn()
    test_model = inference_model_fn()
    tf.nest.map_structure(
        lambda x, y: self.assertShapeEqual(  # pylint: disable=g-long-lambda
            tf.convert_to_tensor(x), tf.convert_to_tensor(y)),
        train_model.global_variables.trainable,
        test_model.trainable_variables)
    self.assertAllClose(
        train_model.model.trainable_variables,
        train_model.global_variables.trainable +
        train_model.client_variables.trainable)
    self.assertEmpty(train_model.model.non_trainable_variables)
    self.assertEmpty(train_model.global_variables.non_trainable)
    self.assertEmpty(train_model.client_variables.non_trainable)
    self.assertEmpty(test_model.non_trainable_variables)

if __name__ == '__main__':
  tf.test.main()
