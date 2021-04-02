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

import numpy as np
import tensorflow as tf

from fedopt_guide.stackoverflow_transformer import transformer_models


class KerasTransformerTest(tf.test.TestCase):

  def test_constructs(self):
    model = transformer_models.create_transformer_lm(
        vocab_size=10,
        num_oov_buckets=1,
        dim_embed=24,
        dim_model=32,
        dim_hidden=64,
        num_heads=4,
        num_layers=1,
        max_position_encoding=100,
        dropout=0.1,
        name='transformer-lm')
    self.assertIsInstance(model, tf.keras.Model)
    self.assertEqual('transformer-lm', model.name)

  def test_shared_embedding_returns_dense_gradient_in_graph_mode(self):
    batch_size = 2
    sequence_length = 20
    batch_x = np.ones((batch_size, sequence_length), dtype=np.int32)
    batch_y = np.ones((batch_size, sequence_length), dtype=np.int32)
    graph = tf.Graph()
    with graph.as_default():
      model = transformer_models.create_transformer_lm(
          vocab_size=10,
          num_oov_buckets=1,
          dim_embed=24,
          dim_model=32,
          dim_hidden=64,
          num_heads=4,
          num_layers=1,
          max_position_encoding=100,
          dropout=0.1,
          name='transformer-lm')
      loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
      with tf.GradientTape() as tape:
        predictions = model(batch_x, training=True)
        loss = loss_fn(y_true=batch_y, y_pred=predictions)
      embedding_gradient = tape.gradient(loss, model.trainable_variables[0])
      init_op = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session(graph=graph) as sess:
      sess.run(init_op)
      embedding_grad = sess.run(embedding_gradient)

    self.assertTrue(np.all(np.linalg.norm(embedding_grad, axis=1) > 0.0))


if __name__ == '__main__':
  tf.test.main()
