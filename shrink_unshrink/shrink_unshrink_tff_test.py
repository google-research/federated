# Copyright 2020, The TensorFlow Federated Authors.
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
"""Tests for shrink_unshrink_tff."""

# from absl import logging
import tensorflow as tf
import tensorflow_federated as tff

from shrink_unshrink import models
from shrink_unshrink import shrink_unshrink_tff
from shrink_unshrink import simple_fedavg_tf
from shrink_unshrink import simple_fedavg_tff


class ShrinkUnshrinkTffTest(tf.test.TestCase):

  def test_make_layerwise_projection_shrink_and_unshrink_typing(self):
    train_client_spec = tff.simulation.baselines.ClientSpec(
        num_epochs=3, batch_size=32, max_elements=1000)
    my_task = tff.simulation.baselines.stackoverflow.create_word_prediction_task(
        train_client_spec, use_synthetic_data=True)
    server_model_fn, client_model_fn = models.make_big_and_small_stackoverflow_model_fn(
        my_task,
        big_embedding_size=96 * 2,
        big_lstm_size=670 * 2,
        small_embedding_size=96,
        small_lstm_size=670)

    shrink_unshrink_info = simple_fedavg_tf.LayerwiseProjectionShrinkUnshrinkInfoV2(
        left_mask=[-1, 0, 2, -1, 2, -1, 0, -1],
        right_mask=[0, 1, 1, 1, 0, 0, -1, -1],
        build_projection_matrix=simple_fedavg_tf.build_normal_projection_matrix)

    _, shrink, unshrink = simple_fedavg_tff.build_federated_shrink_unshrink_process(
        server_model_fn=server_model_fn,
        client_model_fn=client_model_fn,
        make_shrink=shrink_unshrink_tff.make_layerwise_projection_shrink,
        make_unshrink=shrink_unshrink_tff.make_layerwise_projection_unshrink,
        shrink_unshrink_info=shrink_unshrink_info,
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1),
        debugging=True)
    # logging.info('shrink unshrink type_signature')
    # logging.info(shrink.type_signature)
    # logging.info(unshrink.type_signature)

    self.assertEqual(
        str(shrink.type_signature),
        '(<server_state=<model_weights=<trainable=<float32[10004,192],float32[192,5360],float32[1340,5360],float32[5360],float32[1340,192],float32[192],float32[192,10004],float32[10004]>,non_trainable=<>>,optimizer_state=<int64>,round_num=int32>@SERVER,federated_dataset={<int64[?,20],int64[?,20]>*}@CLIENTS> -> <model_weights=<trainable=<tf.float32[None],tf.float32[None],tf.float32[None],tf.float32[None],tf.float32[None],tf.float32[None],tf.float32[None],tf.float32[None]>,non_trainable=<>>,round_num=int32,shrink_unshrink_dynamic_info=<>>@CLIENTS)'
    )
    self.assertEqual(
        str(unshrink.type_signature),
        '(<server_state=<model_weights=<trainable=<float32[10004,192],float32[192,5360],float32[1340,5360],float32[5360],float32[1340,192],float32[192],float32[192,10004],float32[10004]>,non_trainable=<>>,optimizer_state=<int64>,round_num=int32>@SERVER,client_outputs={<weights_delta=<tf.float32[None],tf.float32[None],tf.float32[None],tf.float32[None],tf.float32[None],tf.float32[None],tf.float32[None],tf.float32[None]>,client_weight=float32,model_output=<loss=<float32,float32>>,round_num=int32,shrink_unshrink_dynamic_info=<>>}@CLIENTS> -> <model_weights=<trainable=<float32[10004,192],float32[192,5360],float32[1340,5360],float32[5360],float32[1340,192],float32[192],float32[192,10004],float32[10004]>,non_trainable=<>>,optimizer_state=<int64>,round_num=int32>@SERVER)'
    )

  def test_make_client_specific_layerwise_projection_shrink_and_unshrink_typing(
      self):
    train_client_spec = tff.simulation.baselines.ClientSpec(
        num_epochs=3, batch_size=32, max_elements=1000)
    my_task = tff.simulation.baselines.stackoverflow.create_word_prediction_task(
        train_client_spec, use_synthetic_data=True)
    server_model_fn, client_model_fn = models.make_big_and_small_stackoverflow_model_fn(
        my_task,
        big_embedding_size=96 * 2,
        big_lstm_size=670 * 2,
        small_embedding_size=96,
        small_lstm_size=670)

    shrink_unshrink_info = simple_fedavg_tf.LayerwiseProjectionShrinkUnshrinkInfoV2(
        left_mask=[-1, 0, 2, -1, 2, -1, 0, -1],
        right_mask=[0, 1, 1, 1, 0, 0, -1, -1],
        build_projection_matrix=simple_fedavg_tf.build_normal_projection_matrix)
    _, shrink, unshrink = simple_fedavg_tff.build_federated_shrink_unshrink_process(
        server_model_fn=server_model_fn,
        client_model_fn=client_model_fn,
        make_shrink=shrink_unshrink_tff
        .make_client_specific_layerwise_projection_shrink,
        make_unshrink=shrink_unshrink_tff
        .make_client_specific_layerwise_projection_unshrink,
        shrink_unshrink_info=shrink_unshrink_info,
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1),
        debugging=True)

    self.assertEqual(
        str(shrink.type_signature),
        '(<server_state=<model_weights=<trainable=<float32[10004,192],float32[192,5360],float32[1340,5360],float32[5360],float32[1340,192],float32[192],float32[192,10004],float32[10004]>,non_trainable=<>>,optimizer_state=<int64>,round_num=int32>@SERVER,federated_dataset={<int64[?,20],int64[?,20]>*}@CLIENTS> -> {<model_weights=<trainable=<tf.float32[None],tf.float32[None],tf.float32[None],tf.float32[None],tf.float32[None],tf.float32[None],tf.float32[None],tf.float32[None]>,non_trainable=<>>,round_num=int32,shrink_unshrink_dynamic_info=uint64>}@CLIENTS)'
    )
    self.assertEqual(
        str(unshrink.type_signature),
        '(<server_state=<model_weights=<trainable=<float32[10004,192],float32[192,5360],float32[1340,5360],float32[5360],float32[1340,192],float32[192],float32[192,10004],float32[10004]>,non_trainable=<>>,optimizer_state=<int64>,round_num=int32>@SERVER,client_outputs={<weights_delta=<tf.float32[None],tf.float32[None],tf.float32[None],tf.float32[None],tf.float32[None],tf.float32[None],tf.float32[None],tf.float32[None]>,client_weight=float32,model_output=<loss=<float32,float32>>,round_num=int32,shrink_unshrink_dynamic_info=uint64>}@CLIENTS> -> <model_weights=<trainable=<float32[10004,192],float32[192,5360],float32[1340,5360],float32[5360],float32[1340,192],float32[192],float32[192,10004],float32[10004]>,non_trainable=<>>,optimizer_state=<int64>,round_num=int32>@SERVER)'
    )


if __name__ == '__main__':
  tf.test.main()
