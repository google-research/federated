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

import tensorflow as tf
import tensorflow_federated as tff

from shrink_unshrink import models
from shrink_unshrink import shrink_unshrink_tff
from shrink_unshrink import simple_fedavg_tf
from shrink_unshrink import simple_fedavg_tff


class ShrinkUnshrinkTffTest(tf.test.TestCase):

  def test_make_learnedv2_and_sparse_layerwise_projection_shrink_and_unshrink(
      self):
    if tf.config.list_logical_devices('GPU'):
      self.skipTest('skip GPU test')
    train_client_spec = tff.simulation.baselines.ClientSpec(
        num_epochs=3, batch_size=32, max_elements=1000)
    my_task = tff.simulation.baselines.emnist.create_character_recognition_task(
        train_client_spec, use_synthetic_data=True, model_id='cnn_dropout')
    server_model_fn, client_model_fn = models.make_big_and_small_emnist_cnn_model_fn(
        my_task,
        big_conv1_filters=32,
        big_conv2_filters=64,
        big_dense_size=512,
        small_conv1_filters=24,
        small_conv2_filters=48,
        small_dense_size=384)

    server_model = server_model_fn()
    client_model = client_model_fn()
    server_model_weights = simple_fedavg_tf.get_model_weights(server_model)
    client_model_weights = simple_fedavg_tf.get_model_weights(client_model)

    left_mask = [-1, -1, 0, -1, 1000, -1, 3, -1]
    right_mask = [0, 0, 1, 1, 3, 3, -1, -1]
    shrink_unshrink_info = simple_fedavg_tf.LayerwiseProjectionShrinkUnshrinkInfoV2(
        left_mask=left_mask,
        right_mask=right_mask,
        build_projection_matrix=simple_fedavg_tf.build_normal_projection_matrix,
        new_projection_dict_decimate=1)

    _, shrink, unshrink, server_init_tf = simple_fedavg_tff.build_federated_shrink_unshrink_process(
        server_model_fn=server_model_fn,
        client_model_fn=client_model_fn,
        make_shrink=shrink_unshrink_tff
        .make_learnedv2_layerwise_projection_shrink,
        make_unshrink=shrink_unshrink_tff
        .make_learned_sparse_layerwise_projection_unshrink,
        shrink_unshrink_info=shrink_unshrink_info,
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1),
        oja_hyperparameter=1.0,
        debugging=True)

    server_state = server_init_tf()

    weights_delta = [
        tf.random.stateless_normal(
            shape=tf.shape(weight), seed=(11, 12), stddev=1)
        for weight in simple_fedavg_tf.get_model_weights(client_model).trainable
    ]
    # pylint:disable=g-complex-comprehension
    client_ouputs = [
        simple_fedavg_tf.ClientOutput(
            weights_delta,
            client_weight=1.0,
            model_output=client_model.report_local_outputs(),
            round_num=0,
            shrink_unshrink_dynamic_info=simple_fedavg_tf
            .create_left_maskval_to_projmat_dict(
                seed=10,
                whimsy_server_weights=server_model_weights.trainable,
                whimsy_client_weights=client_model_weights.trainable,
                left_mask=left_mask,
                right_mask=right_mask,
                build_projection_matrix=simple_fedavg_tf
                .build_normal_projection_matrix)) for i in range(6)
    ]
    # pylint:enable=g-complex-comprehension
    federated_dataset = [
        my_task.datasets.train_preprocess_fn(
            my_task.datasets.train_data.create_tf_dataset_for_client(
                my_task.datasets.train_data.client_ids[0]))
    ]
    server_message_1 = shrink(server_state, federated_dataset)
    self.assertEqual(server_message_1[0].round_num, server_state.round_num)

    server_state_1 = unshrink(server_state, client_ouputs)
    self.assertEqual(server_state_1.round_num, 1)
    new_dict, old_dict = server_state_1.shrink_unshrink_server_info.oja_left_maskval_to_projmat_dict, server_state.shrink_unshrink_server_info.oja_left_maskval_to_projmat_dict
    self.assertAllEqual(new_dict.keys(), old_dict.keys())
    for k in new_dict.keys():
      self.assertAllEqual(tf.shape(new_dict[k]), tf.shape(old_dict[k]))
      if not (k == '-1' or (int(k) > 0 and int(k) % 1000 == 0)):
        self.assertNotAllClose(new_dict[k], old_dict[k], msg=f'the key is: {k}')
    self.assertEqual(new_dict['-1'], 1)
    self.assertEqual(old_dict['-1'], 1)
    for i in range(len(server_state.model_weights.trainable)):
      self.assertAllEqual(server_state.model_weights.trainable[i],
                          server_state_1.model_weights.trainable[i])

    server_message_2 = shrink(server_state_1, federated_dataset)
    self.assertEqual(server_message_2[0].round_num, server_state_1.round_num)
    self.assertAllEqual(server_message_1[0].shrink_unshrink_dynamic_info.keys(),
                        server_message_2[0].shrink_unshrink_dynamic_info.keys())
    for k in server_message_1[0].shrink_unshrink_dynamic_info.keys():
      if not (k == '-1' or (int(k) > 0 and int(k) % 1000 == 0)):
        self.assertNotAllClose(
            server_message_1[0].shrink_unshrink_dynamic_info[k],
            server_message_2[0].shrink_unshrink_dynamic_info[k])
      elif k == '-1':
        self.assertEqual(server_message_1[0].shrink_unshrink_dynamic_info[k], 1)
        self.assertEqual(server_message_2[0].shrink_unshrink_dynamic_info[k], 1)

    server_state_2 = unshrink(server_state_1, client_ouputs)
    self.assertEqual(server_state_2.round_num, 2)
    new_dict, old_dict = server_state_2.shrink_unshrink_server_info.oja_left_maskval_to_projmat_dict, server_state_1.shrink_unshrink_server_info.oja_left_maskval_to_projmat_dict
    self.assertAllEqual(new_dict.keys(), old_dict.keys())
    for k in new_dict.keys():
      self.assertAllEqual(new_dict[k], old_dict[k])
    for i in range(len(server_state.model_weights.trainable)):
      self.assertNotAllClose(
          server_state_1.model_weights.trainable[i],
          server_state_2.model_weights.trainable[i],
          msg=f'idx is {i}')

  def test_make_learned_layerwise_projection_shrink_and_unshrink(self):
    train_client_spec = tff.simulation.baselines.ClientSpec(
        num_epochs=3, batch_size=32, max_elements=1000)
    my_task = tff.simulation.baselines.emnist.create_character_recognition_task(
        train_client_spec, use_synthetic_data=True, model_id='cnn_dropout')
    server_model_fn, client_model_fn = models.make_big_and_small_emnist_cnn_model_fn(
        my_task,
        big_conv1_filters=32,
        big_conv2_filters=64,
        big_dense_size=512,
        small_conv1_filters=24,
        small_conv2_filters=48,
        small_dense_size=384)

    server_model = server_model_fn()
    client_model = client_model_fn()
    server_model_weights = simple_fedavg_tf.get_model_weights(server_model)
    client_model_weights = simple_fedavg_tf.get_model_weights(client_model)

    left_mask = [-1, -1, 0, -1, 1000, -1, 3, -1]
    right_mask = [0, 0, 1, 1, 3, 3, -1, -1]
    shrink_unshrink_info = simple_fedavg_tf.LayerwiseProjectionShrinkUnshrinkInfoV2(
        left_mask=left_mask,
        right_mask=right_mask,
        build_projection_matrix=simple_fedavg_tf.build_normal_projection_matrix,
        new_projection_dict_decimate=1)

    _, shrink, unshrink, server_init_tf = simple_fedavg_tff.build_federated_shrink_unshrink_process(
        server_model_fn=server_model_fn,
        client_model_fn=client_model_fn,
        make_shrink=shrink_unshrink_tff
        .make_learned_layerwise_projection_shrink,
        make_unshrink=shrink_unshrink_tff
        .make_learned_layerwise_projection_unshrink,
        shrink_unshrink_info=shrink_unshrink_info,
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1),
        oja_hyperparameter=1.0,
        debugging=True)

    server_state = server_init_tf()

    weights_delta = [
        tf.random.stateless_normal(
            shape=tf.shape(weight), seed=(11, 12), stddev=1)
        for weight in simple_fedavg_tf.get_model_weights(client_model).trainable
    ]
    # pylint:disable=g-complex-comprehension
    client_ouputs = [
        simple_fedavg_tf.ClientOutput(
            weights_delta,
            client_weight=1.0,
            model_output=client_model.report_local_outputs(),
            round_num=0,
            shrink_unshrink_dynamic_info=simple_fedavg_tf
            .create_left_maskval_to_projmat_dict(
                seed=10,
                whimsy_server_weights=server_model_weights.trainable,
                whimsy_client_weights=client_model_weights.trainable,
                left_mask=left_mask,
                right_mask=right_mask,
                build_projection_matrix=simple_fedavg_tf
                .build_normal_projection_matrix)) for i in range(6)
    ]
    # pylint:enable=g-complex-comprehension
    federated_dataset = [
        my_task.datasets.train_preprocess_fn(
            my_task.datasets.train_data.create_tf_dataset_for_client(
                my_task.datasets.train_data.client_ids[0]))
    ]
    server_message_1 = shrink(server_state, federated_dataset)
    self.assertEqual(server_message_1.round_num, server_state.round_num)

    server_state_1 = unshrink(server_state, client_ouputs)
    self.assertEqual(server_state_1.round_num, 1)
    new_dict, old_dict = server_state_1.shrink_unshrink_server_info.oja_left_maskval_to_projmat_dict, server_state.shrink_unshrink_server_info.oja_left_maskval_to_projmat_dict
    self.assertAllEqual(new_dict.keys(), old_dict.keys())
    for k in new_dict.keys():
      self.assertAllEqual(tf.shape(new_dict[k]), tf.shape(old_dict[k]))
      if not (k == '-1' or (int(k) > 0 and int(k) % 1000 == 0)):
        self.assertNotAllClose(new_dict[k], old_dict[k], msg=f'the key is: {k}')
    self.assertEqual(new_dict['-1'], 1)
    self.assertEqual(old_dict['-1'], 1)
    for i in range(len(server_state.model_weights.trainable)):
      self.assertAllEqual(server_state.model_weights.trainable[i],
                          server_state_1.model_weights.trainable[i])

    server_message_2 = shrink(server_state_1, federated_dataset)
    self.assertEqual(server_message_2.round_num, server_state_1.round_num)
    self.assertAllEqual(server_message_1.shrink_unshrink_dynamic_info.keys(),
                        server_message_2.shrink_unshrink_dynamic_info.keys())
    for k in server_message_1.shrink_unshrink_dynamic_info.keys():
      if not (k == '-1' or (int(k) > 0 and int(k) % 1000 == 0)):
        self.assertNotAllClose(server_message_1.shrink_unshrink_dynamic_info[k],
                               server_message_2.shrink_unshrink_dynamic_info[k])
      elif k == '-1':
        self.assertEqual(server_message_1.shrink_unshrink_dynamic_info[k], 1)
        self.assertEqual(server_message_2.shrink_unshrink_dynamic_info[k], 1)

    server_state_2 = unshrink(server_state_1, client_ouputs)
    self.assertEqual(server_state_2.round_num, 2)
    new_dict, old_dict = server_state_2.shrink_unshrink_server_info.oja_left_maskval_to_projmat_dict, server_state_1.shrink_unshrink_server_info.oja_left_maskval_to_projmat_dict
    self.assertAllEqual(new_dict.keys(), old_dict.keys())
    for k in new_dict.keys():
      self.assertAllEqual(new_dict[k], old_dict[k])
    for i in range(len(server_state.model_weights.trainable)):
      self.assertNotAllClose(
          server_state_1.model_weights.trainable[i],
          server_state_2.model_weights.trainable[i],
          msg=f'idx is {i}')

  def test_make_layerwise_projection_shrink_and_unshrink_typing(self):
    train_client_spec = tff.simulation.baselines.ClientSpec(
        num_epochs=3, batch_size=32, max_elements=1000)
    my_task = tff.simulation.baselines.stackoverflow.create_word_prediction_task(
        train_client_spec, use_synthetic_data=False)
    server_model_fn, client_model_fn = models.make_big_and_small_stackoverflow_model_fn(
        my_task,
        big_embedding_size=96 * 2,
        big_lstm_size=670 * 2,
        small_embedding_size=96,
        small_lstm_size=670)

    shrink_unshrink_info = simple_fedavg_tf.LayerwiseProjectionShrinkUnshrinkInfoV2(
        left_mask=[-1, 0, 2, -1, 2, -1, 0, -1],
        right_mask=[0, 1, 1, 1, 0, 0, -1, -1],
        build_projection_matrix=simple_fedavg_tf.build_normal_projection_matrix,
        new_projection_dict_decimate=1)

    _, shrink, unshrink, server_init_tf = simple_fedavg_tff.build_federated_shrink_unshrink_process(
        server_model_fn=server_model_fn,
        client_model_fn=client_model_fn,
        make_shrink=shrink_unshrink_tff.make_layerwise_projection_shrink,
        make_unshrink=shrink_unshrink_tff.make_layerwise_projection_unshrink,
        shrink_unshrink_info=shrink_unshrink_info,
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1),
        debugging=True)

    server_state = server_init_tf()
    client_model = client_model_fn()

    weights_delta = simple_fedavg_tf.get_model_weights(client_model).trainable
    # pylint:disable=g-complex-comprehension
    client_ouputs = [
        simple_fedavg_tf.ClientOutput(
            weights_delta,
            client_weight=1.0,
            model_output=client_model.report_local_outputs(),
            round_num=0) for i in range(6)
    ]
    # pylint:enable=g-complex-comprehension
    unshrink(server_state, client_ouputs)

    self.assertEqual(
        str(shrink.type_signature),
        '(<server_state=<model_weights=<trainable=<float32[10004,192],float32[192,5360],float32[1340,5360],float32[5360],float32[1340,192],float32[192],float32[192,10004],float32[10004]>,non_trainable=<>>,optimizer_state=<int64>,round_num=int32,shrink_unshrink_server_info=<lmbda=float32,oja_left_maskval_to_projmat_dict=<-1=float32[1],0=float32[?,?],2=float32[?,?],1=float32[?,?]>,memory_dict=<0=float32[?,?],2=float32[?,?],1=float32[?,?]>>>@SERVER,federated_dataset={<int64[?,20],int64[?,20]>*}@CLIENTS> -> <model_weights=<trainable=<tf.float32[None],tf.float32[None],tf.float32[None],tf.float32[None],tf.float32[None],tf.float32[None],tf.float32[None],tf.float32[None]>,non_trainable=<>>,round_num=int32,shrink_unshrink_dynamic_info=<>>@CLIENTS)'
    )
    self.assertEqual(
        str(unshrink.type_signature),
        '(<server_state=<model_weights=<trainable=<float32[10004,192],float32[192,5360],float32[1340,5360],float32[5360],float32[1340,192],float32[192],float32[192,10004],float32[10004]>,non_trainable=<>>,optimizer_state=<int64>,round_num=int32,shrink_unshrink_server_info=<lmbda=float32,oja_left_maskval_to_projmat_dict=<-1=float32[1],0=float32[?,?],2=float32[?,?],1=float32[?,?]>,memory_dict=<0=float32[?,?],2=float32[?,?],1=float32[?,?]>>>@SERVER,client_outputs={<weights_delta=<tf.float32[None],tf.float32[None],tf.float32[None],tf.float32[None],tf.float32[None],tf.float32[None],tf.float32[None],tf.float32[None]>,client_weight=float32,model_output=<loss=<float32,float32>>,round_num=int32,shrink_unshrink_dynamic_info=<>>}@CLIENTS> -> <model_weights=<trainable=<float32[10004,192],float32[192,5360],float32[1340,5360],float32[5360],float32[1340,192],float32[192],float32[192,10004],float32[10004]>,non_trainable=<>>,optimizer_state=<int64>,round_num=int32,shrink_unshrink_server_info=<lmbda=float32,oja_left_maskval_to_projmat_dict=<-1=float32[1],0=float32[?,?],2=float32[?,?],1=float32[?,?]>,memory_dict=<0=float32[?,?],2=float32[?,?],1=float32[?,?]>>>@SERVER)'
    )

  def test_make_client_specific_layerwise_projection_shrink_and_unshrink_typing(
      self):
    train_client_spec = tff.simulation.baselines.ClientSpec(
        num_epochs=3, batch_size=32, max_elements=1000)
    my_task = tff.simulation.baselines.stackoverflow.create_word_prediction_task(
        train_client_spec, use_synthetic_data=False)
    server_model_fn, client_model_fn = models.make_big_and_small_stackoverflow_model_fn(
        my_task,
        big_embedding_size=96 * 2,
        big_lstm_size=670 * 2,
        small_embedding_size=96,
        small_lstm_size=670)

    shrink_unshrink_info = simple_fedavg_tf.LayerwiseProjectionShrinkUnshrinkInfoV2(
        left_mask=[-1, 0, 2, -1, 2, -1, 0, -1],
        right_mask=[0, 1, 1, 1, 0, 0, -1, -1],
        build_projection_matrix=simple_fedavg_tf.build_normal_projection_matrix,
        new_projection_dict_decimate=1)
    _, shrink, unshrink, server_init_tf = simple_fedavg_tff.build_federated_shrink_unshrink_process(
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
    del server_init_tf

    self.assertEqual(
        str(shrink.type_signature),
        '(<server_state=<model_weights=<trainable=<float32[10004,192],float32[192,5360],float32[1340,5360],float32[5360],float32[1340,192],float32[192],float32[192,10004],float32[10004]>,non_trainable=<>>,optimizer_state=<int64>,round_num=int32,shrink_unshrink_server_info=<lmbda=float32,oja_left_maskval_to_projmat_dict=<-1=float32[1],0=float32[?,?],2=float32[?,?],1=float32[?,?]>,memory_dict=<0=float32[?,?],2=float32[?,?],1=float32[?,?]>>>@SERVER,federated_dataset={<int64[?,20],int64[?,20]>*}@CLIENTS> -> {<model_weights=<trainable=<tf.float32[None],tf.float32[None],tf.float32[None],tf.float32[None],tf.float32[None],tf.float32[None],tf.float32[None],tf.float32[None]>,non_trainable=<>>,round_num=int32,shrink_unshrink_dynamic_info=uint64>}@CLIENTS)'
    )
    self.assertEqual(
        str(unshrink.type_signature),
        '(<server_state=<model_weights=<trainable=<float32[10004,192],float32[192,5360],float32[1340,5360],float32[5360],float32[1340,192],float32[192],float32[192,10004],float32[10004]>,non_trainable=<>>,optimizer_state=<int64>,round_num=int32,shrink_unshrink_server_info=<lmbda=float32,oja_left_maskval_to_projmat_dict=<-1=float32[1],0=float32[?,?],2=float32[?,?],1=float32[?,?]>,memory_dict=<0=float32[?,?],2=float32[?,?],1=float32[?,?]>>>@SERVER,client_outputs={<weights_delta=<tf.float32[None],tf.float32[None],tf.float32[None],tf.float32[None],tf.float32[None],tf.float32[None],tf.float32[None],tf.float32[None]>,client_weight=float32,model_output=<loss=<float32,float32>>,round_num=int32,shrink_unshrink_dynamic_info=uint64>}@CLIENTS> -> <model_weights=<trainable=<float32[10004,192],float32[192,5360],float32[1340,5360],float32[5360],float32[1340,192],float32[192],float32[192,10004],float32[10004]>,non_trainable=<>>,optimizer_state=<int64>,round_num=int32,shrink_unshrink_server_info=<lmbda=float32,oja_left_maskval_to_projmat_dict=<-1=float32[1],0=float32[?,?],2=float32[?,?],1=float32[?,?]>,memory_dict=<0=float32[?,?],2=float32[?,?],1=float32[?,?]>>>@SERVER)'
    )


if __name__ == '__main__':
  tf.test.main()
