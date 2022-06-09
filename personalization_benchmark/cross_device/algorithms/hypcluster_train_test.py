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
from absl.testing import parameterized
import tensorflow as tf
import tensorflow_federated as tff

from personalization_benchmark.cross_device.algorithms import hypcluster_train


def create_dataset():
  # Create data satisfying y = x + 1
  x = [[1.0], [2.0], [3.0]]
  y = [[2.0], [3.0], [4.0]]
  return tf.data.Dataset.from_tensor_slices((x, y)).batch(1)


def get_input_spec():
  return create_dataset().element_spec


def model_fn(initializer='zeros'):
  keras_model = tf.keras.Sequential([
      tf.keras.layers.Dense(
          1,
          kernel_initializer=initializer,
          bias_initializer=initializer,
          input_shape=(1,))
  ])
  return tff.learning.from_keras_model(
      keras_model=keras_model,
      input_spec=get_input_spec(),
      loss=tf.keras.losses.MeanSquaredError())


@tff.tf_computation
def create_nested_structure():
  return [
      dict(a=tf.zeros((2, 2), dtype=tf.int32), b=1, c=3),
      dict(a=tf.ones((2, 2), dtype=tf.int32), b=2, c=4),
      dict(a=2 * tf.ones((2, 2), dtype=tf.int32), b=3, c=5),
  ]


def create_initial_models(num_models: int):
  model = model_fn(initializer='ones')
  model_weights_tensors = tf.nest.map_structure(
      lambda var: var.numpy(), tff.learning.ModelWeights.from_model(model))
  return [model_weights_tensors for _ in range(num_models)]


class HypClusterTrainTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('clusters1_with_init', 1, create_initial_models(1)),
      ('clusters1_without_init', 1, None),
      ('clusters2_without_int', 2, None),
      ('clusters3_with_init', 3, create_initial_models(3)),
      ('clusters5_without_init', 5, None),
  )
  def test_constructs_with_default_aggregator(self, num_clusters,
                                              initial_model_weights_list):
    client_optimizer = tff.learning.optimizers.build_sgdm(learning_rate=0.01)
    server_optimizer = tff.learning.optimizers.build_sgdm(learning_rate=1.0)
    hyp_alg = hypcluster_train.build_hypcluster_train(
        model_fn=model_fn,
        num_clusters=num_clusters,
        client_optimizer=client_optimizer,
        server_optimizer=server_optimizer,
        initial_model_weights_list=initial_model_weights_list)
    state = hyp_alg.initialize()
    self.assertLen(state.global_model_weights, num_clusters)
    self.assertLen(state.aggregator, num_clusters)
    self.assertLen(state.finalizer, num_clusters)
    if initial_model_weights_list:
      tf.nest.map_structure(self.assertAllEqual, state.global_model_weights,
                            initial_model_weights_list)

  @parameterized.named_parameters(
      ('clusters1_with_init', 1, create_initial_models(1)),
      ('clusters1_without_init', 1, None),
      ('clusters2_without_int', 2, None),
      ('clusters3_with_init', 3, create_initial_models(3)),
      ('clusters5_without_init', 5, None),
  )
  def test_constructs_with_non_default_aggregator(self, num_clusters,
                                                  initial_model_weights_list):
    client_optimizer = tff.learning.optimizers.build_sgdm(learning_rate=0.01)
    server_optimizer = tff.learning.optimizers.build_sgdm(learning_rate=1.0)
    hyp_alg = hypcluster_train.build_hypcluster_train(
        model_fn=model_fn,
        num_clusters=num_clusters,
        client_optimizer=client_optimizer,
        server_optimizer=server_optimizer,
        model_aggregator=tff.learning.robust_aggregator(),
        initial_model_weights_list=initial_model_weights_list)
    state = hyp_alg.initialize()
    self.assertLen(state.global_model_weights, num_clusters)
    self.assertLen(state.aggregator, num_clusters)
    self.assertLen(state.finalizer, num_clusters)
    if initial_model_weights_list:
      tf.nest.map_structure(self.assertAllEqual, state.global_model_weights,
                            initial_model_weights_list)

  def test_construction_fails_with_mismatched_initial_models(self):
    num_clusters = 1
    initial_model_weights_list = create_initial_models(2)
    client_optimizer = tff.learning.optimizers.build_sgdm(learning_rate=0.01)
    server_optimizer = tff.learning.optimizers.build_sgdm(learning_rate=1.0)
    with self.assertRaisesRegex(ValueError, 'does not equal'):
      hypcluster_train.build_hypcluster_train(
          model_fn=model_fn,
          num_clusters=num_clusters,
          client_optimizer=client_optimizer,
          server_optimizer=server_optimizer,
          initial_model_weights_list=initial_model_weights_list)

  def test_matches_fed_avg_with_one_cluster(self):
    client_optimizer = tff.learning.optimizers.build_sgdm(learning_rate=0.01)
    server_optimizer = tff.learning.optimizers.build_sgdm(learning_rate=1.0)
    fed_avg = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn=model_fn,
        client_optimizer_fn=client_optimizer,
        server_optimizer_fn=server_optimizer)
    hyp_alg = hypcluster_train.build_hypcluster_train(
        model_fn=model_fn,
        num_clusters=1,
        client_optimizer=client_optimizer,
        server_optimizer=server_optimizer)

    fed_avg_state = fed_avg.initialize()
    hyp_alg_state = hyp_alg.initialize()
    self.assertAllClose(
        fed_avg.get_model_weights(fed_avg_state).trainable,
        hyp_alg.get_model_weights(hyp_alg_state)[0].trainable)
    federated_data = [create_dataset(), create_dataset()]
    for _ in range(5):
      fed_avg_output = fed_avg.next(fed_avg_state, federated_data)
      fed_avg_state = fed_avg_output.state
      hyp_alg_output = hyp_alg.next(hyp_alg_state, federated_data)
      hyp_alg_state = hyp_alg_output.state
      self.assertAllClose(
          fed_avg.get_model_weights(fed_avg_state).trainable,
          hyp_alg.get_model_weights(hyp_alg_state)[0].trainable)


if __name__ == '__main__':
  tf.test.main()
