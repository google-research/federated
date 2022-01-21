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
"""Test the local k-means updates of the FedTKM algorithm."""

import collections
import numpy as np
import tensorflow as tf

from periodic_distribution_shift import fedavg_temporal_kmeans
from periodic_distribution_shift.models import keras_utils_dual_branch_kmeans


def create_dataset_c1(has_nan=False):
  # Create data approximately satisfying y = 2*x
  x = [[1.0], [2.0], [-3.0]]
  g = [0., 0., 0.]
  y = [[3.0], [5.0], [-7.0]]
  if has_nan:
    x[0][0] = np.nan
  return tf.data.Dataset.from_tensor_slices(
      collections.OrderedDict(x=(x, g), y=y)).batch(3)


def create_dataset_c2(has_nan=False):
  # Create data approximately satisfying y = 2*x
  x = [[-2.0], [-3.0], [-1.0]]
  g = [0., 0., 0.]
  y = [[-5.0], [-7.0], [-3.0]]
  if has_nan:
    x[0][0] = np.nan
  return tf.data.Dataset.from_tensor_slices(
      collections.OrderedDict(x=(x, g), y=y)).batch(3)


def get_input_spec():
  return create_dataset_c1().element_spec


def create_dual_branch_model() -> tf.keras.Model:
  """Create a dual-branch convolutional network.

  Returns:
    An uncompiled `tf.keras.Model`.
  """
  image = tf.keras.layers.Input(shape=(1,), name='image')
  group = tf.keras.layers.Input(shape=(None,), name='group')

  model = tf.keras.layers.Dense(
      1,
      kernel_initializer=tf.keras.initializers.Ones(),
      bias_initializer='zeros',
      input_shape=(1,))

  feature = model(image)
  final_fc_list = [
      tf.keras.layers.Dense(
          1,
          activation=None,
          name='branch_1',
          bias_initializer='zeros',
      ),
      tf.keras.layers.Dense(
          1,
          activation=None,
          name='branch_2',
          bias_initializer='zeros',
      )
  ]

  pred1 = final_fc_list[0](feature)
  pred2 = final_fc_list[1](feature)

  output = [pred1, pred2, feature]

  return tf.keras.Model(inputs=[image, group], outputs=output)


def model_builder():
  keras_model = create_dual_branch_model()
  return keras_utils_dual_branch_kmeans.from_keras_model(
      keras_model=keras_model,
      input_spec=get_input_spec(),
      loss=tf.keras.losses.MeanSquaredError())


class ModelDeltaProcessTest(tf.test.TestCase):

  def test_fed_avg_without_schedule_decreases_loss(self):
    iterproc = fedavg_temporal_kmeans.build_fed_avg_process(
        model_builder,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        client_lr=0.0,
        server_optimizer_fn=tf.keras.optimizers.SGD,
        kmeans_k=2,
        feature_dim=1,
        aggregated_kmeans=True,
        clip_norm=-1.)

    initial_state = iterproc.initialize()

    # Two clients. First client is destined to be assigned to the first
    # cluster, second client will be assigned to the second cluster.
    federated_data = [create_dataset_c1(), create_dataset_c2()]
    initial_state.kmeans_centers[0] = 1.
    initial_state.kmeans_centers[1] = -2.

    state = initial_state

    (_, _, _, kmeans_delta_sum, kmeans_n_samples,
     cluster1_ratio) = iterproc.next(state, federated_data)
    self.assertAllClose(kmeans_n_samples, np.array([[3], [3]]))
    self.assertEqual(cluster1_ratio, 0.5)
    # [1, 2, -3] -> 1, delta_sum=-3; [-2, -3, -1] -> -2, delta_sum=0
    self.assertAllClose(kmeans_delta_sum, np.array([[-3], [0]]))


if __name__ == '__main__':
  tf.test.main()
