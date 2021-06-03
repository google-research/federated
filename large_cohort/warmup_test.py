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

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_federated as tff

from large_cohort import warmup


def keras_model_builder():
  return tf.keras.Sequential([
      tf.keras.layers.Dense(
          1, kernel_initializer='zeros', use_bias=False, input_shape=(1,))
  ])


class WarmupTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('param1', 1.0, 5),
      ('param2', 0.2, 3),
      ('param3', 6.0, 10),
      ('param4', 3.5, 1),
  )
  def test_correctness_during_warmup(self, max_learning_rate, warmup_steps):
    warmup_schedule = warmup.WarmupSchedule(max_learning_rate, warmup_steps)
    for i in range(warmup_steps):
      expected_learning_rate = ((i + 1) / warmup_steps) * max_learning_rate
      self.assertNear(warmup_schedule(i), expected_learning_rate, err=1e-6)

  @parameterized.named_parameters(
      ('param1', 1.0, 5),
      ('param2', 0.2, 3),
      ('param3', 6.0, 10),
      ('param4', 3.5, 1),
  )
  def test_correctness_after_warmup(self, max_learning_rate, warmup_steps):
    warmup_schedule = warmup.WarmupSchedule(max_learning_rate, warmup_steps)
    for i in range(warmup_steps, warmup_steps + 10):
      self.assertNear(warmup_schedule(i), max_learning_rate, err=1e-6)

  def test_warmup_used_by_tff(self):
    max_learning_rate = 3.0
    warmup_steps = 3
    x = [[1.0]]
    y = [[3.0]]
    dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(1)

    def tff_model_fn():
      return tff.learning.from_keras_model(
          keras_model=keras_model_builder(),
          input_spec=dataset.element_spec,
          loss=tf.keras.losses.MeanSquaredError())

    def server_optimizer_fn():
      warmup_schedule = warmup.WarmupSchedule(max_learning_rate, warmup_steps)
      return tf.keras.optimizers.SGD(warmup_schedule)

    fed_sgd_process = tff.learning.build_federated_sgd_process(
        tff_model_fn, server_optimizer_fn)
    state = fed_sgd_process.initialize()

    keras_model = keras_model_builder()
    keras_model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=server_optimizer_fn())

    for _ in range(10):
      keras_model.fit(dataset)
      state, _ = fed_sgd_process.next(state, [dataset])
      keras_model_weights = keras_model.weights[0].numpy()
      tff_model_weights = state.model.trainable[0]
      self.assertAllClose(keras_model_weights, tff_model_weights, rtol=1e-6)


if __name__ == '__main__':
  tf.test.main()
