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
import asyncio
import collections
import os
import tensorflow as tf
import tensorflow_federated as tff
from personalization_benchmark.cross_device.algorithms import checkpoint_utils


def model_fn(initializer='zeros'):
  keras_model = tf.keras.Sequential([
      tf.keras.layers.Dense(
          1,
          kernel_initializer=initializer,
          bias_initializer=initializer,
          input_shape=(1,))
  ])
  input_spec = collections.OrderedDict(
      x=tf.TensorSpec([None, 1], dtype=tf.float32),
      y=tf.TensorSpec([None, 1], dtype=tf.float32))
  return tff.learning.from_keras_model(
      keras_model=keras_model,
      input_spec=input_spec,
      loss=tf.keras.losses.MeanSquaredError())


def create_program_state(initializer='zeros'):
  learning_process = tff.learning.algorithms.build_weighted_fed_avg(
      model_fn=lambda: model_fn(initializer),
      # The actual learning rate values do not matter here.
      client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.5),
      server_optimizer_fn=lambda: tf.keras.optimizers.Adam(2.0),
      model_aggregator=tff.learning.robust_aggregator())
  return learning_process.initialize()


class CheckpointUtilsTest(tf.test.TestCase):

  def test_load_model_weights_succeeds(self):
    path_to_checkpoint_dir = self.get_temp_dir()
    # Save two fake `program_state_*`s to the temp dir.
    state_manager = tff.program.FileProgramStateManager(
        root_dir=path_to_checkpoint_dir, prefix='program_state_')
    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        state_manager.save(
            program_state=create_program_state('zeros'), version=0))
    loop.run_until_complete(
        state_manager.save(
            program_state=create_program_state('ones'), version=1))
    self.assertTrue(
        os.path.exists(os.path.join(path_to_checkpoint_dir, 'program_state_0')))
    self.assertTrue(
        os.path.exists(os.path.join(path_to_checkpoint_dir, 'program_state_1')))
    # Load the latest `program_state_*`. The model weights should be ones.
    loaded_model_weights = checkpoint_utils.extract_model_weights_from_checkpoint(
        model_fn=model_fn, path_to_checkpoint_dir=path_to_checkpoint_dir)
    tf.nest.map_structure(
        self.assertEqual,
        tff.learning.ModelWeights.from_model(model_fn('ones')),
        loaded_model_weights)
    # Load the `program_state_*` at round 0. The model weights should be zeros.
    loaded_model_weights = checkpoint_utils.extract_model_weights_from_checkpoint(
        model_fn=model_fn,
        path_to_checkpoint_dir=path_to_checkpoint_dir,
        round_num=0)
    tf.nest.map_structure(
        self.assertEqual,
        tff.learning.ModelWeights.from_model(model_fn('zeros')),
        loaded_model_weights)


if __name__ == '__main__':
  tf.test.main()
