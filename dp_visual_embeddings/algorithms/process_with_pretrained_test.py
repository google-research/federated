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
"""Tests for process_with_pretrained."""
import collections
import functools
from absl.testing import parameterized
import tensorflow as tf
import tensorflow_federated as tff

from dp_visual_embeddings.algorithms import federated_partial
from dp_visual_embeddings.algorithms import process_with_pretrained
from dp_visual_embeddings.models import keras_utils

_DATA_DIM = 2
_DATA_ELEMENT_SPEC = collections.OrderedDict(
    x=tf.TensorSpec([None, _DATA_DIM], dtype=tf.float32),
    y=tf.TensorSpec([None, 1], dtype=tf.int32))


def _keras_model(output_size: int = 2,
                 non_trainable_global=True) -> keras_utils.EmbeddingModel:
  global_layers = [tf.keras.layers.Dense(5), tf.keras.layers.Dense(3)]
  if non_trainable_global:
    global_layers[1].trainable = False
  client_layers = [tf.keras.layers.Dense(3), tf.keras.layers.Dense(output_size)]
  client_layers[0].trainable = False
  inputs = tf.keras.Input(shape=(_DATA_DIM,))
  x = inputs
  for layer in global_layers + client_layers:
    x = layer(x)
  keras_model = tf.keras.Model(inputs=inputs, outputs=x)

  global_variables = tff.learning.ModelWeights(
      trainable=global_layers[0].trainable_variables,
      non_trainable=global_layers[1].non_trainable_variables)
  client_variables = tff.learning.ModelWeights(
      trainable=client_layers[1].trainable_variables,
      non_trainable=client_layers[0].non_trainable_variables)
  return keras_utils.EmbeddingModel(keras_model, global_variables,
                                    client_variables)


def _tff_model_fn():
  model = _keras_model()
  return keras_utils.from_keras_model(
      model.model,
      # The loss is not used for inference.
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      input_spec=_DATA_ELEMENT_SPEC,
      global_variables=model.global_variables,
      client_variables=model.client_variables)


def _tff_learning_model_fn():
  model = _keras_model()
  return tff.learning.from_keras_model(
      model.model,
      # The loss is not used for inference.
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      input_spec=_DATA_ELEMENT_SPEC)


def _create_dataset():
  # Create a dataset with 4 examples:
  dataset = tf.data.Dataset.from_tensor_slices(
      collections.OrderedDict(
          x=[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
          y=[[0], [0], [1], [1]]))
  # Repeat the dataset 2 times with batches of 3 examples,
  # producing 3 minibatches (the last one with only 2 examples).
  # Note that `batch` is required for this dataset to be useable,
  # as it adds the batch dimension which is expected by the model.
  return dataset.repeat(2).batch(3)


class ProcessWithPretrainedTest(tf.test.TestCase, parameterized.TestCase):

  def test_load_pretrained_model(self):
    # Create a model and save it to disk. The model has different last layer
    # (client variables) compared to the TFF model later.
    save_model = _keras_model(output_size=4, non_trainable_global=True)
    filepath = self.create_tempdir()
    save_model.model.save(filepath)
    save_global_variables = [
        save_model.global_variables.trainable,
        save_model.global_variables.non_trainable
    ]

    # Create a keras model that has the same structure as the saved model to
    # separate the global variables for TFF model.
    load_model = _keras_model(output_size=4)
    load_global_variables = [
        load_model.global_variables.trainable,
        load_model.global_variables.non_trainable
    ]
    self.assertNotAllClose(save_global_variables, load_global_variables)
    process_with_pretrained._load_from_keras_model(filepath, load_model)
    self.assertAllClose(save_global_variables, load_global_variables)

    # Verify load the global variables from saved model to TFF model.
    tff_model = _tff_model_fn()
    model_weights = tff.learning.ModelWeights.from_model(tff_model)
    tff_model_variables = [model_weights.trainable, model_weights.non_trainable]
    self.assertNotAllClose(save_global_variables, tff_model_variables)
    load_model.global_variables.assign_weights_to(tff_model)
    self.assertAllClose(save_global_variables, tff_model_variables)

  @parameterized.named_parameters([
      ('fedpartial', None),
      ('reconst', 1),
  ])
  def test_initialize_process(self, reconst_iters):
    # Create a model and save it to disk. The model has different last layer
    # (client variables) compared to the TFF model later.
    pretrained_model_fn = functools.partial(_keras_model, output_size=4)
    save_model = pretrained_model_fn()
    filepath = self.create_tempdir()
    save_model.model.save(filepath)
    save_global_variables = [
        save_model.global_variables.trainable,
        save_model.global_variables.non_trainable
    ]

    # Verify plain `train_process` without loading saved model uses differet
    # weights for initialization.
    train_process = federated_partial.build_unweighted_averaging_with_optimizer_schedule(
        model_fn=_tff_model_fn,
        client_learning_rate_fn=lambda x: 0.1,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        reconst_iters=reconst_iters)
    state = train_process.initialize()
    state_model_weights = train_process.get_model_weights(state)
    state_model_variables = [
        state_model_weights.trainable, state_model_weights.non_trainable
    ]
    self.assertNotAllClose(save_global_variables, state_model_variables)

    # Verify `init_process` by `process_with_pretrained` can load saved model
    # for initialization.
    init_process = process_with_pretrained.build_process_with_pretrained(
        train_process=train_process,
        pretrained_model_fn=pretrained_model_fn,
        pretrained_model_path=filepath)
    state = init_process.initialize()
    state_model_weights = init_process.get_model_weights(state)
    state_model_variables = [
        state_model_weights.trainable, state_model_weights.non_trainable
    ]
    self.assertAllClose(save_global_variables, state_model_variables)

    # Verify one round of `init_process.next` will update `trainable` global
    # variables.
    outputs = init_process.next(state, [_create_dataset()])
    state_model_weights = init_process.get_model_weights(outputs.state)
    self.assertNotAllClose(save_model.global_variables.trainable,
                           state_model_weights.trainable)
    self.assertAllClose(save_model.global_variables.non_trainable,
                        state_model_weights.non_trainable)

  def test_initialize_fedavg_process(self):
    # Create a model and save it to disk.
    save_model = _keras_model().model
    filepath = self.create_tempdir()
    save_model.save(filepath)
    save_variables = [
        save_model.trainable_variables, save_model.non_trainable_variables
    ]

    # Verify plain `train_process` without loading saved model uses differet
    # weights for initialization.
    train_process = tff.learning.algorithms.build_unweighted_fed_avg(
        model_fn=_tff_learning_model_fn,
        server_optimizer_fn=tf.keras.optimizers.SGD,
        client_optimizer_fn=tf.keras.optimizers.SGD,
        use_experimental_simulation_loop=True)
    state = train_process.initialize()
    state_model_weights = train_process.get_model_weights(state)
    state_model_variables = [
        state_model_weights.trainable, state_model_weights.non_trainable
    ]
    self.assertNotAllClose(save_variables, state_model_variables)

    # Verify `init_process` by `process_with_pretrained` can load saved model
    # for initialization.
    init_process = process_with_pretrained.build_fedavg_process_with_pretrained(
        train_process=train_process, pretrained_model_path=filepath)
    state = init_process.initialize()
    state_model_weights = init_process.get_model_weights(state)
    state_model_variables = [
        state_model_weights.trainable, state_model_weights.non_trainable
    ]
    self.assertAllClose(save_variables, state_model_variables)

    # Verify one round of `init_process.next` will update `trainable` global
    # variables.
    outputs = init_process.next(state, [_create_dataset()])
    state_model_weights = init_process.get_model_weights(outputs.state)
    self.assertNotAllClose(save_model.trainable_variables,
                           state_model_weights.trainable)
    self.assertAllClose(save_model.non_trainable_variables,
                        state_model_weights.non_trainable)

  def test_invalid_save_model_path(self):
    filepath = self.create_tempdir()
    train_process = federated_partial.build_unweighted_averaging_with_optimizer_schedule(
        model_fn=_tff_model_fn,
        client_learning_rate_fn=lambda x: 0.1,
        client_optimizer_fn=tf.keras.optimizers.SGD)
    with self.assertRaises(IOError):
      init_process = process_with_pretrained.build_process_with_pretrained(
          train_process=train_process,
          pretrained_model_fn=_keras_model,
          pretrained_model_path=filepath)
      init_process.initialize()

  def test_invalid_save_model_shape(self):
    save_model = _keras_model(output_size=4)
    filepath = self.create_tempdir()
    save_model.model.save(filepath)
    train_process = federated_partial.build_unweighted_averaging_with_optimizer_schedule(
        model_fn=_tff_model_fn,
        client_learning_rate_fn=lambda x: 0.1,
        client_optimizer_fn=tf.keras.optimizers.SGD)
    with self.assertRaises(ValueError):
      init_process = process_with_pretrained.build_process_with_pretrained(
          train_process=train_process,
          pretrained_model_fn=_keras_model,
          pretrained_model_path=filepath)
      init_process.initialize()


if __name__ == '__main__':
  tf.test.main()
