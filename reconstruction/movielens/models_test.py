# Copyright 2020, Google LLC.
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
"""Tests for models.py."""

import collections
import functools

from absl.testing import absltest
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from reconstruction import reconstruction_utils
from reconstruction import training_process
from reconstruction.movielens import models

# Keras metric names.
_KERAS_LOSS = 'loss'
_KERAS_NUM_EXAMPLES = 'num_examples'
_KERAS_NUM_BATCHES = 'num_batches'
_KERAS_ACCURACY = 'reconstruction_accuracy_metric'


def count_trainable_params(keras_model):
  return np.sum(
      [tf.keras.backend.count_params(w) for w in keras_model.trainable_weights])


class ModelsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    # Toy data to test model fitting.
    self.train_users = np.array([[0], [1], [0], [1], [2]])
    self.train_items = np.array([[7], [5], [5], [7], [7]])
    self.train_preferences = np.array([[0.0], [1.0], [1.0], [0.0], [0.0]])

  def test_matrix_factorization_trains_keras(self):
    num_users = 10
    num_items = 8
    num_latent_factors = 5
    personal_model = False
    add_biases = False
    l2_regularization = 0.0
    learning_rate = 0.5

    matrix_factorization_model = models.get_matrix_factorization_model(
        num_users,
        num_items,
        num_latent_factors,
        personal_model=personal_model,
        add_biases=add_biases,
        l2_regularization=l2_regularization)

    keras_model = models.build_keras_model(
        matrix_factorization_model, tf.keras.optimizers.SGD(learning_rate))

    # Ensure number of parameters of model is as expected as a quick check.
    expected_num_params = (
        num_users * num_latent_factors  # User embeddings.
        + num_items * num_latent_factors)  # Item embeddings.
    num_params = keras_model.count_params()
    self.assertEqual(expected_num_params, num_params)
    num_trainable_params = count_trainable_params(keras_model)
    self.assertEqual(expected_num_params, num_trainable_params)

    history = keras_model.fit([self.train_users, self.train_items],
                              self.train_preferences,
                              batch_size=1,
                              epochs=1)

    # Ensure the model has a valid loss after one epoch (not NaN).
    self.assertIn(_KERAS_LOSS, history.history)
    losses = history.history[_KERAS_LOSS]
    self.assertLen(losses, 1)
    self.assertFalse(np.isnan(losses[0]))

    # Ensure the model has valid number of examples after one epoch (not NaN).
    self.assertIn(_KERAS_NUM_EXAMPLES, history.history)
    num_examples = history.history[_KERAS_NUM_EXAMPLES]
    self.assertLen(num_examples, 1)
    self.assertEqual(num_examples[0], 5)

    # Ensure the model has valid number of batches after one epoch (not NaN).
    self.assertIn(_KERAS_NUM_BATCHES, history.history)
    num_batches = history.history[_KERAS_NUM_BATCHES]
    self.assertLen(num_batches, 1)
    self.assertEqual(num_batches[0], 5)

  def test_matrix_factorization_trains_keras_add_biases(self):
    num_users = 10
    num_items = 8
    num_latent_factors = 5
    personal_model = False
    add_biases = True
    l2_regularization = 0.0

    matrix_factorization_model = models.get_matrix_factorization_model(
        num_users,
        num_items,
        num_latent_factors,
        personal_model=personal_model,
        add_biases=add_biases,
        l2_regularization=l2_regularization)

    keras_model = models.build_keras_model(matrix_factorization_model)

    # Ensure number of parameters of model is as expected as a quick check.
    expected_num_params = (
        num_users * num_latent_factors  # User embeddings.
        + num_items * num_latent_factors  # Item embeddings.
        + num_users  # User-specific biases.
        + num_items  # Item-specific biases.
        + 1)  # Global bias.
    num_params = keras_model.count_params()
    self.assertEqual(expected_num_params, num_params)
    num_trainable_params = count_trainable_params(keras_model)
    self.assertEqual(expected_num_params, num_trainable_params)

    history = keras_model.fit([self.train_users, self.train_items],
                              self.train_preferences,
                              batch_size=1,
                              epochs=1)

    # Ensure the model has a valid loss after one epoch (not NaN).
    self.assertIn(_KERAS_LOSS, history.history)
    losses = history.history[_KERAS_LOSS]
    self.assertLen(losses, 1)
    self.assertFalse(np.isnan(losses[0]))

    # Ensure the model has valid number of examples after one epoch (not NaN).
    self.assertIn(_KERAS_NUM_EXAMPLES, history.history)
    num_examples = history.history[_KERAS_NUM_EXAMPLES]
    self.assertLen(num_examples, 1)
    self.assertEqual(num_examples[0], 5)

    # Ensure the model has valid number of batches after one epoch (not NaN).
    self.assertIn(_KERAS_NUM_BATCHES, history.history)
    num_batches = history.history[_KERAS_NUM_BATCHES]
    self.assertLen(num_batches, 1)
    self.assertEqual(num_batches[0], 5)

  def test_matrix_factorization_trains_keras_l2_regularization(self):
    num_users = 10
    num_items = 8
    num_latent_factors = 5
    personal_model = False
    add_biases = False
    l2_regularization = 0.5

    matrix_factorization_model = models.get_matrix_factorization_model(
        num_users,
        num_items,
        num_latent_factors,
        personal_model=personal_model,
        add_biases=add_biases,
        l2_regularization=l2_regularization)

    keras_model = models.build_keras_model(matrix_factorization_model)

    # Ensure number of parameters of model is as expected as a quick check.
    expected_num_params = (
        num_users * num_latent_factors  # User embeddings.
        + num_items * num_latent_factors)  # Item embeddings.
    num_params = keras_model.count_params()
    self.assertEqual(expected_num_params, num_params)
    num_trainable_params = count_trainable_params(keras_model)
    self.assertEqual(expected_num_params, num_trainable_params)

    history = keras_model.fit([self.train_users, self.train_items],
                              self.train_preferences,
                              batch_size=1,
                              epochs=1)

    # Ensure the model has a valid loss after one epoch (not NaN).
    self.assertIn(_KERAS_LOSS, history.history)
    losses = history.history[_KERAS_LOSS]
    self.assertLen(losses, 1)
    self.assertFalse(np.isnan(losses[0]))

    # Ensure the model has valid number of examples after one epoch (not NaN).
    self.assertIn(_KERAS_NUM_EXAMPLES, history.history)
    num_examples = history.history[_KERAS_NUM_EXAMPLES]
    self.assertLen(num_examples, 1)
    self.assertEqual(num_examples[0], 5)

    # Ensure the model has valid number of batches after one epoch (not NaN).
    self.assertIn(_KERAS_NUM_BATCHES, history.history)
    num_batches = history.history[_KERAS_NUM_BATCHES]
    self.assertLen(num_batches, 1)
    self.assertEqual(num_batches[0], 5)

  def test_matrix_factorization_trains_keras_batch_size_2(self):
    num_users = 10
    num_items = 8
    num_latent_factors = 5
    personal_model = False
    add_biases = False
    l2_regularization = 0.0

    matrix_factorization_model = models.get_matrix_factorization_model(
        num_users,
        num_items,
        num_latent_factors,
        personal_model=personal_model,
        add_biases=add_biases,
        l2_regularization=l2_regularization)

    keras_model = models.build_keras_model(matrix_factorization_model)

    # Ensure number of parameters of model is as expected as a quick check.
    expected_num_params = (
        num_users * num_latent_factors  # User embeddings.
        + num_items * num_latent_factors)  # Item embeddings.
    num_params = keras_model.count_params()
    self.assertEqual(expected_num_params, num_params)
    num_trainable_params = count_trainable_params(keras_model)
    self.assertEqual(expected_num_params, num_trainable_params)

    history = keras_model.fit([self.train_users, self.train_items],
                              self.train_preferences,
                              batch_size=2,
                              epochs=1)

    # Ensure the model has a valid loss after one epoch (not NaN).
    self.assertIn(_KERAS_LOSS, history.history)
    losses = history.history[_KERAS_LOSS]
    self.assertLen(losses, 1)
    self.assertFalse(np.isnan(losses[0]))

    # Ensure the model has valid number of examples after one epoch (not NaN).
    self.assertIn(_KERAS_NUM_EXAMPLES, history.history)
    num_examples = history.history[_KERAS_NUM_EXAMPLES]
    self.assertLen(num_examples, 1)
    self.assertEqual(num_examples[0], 5)

    # Ensure the model has valid number of batches after one epoch (not NaN).
    self.assertIn(_KERAS_NUM_BATCHES, history.history)
    num_batches = history.history[_KERAS_NUM_BATCHES]
    self.assertLen(num_batches, 1)
    self.assertEqual(num_batches[0], 3)

  def test_matrix_factorization_trains_tff(self):
    train_data = [
        self.train_users.flatten().tolist(),
        self.train_items.flatten().tolist(),
        self.train_preferences.flatten().tolist()
    ]
    train_tf_dataset = tf.data.Dataset.from_tensor_slices(
        list(zip(*train_data)))

    def batch_map_fn(example_batch):
      return collections.OrderedDict(
          x=(tf.cast(example_batch[:, 0:1],
                     tf.int64), tf.cast(example_batch[:, 1:2], tf.int64)),
          y=example_batch[:, 2:3])

    train_tf_dataset = train_tf_dataset.batch(1).map(batch_map_fn).repeat(5)
    train_tf_datasets = [train_tf_dataset] * 2

    num_users = 10
    num_items = 8
    num_latent_factors = 10
    personal_model = False
    add_biases = False
    l2_regularization = 0.0

    tff_model_fn = models.build_tff_model(
        functools.partial(
            models.get_matrix_factorization_model,
            num_users,
            num_items,
            num_latent_factors,
            personal_model=personal_model,
            add_biases=add_biases,
            l2_regularization=l2_regularization))

    trainer = tff.learning.build_federated_averaging_process(
        tff_model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1e-2))

    state = trainer.initialize()
    trainer.next(state, train_tf_datasets)

  def test_personal_matrix_factorization_trains_keras(self):
    num_users = 1
    num_items = 8
    num_latent_factors = 5
    personal_model = True
    add_biases = False
    l2_regularization = 0.0

    matrix_factorization_model = models.get_matrix_factorization_model(
        num_users,
        num_items,
        num_latent_factors,
        personal_model=personal_model,
        add_biases=add_biases,
        l2_regularization=l2_regularization)

    keras_model = models.build_keras_model(matrix_factorization_model)

    # Ensure number of parameters of model is as expected as a quick check.
    expected_num_params = (
        num_latent_factors  # User embeddings.
        + num_items * num_latent_factors)  # Item embeddings.
    num_params = keras_model.count_params()
    self.assertEqual(expected_num_params, num_params)
    num_trainable_params = count_trainable_params(keras_model)
    self.assertEqual(expected_num_params, num_trainable_params)

    history = keras_model.fit(
        self.train_items, self.train_preferences, batch_size=1, epochs=1)

    # Ensure the model has a valid loss after one epoch (not NaN).
    self.assertIn(_KERAS_LOSS, history.history)
    losses = history.history[_KERAS_LOSS]
    self.assertLen(losses, 1)
    self.assertFalse(np.isnan(losses[0]))

    # Ensure the model has valid number of examples after one epoch (not NaN).
    self.assertIn(_KERAS_NUM_EXAMPLES, history.history)
    num_examples = history.history[_KERAS_NUM_EXAMPLES]
    self.assertLen(num_examples, 1)
    self.assertEqual(num_examples[0], 5)

    # Ensure the model has valid number of batches after one epoch (not NaN).
    self.assertIn(_KERAS_NUM_BATCHES, history.history)
    num_batches = history.history[_KERAS_NUM_BATCHES]
    self.assertLen(num_batches, 1)
    self.assertEqual(num_batches[0], 5)

  def test_personal_matrix_factorization_trains_multiple_users_fails(self):
    num_users = 2
    num_items = 8
    num_latent_factors = 5
    personal_model = True
    add_biases = False
    l2_regularization = 0.0

    with self.assertRaises(ValueError):
      models.get_matrix_factorization_model(
          num_users,
          num_items,
          num_latent_factors,
          personal_model=personal_model,
          add_biases=add_biases,
          l2_regularization=l2_regularization)

  def test_personal_matrix_factorization_trains_keras_add_biases(self):
    num_users = 1
    num_items = 8
    num_latent_factors = 5
    personal_model = True
    add_biases = True
    l2_regularization = 0.0

    matrix_factorization_model = models.get_matrix_factorization_model(
        num_users,
        num_items,
        num_latent_factors,
        personal_model=personal_model,
        add_biases=add_biases,
        l2_regularization=l2_regularization)

    keras_model = models.build_keras_model(matrix_factorization_model)

    # Ensure number of parameters of model is as expected as a quick check.
    expected_num_params = (
        num_latent_factors  # User embeddings.
        + num_items * num_latent_factors  # Item embeddings.
        + 1  # User-specific biases.
        + num_items  # Item-specific biases.
        + 1)  # Global bias.
    num_params = keras_model.count_params()
    self.assertEqual(expected_num_params, num_params)
    num_trainable_params = count_trainable_params(keras_model)
    self.assertEqual(expected_num_params, num_trainable_params)

    history = keras_model.fit(
        self.train_items, self.train_preferences, batch_size=1, epochs=1)

    # Ensure the model has a valid loss after one epoch (not NaN).
    self.assertIn(_KERAS_LOSS, history.history)
    losses = history.history[_KERAS_LOSS]
    self.assertLen(losses, 1)
    self.assertFalse(np.isnan(losses[0]))

    # Ensure the model has valid number of examples after one epoch (not NaN).
    self.assertIn(_KERAS_NUM_EXAMPLES, history.history)
    num_examples = history.history[_KERAS_NUM_EXAMPLES]
    self.assertLen(num_examples, 1)
    self.assertEqual(num_examples[0], 5)

    # Ensure the model has valid number of batches after one epoch (not NaN).
    self.assertIn(_KERAS_NUM_BATCHES, history.history)
    num_batches = history.history[_KERAS_NUM_BATCHES]
    self.assertLen(num_batches, 1)
    self.assertEqual(num_batches[0], 5)

  def test_personal_matrix_factorization_trains_keras_l2_regularization(self):
    num_users = 1
    num_items = 8
    num_latent_factors = 5
    personal_model = True
    add_biases = False
    l2_regularization = 0.5

    matrix_factorization_model = models.get_matrix_factorization_model(
        num_users,
        num_items,
        num_latent_factors,
        personal_model=personal_model,
        add_biases=add_biases,
        l2_regularization=l2_regularization)

    keras_model = models.build_keras_model(matrix_factorization_model)

    # Ensure number of parameters of model is as expected as a quick check.
    expected_num_params = (
        num_latent_factors  # User embeddings.
        + num_items * num_latent_factors)  # Item embeddings.
    num_params = keras_model.count_params()
    self.assertEqual(expected_num_params, num_params)
    num_trainable_params = count_trainable_params(keras_model)
    self.assertEqual(expected_num_params, num_trainable_params)

    history = keras_model.fit(
        self.train_items, self.train_preferences, batch_size=1, epochs=1)

    # Ensure the model has a valid loss after one epoch (not NaN).
    self.assertIn(_KERAS_LOSS, history.history)
    losses = history.history[_KERAS_LOSS]
    self.assertLen(losses, 1)
    self.assertFalse(np.isnan(losses[0]))

    # Ensure the model has valid number of examples after one epoch (not NaN).
    self.assertIn(_KERAS_NUM_EXAMPLES, history.history)
    num_examples = history.history[_KERAS_NUM_EXAMPLES]
    self.assertLen(num_examples, 1)
    self.assertEqual(num_examples[0], 5)

    # Ensure the model has valid number of batches after one epoch (not NaN).
    self.assertIn(_KERAS_NUM_BATCHES, history.history)
    num_batches = history.history[_KERAS_NUM_BATCHES]
    self.assertLen(num_batches, 1)
    self.assertEqual(num_batches[0], 5)

  def test_personal_matrix_factorization_trains_keras_accuracy_threshold(self):
    num_users = 1
    num_items = 8
    num_latent_factors = 5
    personal_model = True
    add_biases = False
    l2_regularization = 0.0

    matrix_factorization_model = models.get_matrix_factorization_model(
        num_users,
        num_items,
        num_latent_factors,
        personal_model=personal_model,
        add_biases=add_biases,
        l2_regularization=l2_regularization)

    keras_model = models.build_keras_model(matrix_factorization_model)

    # Ensure number of parameters of model is as expected as a quick check.
    expected_num_params = (
        num_latent_factors  # User embeddings.
        + num_items * num_latent_factors)  # Item embeddings.
    num_params = keras_model.count_params()
    self.assertEqual(expected_num_params, num_params)
    num_trainable_params = count_trainable_params(keras_model)
    self.assertEqual(expected_num_params, num_trainable_params)

    history = keras_model.fit(
        self.train_items, self.train_preferences, batch_size=1, epochs=1)

    # Ensure the model has a valid loss after one epoch (not NaN).
    self.assertIn(_KERAS_LOSS, history.history)
    losses = history.history[_KERAS_LOSS]
    self.assertLen(losses, 1)
    self.assertFalse(np.isnan(losses[0]))

    # Ensure the model has valid number of examples after one epoch (not NaN).
    self.assertIn(_KERAS_NUM_EXAMPLES, history.history)
    num_examples = history.history[_KERAS_NUM_EXAMPLES]
    self.assertLen(num_examples, 1)
    self.assertEqual(num_examples[0], 5)

    # Ensure the model has valid number of batches after one epoch (not NaN).
    self.assertIn(_KERAS_NUM_BATCHES, history.history)
    num_batches = history.history[_KERAS_NUM_BATCHES]
    self.assertLen(num_batches, 1)
    self.assertEqual(num_batches[0], 5)

    # Ensure the model has a valid reconstruction accuracy after one epoch (not
    # NaN).
    self.assertIn(_KERAS_ACCURACY, history.history)
    accuracies = history.history[_KERAS_ACCURACY]
    self.assertLen(accuracies, 1)
    self.assertFalse(np.isnan(accuracies[0]))

  def test_build_reconstruction_model(self):
    num_users = 1
    num_items = 8
    num_latent_factors = 10
    personal_model = True
    add_biases = False
    l2_regularization = 0.0

    recon_model_fn = models.build_reconstruction_model(
        functools.partial(
            models.get_matrix_factorization_model,
            num_users,
            num_items,
            num_latent_factors,
            personal_model=personal_model,
            add_biases=add_biases,
            l2_regularization=l2_regularization))
    recon_model = recon_model_fn()

    # Check global/local trainable variables.
    local_trainable_variable_names = [
        var.name for var in recon_model.local_trainable_variables
    ]
    global_trainable_variable_names = [
        var.name for var in recon_model.global_trainable_variables
    ]

    self.assertEmpty(
        recon_model.local_non_trainable_variables,
        msg='Expected local_non_trainable_variables to be empty.')
    self.assertEmpty(
        recon_model.global_non_trainable_variables,
        msg='Expected global_non_trainable_variables to be empty.')

    expected_global_variable_names = ['ItemEmbedding/embeddings:0']
    self.assertSequenceEqual(global_trainable_variable_names,
                             expected_global_variable_names)

    expected_local_variable_names = ['UserEmbedding/UserEmbeddingKernel:0']
    self.assertSequenceEqual(local_trainable_variable_names,
                             expected_local_variable_names)

  def test_build_reconstruction_model_add_biases(self):
    num_users = 1
    num_items = 8
    num_latent_factors = 10
    personal_model = True
    add_biases = True
    l2_regularization = 0.0

    recon_model_fn = models.build_reconstruction_model(
        functools.partial(
            models.get_matrix_factorization_model,
            num_users,
            num_items,
            num_latent_factors,
            personal_model=personal_model,
            add_biases=add_biases,
            l2_regularization=l2_regularization))
    recon_model = recon_model_fn()

    # Check global/local trainable variables.
    local_trainable_variable_names = [
        var.name for var in recon_model.local_trainable_variables
    ]
    global_trainable_variable_names = [
        var.name for var in recon_model.global_trainable_variables
    ]

    self.assertEmpty(
        recon_model.local_non_trainable_variables,
        msg='Expected local_non_trainable_variables to be empty.')
    self.assertEmpty(
        recon_model.global_non_trainable_variables,
        msg='Expected global_non_trainable_variables to be empty.')

    expected_global_variable_names = [
        'ItemEmbedding/embeddings:0', 'ItemBias/embeddings:0',
        'GlobalBias/Bias:0'
    ]
    self.assertSequenceEqual(global_trainable_variable_names,
                             expected_global_variable_names)

    expected_local_variable_names = [
        'UserEmbedding/UserEmbeddingKernel:0', 'UserBias/UserEmbeddingKernel:0'
    ]
    self.assertSequenceEqual(local_trainable_variable_names,
                             expected_local_variable_names)

  def test_build_reconstruction_model_global_variables_only(self):
    num_users = 1
    num_items = 8
    num_latent_factors = 10
    personal_model = True
    add_biases = False
    l2_regularization = 0.0

    recon_model_fn = models.build_reconstruction_model(
        functools.partial(
            models.get_matrix_factorization_model,
            num_users,
            num_items,
            num_latent_factors,
            personal_model=personal_model,
            add_biases=add_biases,
            l2_regularization=l2_regularization),
        global_variables_only=True)
    recon_model = recon_model_fn()

    # Check global/local trainable variables.
    local_trainable_variable_names = [
        var.name for var in recon_model.local_trainable_variables
    ]
    global_trainable_variable_names = [
        var.name for var in recon_model.global_trainable_variables
    ]

    self.assertEmpty(
        recon_model.local_non_trainable_variables,
        msg='Expected local_non_trainable_variables to be empty.')
    self.assertEmpty(
        recon_model.global_non_trainable_variables,
        msg='Expected global_non_trainable_variables to be empty.')

    expected_global_variable_names = [
        'ItemEmbedding/embeddings:0', 'UserEmbedding/UserEmbeddingKernel:0'
    ]
    self.assertSequenceEqual(global_trainable_variable_names,
                             expected_global_variable_names)
    self.assertEmpty(
        local_trainable_variable_names,
        msg='Expected local_trainable_variables to be empty.')

  def test_build_reconstruction_model_add_biases_global_variables_only(self):
    num_users = 1
    num_items = 8
    num_latent_factors = 10
    personal_model = True
    add_biases = True
    l2_regularization = 0.0

    recon_model_fn = models.build_reconstruction_model(
        functools.partial(
            models.get_matrix_factorization_model,
            num_users,
            num_items,
            num_latent_factors,
            personal_model=personal_model,
            add_biases=add_biases,
            l2_regularization=l2_regularization),
        global_variables_only=True)
    recon_model = recon_model_fn()

    # Check global/local trainable variables.
    local_trainable_variable_names = [
        var.name for var in recon_model.local_trainable_variables
    ]
    global_trainable_variable_names = [
        var.name for var in recon_model.global_trainable_variables
    ]

    self.assertEmpty(
        recon_model.local_non_trainable_variables,
        msg='Expected local_non_trainable_variables to be empty.')
    self.assertEmpty(
        recon_model.global_non_trainable_variables,
        msg='Expected global_non_trainable_variables to be empty.')

    expected_global_variable_names = [
        'ItemEmbedding/embeddings:0', 'ItemBias/embeddings:0',
        'GlobalBias/Bias:0', 'UserEmbedding/UserEmbeddingKernel:0',
        'UserBias/UserEmbeddingKernel:0'
    ]
    self.assertSequenceEqual(global_trainable_variable_names,
                             expected_global_variable_names)
    self.assertEmpty(
        local_trainable_variable_names,
        msg='Expected local_trainable_variables to be empty.')

  def test_personal_matrix_factorization_trains_reconstruction_model(self):
    train_data = [
        self.train_users.flatten().tolist(),
        self.train_items.flatten().tolist(),
        self.train_preferences.flatten().tolist()
    ]
    train_tf_dataset = tf.data.Dataset.from_tensor_slices(
        list(zip(*train_data)))

    def batch_map_fn(example_batch):
      return collections.OrderedDict(
          x=tf.cast(example_batch[:, 0:1], tf.int64), y=example_batch[:, 1:2])

    train_tf_dataset = train_tf_dataset.batch(1).map(batch_map_fn).repeat(5)
    train_tf_datasets = [train_tf_dataset] * 2

    num_users = 1
    num_items = 8
    num_latent_factors = 10
    personal_model = True
    add_biases = False
    l2_regularization = 0.0

    tff_model_fn = models.build_reconstruction_model(
        functools.partial(
            models.get_matrix_factorization_model,
            num_users,
            num_items,
            num_latent_factors,
            personal_model=personal_model,
            add_biases=add_biases,
            l2_regularization=l2_regularization))

    # Also test `models.get_loss_fn` and `models.get_metrics_fn`.
    trainer = training_process.build_federated_reconstruction_process(
        tff_model_fn,
        loss_fn=models.get_loss_fn(),
        metrics_fn=models.get_metrics_fn(),
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1e-2),
        reconstruction_optimizer_fn=(
            lambda: tf.keras.optimizers.SGD(learning_rate=1e-3)),
        dataset_split_fn=reconstruction_utils.build_dataset_split_fn(
            recon_epochs_max=10))

    state = trainer.initialize()
    trainer.next(state, train_tf_datasets)

  def test_reconstruction_accuracy_metric(self):
    metric = models.ReconstructionAccuracyMetric(0.4)

    # Ensure weighting across batches with multiple calls to update_state
    # is accurate.
    y_true = tf.constant([[1.0], [2.0], [3.0], [4.0], [1.0], [2.0]],
                         dtype=tf.float32)
    y_pred = tf.constant([[0.61], [2.2], [3.41], [4.39], [9.0], [9.0]],
                         dtype=tf.float32)
    metric.update_state(y_true, y_pred)

    y_true = tf.constant([[1.0], [2.0], [3.0], [4.0], [5.1]], dtype=tf.float32)
    y_pred = tf.constant([[0.61], [2.2], [3.41], [5.0], [5.0]],
                         dtype=tf.float32)
    metric.update_state(y_true, y_pred)

    accuracy = metric.result()

    # 6 out of 11 predictions are within the threshold.
    expected_accuracy = 6 / 11

    tf.debugging.assert_near(expected_accuracy, accuracy)

  def test_reconstruction_accuracy_metric_weighted(self):
    metric = models.ReconstructionAccuracyMetric(0.4)

    # Ensure weighting across batches with multiple calls to update_state
    # is accurate.
    y_true = tf.constant([[1.0], [2.0], [3.0], [4.0], [1.0], [2.0]],
                         dtype=tf.float32)
    y_pred = tf.constant([[0.61], [2.2], [3.41], [4.39], [9.0], [9.0]],
                         dtype=tf.float32)
    sample_weight = tf.constant([1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
                                dtype=tf.float32)
    metric.update_state(y_true, y_pred, sample_weight=sample_weight)

    y_true = tf.constant([[1.0], [2.0], [3.0], [4.0], [5.1]], dtype=tf.float32)
    y_pred = tf.constant([[0.61], [2.2], [3.41], [5.0], [5.0]],
                         dtype=tf.float32)
    sample_weight = tf.constant([1.0, 1.0, 1.0, 1.0, 0.5], dtype=tf.float32)
    metric.update_state(y_true, y_pred, sample_weight=sample_weight)

    accuracy = metric.result()

    # 5.5 out of 9.5 weighted predictions are within the threshold.
    expected_accuracy = 5.5 / 9.5

    tf.debugging.assert_near(expected_accuracy, accuracy)

  def test_num_examples_counter(self):
    metric = models.NumExamplesCounter()

    y_true = tf.constant([[1.0], [2.0], [3.0], [4.0], [1.0], [2.0]],
                         dtype=tf.float32)
    y_pred = tf.constant([[0.61], [2.2], [3.41], [4.39], [9.0], [9.0]],
                         dtype=tf.float32)
    metric.update_state(y_true, y_pred)

    y_true = tf.constant([1.0, 2.0, 4.0], dtype=tf.float32)
    y_pred = tf.constant([.52, .42, .31], dtype=tf.float32)
    metric.update_state(y_true, y_pred)

    num_examples = metric.result()
    expected_num_examples = 9.0

    tf.debugging.assert_near(expected_num_examples, num_examples)

  def test_num_batches_counter(self):
    metric = models.NumBatchesCounter()

    y_true = tf.constant([[1.0], [2.0], [3.0], [4.0], [1.0], [2.0]],
                         dtype=tf.float32)
    y_pred = tf.constant([[0.61], [2.2], [3.41], [4.39], [9.0], [9.0]],
                         dtype=tf.float32)
    metric.update_state(y_true, y_pred)

    y_true = tf.constant([1.0, 2.0, 5.0], dtype=tf.float32)
    y_pred = tf.constant([.52, .42, .31], dtype=tf.float32)
    metric.update_state(y_true, y_pred)

    num_batches = metric.result()
    expected_num_batches = 2.0

    tf.debugging.assert_near(expected_num_batches, num_batches)


if __name__ == '__main__':
  absltest.main()
