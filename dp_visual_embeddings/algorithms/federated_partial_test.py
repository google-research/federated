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
import collections
import functools
from unittest import mock
from absl.testing import parameterized

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from dp_visual_embeddings.algorithms import federated_partial
from dp_visual_embeddings.models import keras_utils

_DATA_DIM = 2
_DATA_ELEMENT_SPEC = (tf.TensorSpec([None, _DATA_DIM], dtype=tf.float32),
                      tf.TensorSpec([None], dtype=tf.int32))


def _model_fn(hidden=5):
  global_layer = tf.keras.layers.Dense(
      hidden, kernel_initializer='ones', bias_initializer='zeros')
  client_layer = tf.keras.layers.Dense(
      2, kernel_initializer='ones', bias_initializer='zeros')
  inputs = tf.keras.Input(shape=(_DATA_DIM,))
  x = global_layer(inputs)
  outputs = client_layer(x)
  keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)

  global_variables = tff.learning.ModelWeights(
      trainable=global_layer.trainable_variables,
      non_trainable=global_layer.non_trainable_variables)
  client_variables = tff.learning.ModelWeights(
      trainable=client_layer.trainable_variables,
      non_trainable=client_layer.non_trainable_variables)
  tff_model = keras_utils.from_keras_model(
      keras_model,
      # The loss is not used for inference.
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      input_spec=_DATA_ELEMENT_SPEC,
      global_variables=global_variables,
      client_variables=client_variables)
  return tff_model


def _initial_weights(hidden=5):
  # The weights match the global variables of `_model_fn()`.
  return tff.learning.ModelWeights(
      trainable=[tf.ones((_DATA_DIM, hidden)),
                 tf.zeros((hidden))],
      non_trainable=[],
  )


def _create_dataset():
  # Create a dataset with 4 examples:
  dataset = tf.data.Dataset.from_tensor_slices(
      collections.OrderedDict(
          x=[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
          y=[[0.0], [0.0], [1.0], [1.0]]))
  # Repeat the dataset 2 times with batches of 3 examples,
  # producing 3 minibatches (the last one with only 2 examples).
  # Note that `batch` is required for this dataset to be useable,
  # as it adds the batch dimension which is expected by the model.
  return dataset.repeat(2).batch(3)


class ClientScheduledFedAvgTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters([
      ('keras_optimizer', lambda x: tf.keras.optimizers.SGD(), None),
      ('keras_optimizer_reconst', lambda x: tf.keras.optimizers.SGD(), 2),
  ])
  def test_construction_calls_model_fn(self, optimizer_fn, reconst_iters):
    # Assert that the process building does not call `model_fn` too many
    # times. `model_fn` can potentially be expensive (loading weights,
    # processing, etc).
    learning_rate_fn = lambda x: 0.1
    mock_model_fn = mock.Mock(side_effect=_model_fn)
    federated_partial.build_unweighted_averaging_with_optimizer_schedule(
        model_fn=mock_model_fn,
        client_learning_rate_fn=learning_rate_fn,
        client_optimizer_fn=optimizer_fn,
        reconst_iters=reconst_iters)
    self.assertEqual(mock_model_fn.call_count, 3)

  @parameterized.named_parameters([
      ('keras_optimizer', lambda x: tf.keras.optimizers.SGD(), None),
      ('keras_optimizer_reconst', lambda x: tf.keras.optimizers.SGD(), 2),
  ])
  def test_construction_calls_client_learning_rate_fn(self, optimizer_fn,
                                                      reconst_iters):
    mock_learning_rate_fn = mock.Mock(side_effect=lambda x: 1.0)
    optimizer_fn = tf.keras.optimizers.SGD
    federated_partial.build_unweighted_averaging_with_optimizer_schedule(
        model_fn=_model_fn,
        client_learning_rate_fn=mock_learning_rate_fn,
        client_optimizer_fn=optimizer_fn,
        reconst_iters=reconst_iters)

    self.assertEqual(mock_learning_rate_fn.call_count, 1)

  @parameterized.named_parameters([
      ('keras_optimizer', lambda x: tf.keras.optimizers.SGD(), None),
      ('keras_optimizer_reconst', lambda x: tf.keras.optimizers.SGD(), 2),
  ])
  def test_construction_calls_client_optimizer_fn(self, optimizer_fn,
                                                  reconst_iters):
    learning_rate_fn = lambda x: 0.5
    mock_optimizer_fn = mock.Mock(side_effect=optimizer_fn)
    federated_partial.build_unweighted_averaging_with_optimizer_schedule(
        model_fn=_model_fn,
        client_learning_rate_fn=learning_rate_fn,
        client_optimizer_fn=mock_optimizer_fn,
        head_lr_scale=0.1,
        reconst_iters=reconst_iters)

    self.assertEqual(mock_optimizer_fn.call_count, 2)
    self.assertEqual(mock_optimizer_fn.call_args_list[0][0][0], 0.5)
    self.assertEqual(mock_optimizer_fn.call_args_list[1][0][0], 0.05)

  def test_construction_calls_server_optimizer_fn(self):
    learning_rate_fn = lambda x: 0.5
    client_optimizer_fn = tf.keras.optimizers.SGD
    mock_server_optimizer_fn = mock.Mock(side_effect=tf.keras.optimizers.SGD)

    federated_partial.build_unweighted_averaging_with_optimizer_schedule(
        model_fn=_model_fn,
        client_learning_rate_fn=learning_rate_fn,
        client_optimizer_fn=client_optimizer_fn,
        server_optimizer_fn=mock_server_optimizer_fn)

    mock_server_optimizer_fn.assert_called()

  @parameterized.named_parameters([
      ('keras_optimizer', lambda x: tf.keras.optimizers.SGD(), None),
      ('keras_optimizer_reconst', lambda x: tf.keras.optimizers.SGD(), 2),
  ])
  def test_constructs_with_non_constant_learning_rate(self, optimizer_fn,
                                                      reconst_iters):

    def learning_rate_fn(round_num):
      return tf.cond(tf.less(round_num, 2), lambda: 0.1, lambda: 0.01)

    federated_partial.build_unweighted_averaging_with_optimizer_schedule(
        model_fn=_model_fn,
        client_learning_rate_fn=learning_rate_fn,
        client_optimizer_fn=optimizer_fn,
        reconst_iters=reconst_iters)

  def test_raises_on_non_callable_model_fn(self):
    with self.assertRaises(TypeError):
      federated_partial.build_unweighted_averaging_with_optimizer_schedule(
          model_fn=_model_fn(),
          client_learning_rate_fn=lambda x: 0.1,
          client_optimizer_fn=tf.keras.optimizers.SGD)


class TensorUtilsTest(tf.test.TestCase):

  def expect_ok_graph_mode(self, structure):
    with tf.Graph().as_default():
      result, error = (federated_partial._zero_all_if_any_non_finite(structure))
      with self.session() as sess:
        result, error = sess.run((result, error))
      try:
        tf.nest.map_structure(np.testing.assert_allclose, result, structure)
      except AssertionError:
        self.fail('Expected to get input {} back, but instead got {}'.format(
            structure, result))
      self.assertEqual(error, 0)

  def expect_ok_eager_mode(self, structure):
    result, error = (federated_partial._zero_all_if_any_non_finite(structure))
    try:
      tf.nest.map_structure(np.testing.assert_allclose, result, structure)
    except AssertionError:
      self.fail('Expected to get input {} back, but instead got {}'.format(
          structure, result))
    self.assertEqual(error, 0)

  def expect_zeros_graph_mode(self, structure, expected):
    with tf.Graph().as_default():
      result, error = (federated_partial._zero_all_if_any_non_finite(structure))
      with self.session() as sess:
        result, error = sess.run((result, error))
      try:
        tf.nest.map_structure(np.testing.assert_allclose, result, expected)
      except AssertionError:
        self.fail('Expected to get zeros, but instead got {}'.format(result))
      self.assertEqual(error, 1)

  def expect_zeros_eager_mode(self, structure, expected):
    result, error = (federated_partial._zero_all_if_any_non_finite(structure))
    try:
      tf.nest.map_structure(np.testing.assert_allclose, result, expected)
    except AssertionError:
      self.fail('Expected to get zeros, but instead got {}'.format(result))
    self.assertEqual(error, 1)

  def test_zero_all_if_any_non_finite_graph_mode(self):
    tf.config.experimental_run_functions_eagerly(False)
    self.expect_ok_graph_mode([])
    self.expect_ok_graph_mode([(), {}])
    self.expect_ok_graph_mode(1.1)
    self.expect_ok_graph_mode([1.0, 0.0])
    self.expect_ok_graph_mode([1.0, 2.0, {'a': 0.0, 'b': -3.0}])
    self.expect_zeros_graph_mode(np.inf, 0.0)
    self.expect_zeros_graph_mode((1.0, (2.0, np.nan)), (0.0, (0.0, 0.0)))
    self.expect_zeros_graph_mode((1.0, (2.0, {
        'a': 3.0,
        'b': [[np.inf], [np.nan]]
    })), (0.0, (0.0, {
        'a': 0.0,
        'b': [[0.0], [0.0]]
    })))

  def test_zero_all_if_any_non_finite_eager_mode(self):
    tf.config.experimental_run_functions_eagerly(True)
    self.expect_ok_eager_mode([])
    self.expect_ok_eager_mode([(), {}])
    self.expect_ok_eager_mode(1.1)
    self.expect_ok_eager_mode([1.0, 0.0])
    self.expect_ok_eager_mode([1.0, 2.0, {'a': 0.0, 'b': -3.0}])
    self.expect_zeros_eager_mode(np.inf, 0.0)
    self.expect_zeros_eager_mode((1.0, (2.0, np.nan)), (0.0, (0.0, 0.0)))
    self.expect_zeros_eager_mode((1.0, (2.0, {
        'a': 3.0,
        'b': [[np.inf], [np.nan]]
    })), (0.0, (0.0, {
        'a': 0.0,
        'b': [[0.0], [0.0]]
    })))


class ModelDeltaUpdateTest(tf.test.TestCase, parameterized.TestCase):

  def test_client_tf(self):
    optimizer_kwargs, expected_norm = {}, 0.01
    client_tf = federated_partial._build_model_delta_update_with_keras_optimizer(
        model_fn=_model_fn)
    gopt = tf.keras.optimizers.SGD(learning_rate=0.1, **optimizer_kwargs)
    lopt = tf.keras.optimizers.SGD(learning_rate=0.1, **optimizer_kwargs)
    dataset = _create_dataset()
    client_result, model_output = self.evaluate(
        client_tf(gopt, lopt, _initial_weights(), dataset))
    self.assertGreater(
        tf.linalg.global_norm(client_result.update), expected_norm)
    self.assertEqual(client_result.update_weight, 1.0)
    self.assertDictContainsSubset({
        'num_examples': [8],
    }, model_output)
    self.assertBetween(model_output['loss'][0], np.finfo(np.float32).eps, 10.0)

  def test_client_tf_reconst(self):
    optimizer_kwargs, expected_norm = {}, 0.01
    client_tf = federated_partial._build_model_delta_update_reconstruction(
        model_fn=_model_fn, reconst_iters=2)
    gopt = tf.keras.optimizers.SGD(learning_rate=0.1, **optimizer_kwargs)
    lopt = tf.keras.optimizers.SGD(learning_rate=0.1, **optimizer_kwargs)
    dataset = _create_dataset()
    client_result, model_output = self.evaluate(
        client_tf(gopt, lopt, _initial_weights(), dataset))
    self.assertGreater(
        tf.linalg.global_norm(client_result.update), expected_norm)
    self.assertEqual(client_result.update_weight, 1.0)
    self.assertDictContainsSubset({
        'num_examples': [8],
    }, model_output)
    self.assertBetween(model_output['loss'][0], np.finfo(np.float32).eps, 10.0)

  @parameterized.named_parameters(('only_head', 10, 0), ('only_backbone', 0, 0),
                                  ('1', 1, 5.22e-3), ('2', 2, 1.18e-2))
  def test_client_tf_reconst_iters(self, reconst_iters, expected_norm):
    hidden = 1
    client_tf = federated_partial._build_model_delta_update_reconstruction(
        model_fn=functools.partial(_model_fn, hidden=hidden),
        reconst_iters=reconst_iters)
    gopt = tf.keras.optimizers.SGD(learning_rate=0.1)
    lopt = tf.keras.optimizers.SGD(learning_rate=0.1)
    dataset = _create_dataset()
    client_result, _ = self.evaluate(
        client_tf(gopt, lopt, _initial_weights(hidden), dataset))
    self.assertNear(
        tf.linalg.global_norm(client_result.update),
        expected_norm,
        err=expected_norm * 5e-3)

  @parameterized.named_parameters(('_inf', np.inf), ('_nan', np.nan))
  def test_non_finite_aggregation(self, bad_value):
    client_tf = (
        federated_partial._build_model_delta_update_with_keras_optimizer(
            model_fn=_model_fn))
    gopt = tf.keras.optimizers.SGD(learning_rate=0.1)
    lopt = tf.keras.optimizers.SGD(learning_rate=0.1)
    dataset = _create_dataset()
    init_weights = _initial_weights()
    init_weights.trainable[0] = tf.constant(bad_value, shape=(_DATA_DIM, 5))
    client_outputs = client_tf(gopt, lopt, init_weights, dataset)
    self.assertEqual(self.evaluate(client_outputs[0].update_weight), 0.0)
    self.assertAllClose(
        self.evaluate(client_outputs[0].update),
        [tf.zeros((_DATA_DIM, 5)), tf.zeros((5))])


if __name__ == '__main__':
  tf.test.main()
