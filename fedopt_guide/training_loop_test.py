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

import collections
import os

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from fedopt_guide import training_loop

_Batch = collections.namedtuple('Batch', ['x', 'y'])

_COMPATIBILITY_ERROR_MESSAGE = (
    'The iterative_process argument must be of '
    'type`tff.templates.IterativeProcess`, and must have an '
    'attribute `get_model_weights`, which must be a `tff.Computation`. This '
    'computation must accept as input the state of `iterative_process`, and '
    'its output must be a nested structure of tensors matching the expected '
    'shape of the first input argument to `evaluation_fn`.')


def _keras_model_builder():
  # Create a simple linear regression model, single output.
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(
          1,
          kernel_initializer='zeros',
          bias_initializer='zeros',
          input_shape=(1,))
  ])
  return model


def _compiled_keras_model_builder():
  model = _keras_model_builder()
  model.compile(loss=tf.keras.losses.MeanSquaredError())
  return model


def _create_tf_dataset_for_client(client_id, batch_data=True):
  # Create client data for y = 2*x+3
  np.random.seed(client_id)
  x = np.random.rand(6, 1).astype(np.float32)
  y = 2 * x + 3
  dataset = tf.data.Dataset.from_tensor_slices(
      collections.OrderedDict([('x', x), ('y', y)]))
  if batch_data:
    dataset = dataset.batch(2)
  return dataset


def _get_input_spec():
  return _create_tf_dataset_for_client(0).element_spec


def _tff_model_fn():
  return tff.learning.from_keras_model(
      keras_model=_keras_model_builder(),
      input_spec=_get_input_spec(),
      loss=tf.keras.losses.MeanSquaredError(),
      metrics=[tf.keras.metrics.MeanSquaredError()])


def _build_federated_averaging_process():
  return tff.learning.build_federated_averaging_process(
      _tff_model_fn,
      client_optimizer_fn=tf.keras.optimizers.SGD,
      server_optimizer_fn=tf.keras.optimizers.SGD)


def _evaluation_fn():
  fed_eval_fn = tff.learning.build_federated_evaluation(_tff_model_fn)
  fed_data = _federated_data()

  def eval_fn(model, round_num):
    del round_num
    return fed_eval_fn(model, fed_data)

  return eval_fn


def _federated_data():
  return [_create_tf_dataset_for_client(1), _create_tf_dataset_for_client(2)]


class TrainingLoopArgumentsTest(tf.test.TestCase):

  def test_raises_non_iterative_process(self):
    bad_iterative_process = _build_federated_averaging_process().next

    def client_datasets_fn(round_num):
      del round_num
      return _federated_data()

    def evaluation_fn(model, round_num):
      del model, round_num
      return {}

    root_output_dir = self.get_temp_dir()
    with self.assertRaises(TypeError):
      training_loop.run(
          iterative_process=[bad_iterative_process],
          train_client_datasets_fn=client_datasets_fn,
          evaluation_fn=evaluation_fn,
          total_rounds=10,
          experiment_name='non_iterative_process',
          root_output_dir=root_output_dir)

  def test_raises_non_callable_train_client_dataset(self):
    iterative_process = _build_federated_averaging_process()
    client_dataset = _create_tf_dataset_for_client(3)

    def evaluation_fn(model, round_num):
      del model, round_num
      return {}

    root_output_dir = self.get_temp_dir()
    with self.assertRaises(TypeError):
      training_loop.run(
          iterative_process=iterative_process,
          train_client_datasets_fn=client_dataset,
          evaluation_fn=evaluation_fn,
          total_rounds=10,
          experiment_name='non_callable_client_dataset',
          root_output_dir=root_output_dir)

  def test_raises_non_callable_evaluate_fn(self):
    iterative_process = _build_federated_averaging_process()

    def client_datasets_fn(round_num):
      del round_num
      return _federated_data()

    metrics_dict = {}
    root_output_dir = self.get_temp_dir()
    with self.assertRaises(TypeError):
      training_loop.run(
          iterative_process=iterative_process,
          train_client_datasets_fn=client_datasets_fn,
          evaluation_fn=metrics_dict,
          total_rounds=10,
          experiment_name='non_callable_evaluate',
          root_output_dir=root_output_dir)

  def test_raises_non_str_output_dir(self):
    iterative_process = _build_federated_averaging_process()

    def client_datasets_fn(round_num):
      del round_num
      return _federated_data()

    def evaluation_fn(model, round_num):
      del model, round_num
      return {}

    with self.assertRaises(TypeError):
      training_loop.run(
          iterative_process=iterative_process,
          train_client_datasets_fn=client_datasets_fn,
          evaluation_fn=evaluation_fn,
          total_rounds=10,
          experiment_name='non_str_output_dir',
          root_output_dir=1)

  def test_raises_no_get_model_weights_attribute_in_state(self):

    class BadIterativeProcess(tff.templates.IterativeProcess):

      def __init__(self):
        pass

      def initialize(self):
        return {}

      def next(self, state, data):
        return {}

    iterative_process = BadIterativeProcess()

    def client_datasets_fn(round_num):
      del round_num
      return _federated_data()

    def evaluation_fn(model, round_num):
      del model, round_num
      return {}

    with self.assertRaisesRegex(
        training_loop.IterativeProcessCompatibilityError,
        _COMPATIBILITY_ERROR_MESSAGE):
      training_loop.run(
          iterative_process=iterative_process,
          train_client_datasets_fn=client_datasets_fn,
          evaluation_fn=evaluation_fn,
          total_rounds=10,
          experiment_name='bad_iterative_process')

  def test_raises_non_callable_get_model_weights_attribute(self):

    class BadIterativeProcess(tff.templates.IterativeProcess):

      def __init__(self):
        pass

      def initialize(self):
        return {}

      def next(self, state, data):
        return {}

    iterative_process = BadIterativeProcess()
    iterative_process.get_model_weights = 2

    def client_datasets_fn(round_num):
      del round_num
      return _federated_data()

    def evaluation_fn(model, round_num):
      del model, round_num
      return {}

    with self.assertRaisesRegex(
        training_loop.IterativeProcessCompatibilityError,
        _COMPATIBILITY_ERROR_MESSAGE):
      training_loop.run(
          iterative_process=iterative_process,
          train_client_datasets_fn=client_datasets_fn,
          evaluation_fn=evaluation_fn,
          total_rounds=10,
          experiment_name='bad_iterative_process')

  def test_raises_non_tff_computation_get_model_weights_attribute(self):

    class BadIterativeProcess(tff.templates.IterativeProcess):

      def __init__(self):
        pass

      def initialize(self):
        return {}

      def next(self, state, data):
        return {}

      def get_model_weights(self, state):
        return {}

    iterative_process = BadIterativeProcess()

    def client_datasets_fn(round_num):
      del round_num
      return _federated_data()

    def evaluation_fn(model, round_num):
      del model, round_num
      return {}

    with self.assertRaisesRegex(
        training_loop.IterativeProcessCompatibilityError,
        _COMPATIBILITY_ERROR_MESSAGE):

      training_loop.run(
          iterative_process=iterative_process,
          train_client_datasets_fn=client_datasets_fn,
          evaluation_fn=evaluation_fn,
          total_rounds=10,
          experiment_name='bad_iterative_process')


class ExperimentRunnerTest(tf.test.TestCase):

  def test_fedavg_training_decreases_loss(self):
    iterative_process = _build_federated_averaging_process()

    def client_datasets_fn(round_num):
      del round_num
      return _federated_data()

    def test_fn(model):
      keras_model = _compiled_keras_model_builder()
      model.assign_weights_to(keras_model)
      batch = next(iter(_create_tf_dataset_for_client(5)))
      return {'loss': keras_model.evaluate(batch['x'], batch['y'])}

    initial_state = iterative_process.initialize()
    initial_model = iterative_process.get_model_weights(initial_state)

    root_output_dir = self.get_temp_dir()
    final_state = training_loop.run(
        iterative_process=iterative_process,
        train_client_datasets_fn=client_datasets_fn,
        evaluation_fn=_evaluation_fn(),
        total_rounds=1,
        experiment_name='fedavg_decreases_loss',
        root_output_dir=root_output_dir)
    final_model = iterative_process.get_model_weights(final_state)
    self.assertLess(
        test_fn(final_model)['loss'],
        test_fn(initial_model)['loss'])

  def test_checkpoint_manager_saves_state(self):
    experiment_name = 'checkpoint_manager_saves_state'
    iterative_process = _build_federated_averaging_process()

    def client_datasets_fn(round_num):
      del round_num
      return _federated_data()

    def evaluation_fn(model, round_num):
      del model, round_num
      return {}

    root_output_dir = self.get_temp_dir()
    final_state = training_loop.run(
        iterative_process=iterative_process,
        train_client_datasets_fn=client_datasets_fn,
        evaluation_fn=evaluation_fn,
        total_rounds=1,
        experiment_name=experiment_name,
        root_output_dir=root_output_dir)
    final_model = iterative_process.get_model_weights(final_state)

    ckpt_manager = tff.simulation.FileCheckpointManager(
        os.path.join(root_output_dir, 'checkpoints', experiment_name))
    restored_state, restored_round = ckpt_manager.load_latest_checkpoint(
        final_state)

    self.assertEqual(restored_round, 0)

    keras_model = _compiled_keras_model_builder()
    restored_model = iterative_process.get_model_weights(restored_state)

    restored_model.assign_weights_to(keras_model)
    batch = next(iter(_create_tf_dataset_for_client(5)))
    restored_loss = keras_model.test_on_batch(batch['x'], batch['y'])

    final_model.assign_weights_to(keras_model)
    final_loss = keras_model.test_on_batch(batch['x'], batch['y'])
    self.assertEqual(final_loss, restored_loss)

  def test_fn_writes_metrics(self):
    experiment_name = 'test_metrics'
    iterative_process = _build_federated_averaging_process()

    def client_datasets_fn(round_num):
      del round_num
      return _federated_data()

    def test_fn(model):
      keras_model = _compiled_keras_model_builder()
      model.assign_weights_to(keras_model)
      batch = next(iter(_create_tf_dataset_for_client(5)))
      return {'loss': keras_model.evaluate(batch['x'], batch['y'])}

    root_output_dir = self.get_temp_dir()
    training_loop.run(
        iterative_process=iterative_process,
        train_client_datasets_fn=client_datasets_fn,
        evaluation_fn=_evaluation_fn(),
        total_rounds=1,
        experiment_name=experiment_name,
        root_output_dir=root_output_dir,
        rounds_per_eval=10,
        test_fn=test_fn)

    csv_file = os.path.join(root_output_dir, 'results', experiment_name,
                            'experiment.metrics.csv')
    metrics_manager = tff.simulation.CSVMetricsManager(csv_file)
    fieldnames, metrics = metrics_manager.get_metrics()
    self.assertLen(metrics, 2)
    self.assertIn('test/loss', fieldnames)


if __name__ == '__main__':
  tf.test.main()
