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
import os

from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_federated as tff

from dp_visual_embeddings import centralized_training_loop


FLAGS = flags.FLAGS


def create_dataset():
  x_list = [
      [1.0, 2.0],
      [3.0, 4.0],
  ]
  y_list = [
      [5.0],
      [6.0],
  ]

  def generate_examples():
    for x, y in zip(x_list, y_list):
      yield collections.OrderedDict(x=tf.constant(x), y=tf.constant(y))

  output_signature = collections.OrderedDict(
      x=tf.TensorSpec(shape=(2,), dtype=tf.float32),
      y=tf.TensorSpec(shape=(1,), dtype=tf.float32))
  # Create a dataset with 4 examples:
  dataset = tf.data.Dataset.from_generator(
      generate_examples, output_signature=output_signature)
  # Repeat the dataset 2 times with batches of 3 examples,
  # producing 3 minibatches (the last one with only 2 examples).
  return dataset.repeat(3).batch(2)


def create_sequential_model(input_dims=2):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(
          1,
          kernel_initializer='zeros',
          bias_initializer='zeros',
          input_shape=(input_dims,))
  ])


def mse_loss_fn():
  return tf.keras.losses.MeanSquaredError(
      reduction=tf.keras.losses.Reduction.NONE)


def _get_preprocess_spec():
  return tff.simulation.baselines.ClientSpec(
      num_epochs=1, batch_size=2, max_elements=3, shuffle_buffer_size=4)


class TestResultManager(tff.program.ReleaseManager):
  """TFF release manager that accumulates values for tests to assert on."""

  def __init__(self):
    super().__init__()
    self.history = collections.defaultdict(collections.OrderedDict)

  async def release(self, value, type_signature, key):
    del type_signature  # Unused.
    for metric_key, metric_value in value.items():
      self.history[metric_key][key] = metric_value

  def get_history_values(self, metric_key):
    return list(self.history[metric_key].values())


class CentralizedTrainingLoopTest(tf.test.TestCase):

  def assertMetricDecreases(self, metric, expected_len):
    self.assertLen(metric, expected_len)
    self.assertLess(metric[-1], metric[0])

  def assertMetricIncreases(self, metric, expected_len):
    self.assertLen(metric, expected_len)
    self.assertGreater(metric[-1], metric[0])

  def test_training_reduces_loss(self):
    metrics_fn = lambda: []
    optimizer_fn = lambda: tf.keras.optimizers.SGD(learning_rate=0.01)
    dataset = create_dataset()

    result_manager = TestResultManager()
    centralized_training_loop.run(
        model_fn=create_sequential_model,
        loss_fn=mse_loss_fn,
        metrics_fn=metrics_fn,
        optimizer_fn=optimizer_fn,
        train_dataset=dataset,
        validation_dataset=None,
        test_dataset=None,
        experiment_name='test_experiment',
        root_output_dir=self.get_temp_dir(),
        metrics_managers=[result_manager],
        per_replica_batch_size=2,
        num_steps=15,
        eval_interval_steps=3)

    self.assertMetricDecreases(
        result_manager.get_history_values('train/loss'), expected_len=15)

  def test_computes_metric(self):
    metrics_fn = lambda: [tf.keras.metrics.MeanSquaredError()]
    optimizer_fn = lambda: tf.keras.optimizers.SGD(learning_rate=0.01)
    dataset = create_dataset()

    result_manager = TestResultManager()
    centralized_training_loop.run(
        model_fn=create_sequential_model,
        loss_fn=mse_loss_fn,
        metrics_fn=metrics_fn,
        optimizer_fn=optimizer_fn,
        train_dataset=dataset,
        validation_dataset=dataset,
        test_dataset=dataset,
        experiment_name='test_experiment',
        root_output_dir=self.get_temp_dir(),
        metrics_managers=[result_manager],
        per_replica_batch_size=2,
        num_steps=15,
        eval_interval_steps=3)

    self.assertMetricDecreases(
        result_manager.get_history_values('train/loss'), expected_len=15)

    val_error = result_manager.get_history_values('val/mean_squared_error')
    test_error = result_manager.get_history_values('test/mean_squared_error')
    self.assertMetricDecreases(val_error, expected_len=5)
    self.assertLen(test_error, 1)
    # Because the validation and test datasets are the same, the metric should
    # have the same value on the last epoch.
    self.assertAlmostEqual(val_error[-1], test_error[0])

  def test_hparam_writing(self):
    metrics_fn = lambda: []
    optimizer_fn = lambda: tf.keras.optimizers.SGD(learning_rate=0.01)
    dataset = create_dataset()
    exp_name = 'write_hparams'
    temp_filepath = self.get_temp_dir()
    root_output_dir = temp_filepath

    hparams_dict = {
        'param1': 0,
        'param2': 5.02,
        'param3': 'sample',
        'param4': True
    }

    centralized_training_loop.run(
        model_fn=create_sequential_model,
        loss_fn=mse_loss_fn,
        metrics_fn=metrics_fn,
        optimizer_fn=optimizer_fn,
        train_dataset=dataset,
        validation_dataset=None,
        test_dataset=None,
        experiment_name=exp_name,
        root_output_dir=root_output_dir,
        metrics_managers=None,
        per_replica_batch_size=2,
        num_steps=1,
        hparams_dict=hparams_dict)

    self.assertTrue(tf.io.gfile.exists(root_output_dir))

    results_dir = os.path.join(root_output_dir, 'results', exp_name)
    self.assertTrue(tf.io.gfile.exists(results_dir))
    hparams_file = os.path.join(results_dir, 'hparams.csv')
    self.assertTrue(tf.io.gfile.exists(hparams_file))

  def test_writes_with_tensorboard_manager(self):
    """Integration test with TFF TensorBoard manager."""
    metrics_fn = lambda: []
    optimizer_fn = lambda: tf.keras.optimizers.SGD(learning_rate=0.01)
    dataset = create_dataset()
    output_dir = self.get_temp_dir()
    tensorboard_manager = tff.program.TensorBoardReleaseManager(
        summary_dir=output_dir)
    centralized_training_loop.run(
        model_fn=create_sequential_model,
        loss_fn=mse_loss_fn,
        metrics_fn=metrics_fn,
        optimizer_fn=optimizer_fn,
        train_dataset=dataset,
        validation_dataset=None,
        test_dataset=None,
        experiment_name='test_experiment',
        root_output_dir=output_dir,
        metrics_managers=[tensorboard_manager],
        per_replica_batch_size=2,
        num_steps=15,
        eval_interval_steps=3)

    events_files = tf.io.gfile.glob(os.path.join(output_dir, 'events*'))
    self.assertLen(events_files, 1)
    events_file = events_files[0]

    logging.info('Reading events from \"%s\"', events_file)
    events = []
    for event in tf.compat.v1.train.summary_iterator(events_file):
      step = event.step
      if not event.HasField('summary'):
        continue
      summary = event.summary
      for value in summary.value:
        # 'train/loss' is the only value being reported, we don't expect other
        # summaries to be present.
        self.assertEqual(value.tag, 'train/loss')
        self.assertTrue(value.HasField('tensor'))
        tensor = tf.io.parse_tensor(
            value.tensor.SerializeToString(), out_type=tf.float32)
        events.append((step, float(tensor)))

    events.sort()
    steps, train_loss = zip(*tuple(events))
    self.assertSequenceEqual(steps, range(1, 16))
    self.assertMetricDecreases(train_loss, expected_len=15)

  def test_resumes_from_checkpoint(self):
    metrics_fn = lambda: []
    optimizer_fn = lambda: tf.keras.optimizers.SGD(learning_rate=0.01)
    dataset = create_dataset()

    result_manager = TestResultManager()
    output_dir = self.get_temp_dir()
    centralized_training_loop.run(
        model_fn=create_sequential_model,
        loss_fn=mse_loss_fn,
        metrics_fn=metrics_fn,
        optimizer_fn=optimizer_fn,
        train_dataset=dataset,
        validation_dataset=None,
        test_dataset=None,
        experiment_name='test_experiment',
        root_output_dir=output_dir,
        metrics_managers=[result_manager],
        per_replica_batch_size=2,
        num_steps=3,
        checkpoint_interval_steps=1,
        eval_interval_steps=1)
    self.assertLen(result_manager.get_history_values('train/loss'), 3)

    centralized_training_loop.run(
        model_fn=create_sequential_model,
        loss_fn=mse_loss_fn,
        metrics_fn=metrics_fn,
        optimizer_fn=optimizer_fn,
        train_dataset=dataset,
        validation_dataset=None,
        test_dataset=None,
        experiment_name='test_experiment',
        root_output_dir=output_dir,
        metrics_managers=[result_manager],
        per_replica_batch_size=2,
        num_steps=5,
        checkpoint_interval_steps=1,
        eval_interval_steps=1)
    self.assertLen(result_manager.get_history_values('train/loss'), 5)


if __name__ == '__main__':
  # Since the test runs multiple training loops, in different test cases, it
  # creates different sets of variables within the _train_step, which isn't
  # allowed in tf.function. So we run the functions eagerly in this test, but
  # not in the actual binary, that only runs a single training loop.
  tf.config.run_functions_eagerly(True)
  tf.test.main()
