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

import itertools
import os.path

from absl.testing import parameterized
import pandas as pd
import tensorflow as tf

from generalization.utils import centralized_training_loop
from generalization.utils import metric_utils


def create_dataset():
  # Create a dataset with 4 examples:
  dataset = tf.data.Dataset.from_tensor_slices((
      [
          [1.0, 2.0],
          [3.0, 4.0],
      ],
      [
          [5.0],
          [6.0],
      ],
  ))
  # Repeat the dataset 2 times with batches of 3 examples,
  # producing 3 minibatches (the last one with only 2 examples).
  return dataset.repeat(3).batch(2)


def create_sequential_model(input_dims=2):
  dense_layer = tf.keras.layers.Dense(
      1,
      kernel_initializer='zeros',
      bias_initializer='zeros',
      input_shape=(input_dims,),
      name='dense')  # specify names to facilitate testing.
  return tf.keras.Sequential([dense_layer], name='sequential')


def compiled_keras_model(input_dims=2,
                         optimizer=tf.keras.optimizers.SGD(learning_rate=0.01)):
  model = create_sequential_model(input_dims)
  model.compile(
      loss=tf.keras.losses.MeanSquaredError(),
      optimizer=optimizer,
      metrics=[tf.keras.metrics.MeanSquaredError()])
  return model


class CentralizedTrainingLoopTest(tf.test.TestCase, parameterized.TestCase):

  def assertMetricDecreases(self, metric, expected_len):
    self.assertLen(metric, expected_len)
    self.assertLess(metric[-1], metric[0])

  @parameterized.named_parameters(
      ('train_train_eval={},train_val={},val={}'.format(*eval_fn_bools),
       *eval_fn_bools)
      for eval_fn_bools in itertools.product([False, True], repeat=3))
  def test_training_reduces_loss(self, use_part_train_eval_fn, use_part_val_fn,
                                 use_unpart_fn):
    keras_model = compiled_keras_model()
    dataset = create_dataset()
    eval_fn = lambda model: model.evaluate(dataset, return_dict=True, verbose=0)

    part_train_eval_fn = eval_fn if use_part_train_eval_fn else None
    part_val_fn = eval_fn if use_part_val_fn else None
    unpart_fn = eval_fn if use_unpart_fn else None

    history = centralized_training_loop.run(
        keras_model=keras_model,
        train_dataset=dataset,
        part_train_eval_fn=part_train_eval_fn,
        part_val_fn=part_val_fn,
        unpart_fn=unpart_fn,
        num_epochs=5)

    expected_metrics = ['loss', 'mean_squared_error',
                        'epoch_time_in_seconds']  # running training metrics

    for eval_fn, prefix in ((part_train_eval_fn,
                             metric_utils.PART_TRAIN_EVAL_METRICS_PREFIX),
                            (part_val_fn, metric_utils.PART_VAL_METRICS_PREFIX),
                            (unpart_fn, metric_utils.UNPART_METRICS_PREFIX)):
      if eval_fn is not None:
        for metric in ('loss', 'mean_squared_error'):
          prefixed_metric = prefix + metric
          self.assertIn(prefixed_metric, history.history.keys())
          self.assertMetricDecreases(
              history.history[prefixed_metric], expected_len=5)

          expected_metrics.append(prefixed_metric)
        expected_metrics.append(prefix + metric_utils.TIME_KEY)

    self.assertCountEqual(history.history.keys(), expected_metrics)

  def test_lr_callback(self):
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
    keras_model = compiled_keras_model(optimizer=optimizer)
    dataset = create_dataset()
    history = centralized_training_loop.run(
        keras_model=keras_model,
        train_dataset=dataset,
        num_epochs=10,
        decay_epochs=8,
        lr_decay=0.5)

    self.assertCountEqual(
        history.history.keys(),
        ['loss', 'mean_squared_error', 'epoch_time_in_seconds', 'lr'])
    self.assertAllClose(history.history['lr'], [0.1] * 8 + [0.05] * 2)


class CentralizedTrainingLoopWithDefaultCallbacksTest(tf.test.TestCase,
                                                      parameterized.TestCase):
  """Integrated test with `metric_utils.configure_default_callbacks()`."""

  def test_checkpoint_callback_can_restore(self):
    keras_model = compiled_keras_model()
    dataset = create_dataset()
    exp_name = 'test_ckpt'
    root_output_dir = self.get_temp_dir()

    checkpoiont_callback, _ = metric_utils.configure_default_callbacks(
        root_output_dir=root_output_dir,
        experiment_name=exp_name,
        epochs_per_checkpoint=1)

    centralized_training_loop.run(
        keras_model=keras_model,
        train_dataset=dataset,
        num_epochs=2,
        checkpoint_callback=checkpoiont_callback)

    self.assertTrue(tf.io.gfile.exists(root_output_dir))
    ckpt_dir = os.path.join(root_output_dir, 'checkpoints', exp_name)
    self.assertTrue(tf.io.gfile.exists(ckpt_dir))

    restored_model = compiled_keras_model()
    restored_model.load_weights(ckpt_dir)
    self.assertAllEqual(
        keras_model.get_config(),
        restored_model.get_config(),
    )

  @parameterized.named_parameters(
      ('train_train_eval={},train_val={},val={},test={}'.format(*eval_fn_bools),
       *eval_fn_bools)
      for eval_fn_bools in itertools.product([False, True], repeat=4))
  def test_writing_with_various_evaluation_combination_to_csv(
      self, use_part_train_eval_fn, use_part_val_fn, use_unpart_fn,
      use_test_fn):
    keras_model = compiled_keras_model()
    dataset = create_dataset()
    exp_name = 'write_eval_metrics'
    root_output_dir = self.get_temp_dir()
    eval_fn = lambda model: model.evaluate(dataset, return_dict=True, verbose=0)

    _, metrics_callbacks = metric_utils.configure_default_callbacks(
        root_output_dir=root_output_dir,
        experiment_name=exp_name,
        epochs_per_checkpoint=1)

    part_train_eval_fn = eval_fn if use_part_train_eval_fn else None
    part_val_fn = eval_fn if use_part_val_fn else None
    unpart_fn = eval_fn if use_unpart_fn else None
    test_fn = eval_fn if use_test_fn else None

    centralized_training_loop.run(
        keras_model=keras_model,
        train_dataset=dataset,
        num_epochs=2,
        part_train_eval_fn=part_train_eval_fn,
        part_val_fn=part_val_fn,
        unpart_fn=unpart_fn,
        test_fn=test_fn,
        metrics_callbacks=metrics_callbacks)

    log_dir = os.path.join(root_output_dir, 'logdir', exp_name)
    self.assertTrue(tf.io.gfile.exists(log_dir))

    results_dir = os.path.join(root_output_dir, 'results', exp_name)
    self.assertTrue(tf.io.gfile.exists(results_dir))
    metrics_file = os.path.join(results_dir, 'experiment.metrics.csv')
    self.assertTrue(tf.io.gfile.exists(metrics_file))
    metrics_csv = pd.read_csv(metrics_file, index_col=0)

    # Build expected columns.
    expected_columns = ['loss', 'mean_squared_error',
                        'epoch_time_in_seconds']  # running training metrics

    for eval_fn, prefix in ((part_train_eval_fn,
                             metric_utils.PART_TRAIN_EVAL_METRICS_PREFIX),
                            (part_val_fn, metric_utils.PART_VAL_METRICS_PREFIX),
                            (unpart_fn, metric_utils.UNPART_METRICS_PREFIX),
                            (test_fn, metric_utils.TEST_METRICS_PREFIX)):

      if eval_fn is not None:
        expected_columns.extend([
            prefix + metric
            for metric in ('loss', 'mean_squared_error', metric_utils.TIME_KEY)
        ])
    expected_num_rows = 2 if test_fn is None else 3
    self.assertEqual(metrics_csv.shape,
                     (expected_num_rows, len(expected_columns)))
    self.assertCountEqual(metrics_csv.columns, expected_columns)


if __name__ == '__main__':
  tf.test.main()
