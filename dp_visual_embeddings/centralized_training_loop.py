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
"""Centralized training loops."""

import asyncio
from collections.abc import Callable, Mapping, Sequence
import os.path
from typing import Any, Optional, Union

from absl import logging
import tensorflow as tf
import tensorflow_federated as tff

from dp_visual_embeddings.tasks import task_utils
from dp_visual_embeddings.utils import export
from dp_visual_embeddings.utils import file_utils


def _check_positive(value: Union[int, float], name: str):
  if value <= 0:
    raise ValueError(f'Got {value} for {name}, expected positive value.')


@tf.function
def _train_step(keras_model: tf.keras.Model, loss: tf.keras.losses.Loss,
                optimizer: tf.keras.optimizers.Optimizer,
                global_batch_size: int, train_datum: Mapping[str,
                                                             Any]) -> tf.Tensor:
  """Applies a single step of training on one replica.

  Args:
    keras_model: The model to be trained.
    loss: The loss to be optimized over the model. Should have reduction set to
      NONE.
    optimizer: The optimizer to apply gradients.
    global_batch_size: The total number of examples, across all replicas used to
      compute the gradient.
    train_datum: A (per-replica) minibatch from the training dataset.

  Returns:
    The value of the loss on this replica's batch.
  """
  train_input, train_target = train_datum['x'], train_datum['y']
  with tf.GradientTape() as tape:
    model_outputs = keras_model(train_input, training=True)
    per_example_losses = loss(train_target, model_outputs)
    loss_value = tf.nn.compute_average_loss(
        per_example_losses, global_batch_size=global_batch_size)
  grads = tape.gradient(loss_value, keras_model.trainable_weights)
  optimizer.apply_gradients(zip(grads, keras_model.trainable_weights))
  return loss_value


@tf.function
def _distributed_train_step(strategy: tf.distribute.Strategy,
                            keras_model: tf.keras.Model,
                            loss: tf.keras.losses.Loss,
                            optimizer: tf.keras.optimizers.Optimizer,
                            global_batch_size: int,
                            train_datum: Mapping[str, Any]) -> tf.Tensor:
  """Applies a single step of training across all replicas.

  Args:
    strategy: Distribution strategy used to run training across the replicas.
    keras_model: The model to be trained.
    loss: The loss to be optimized over the model. Should have reduction set to
      NONE.
    optimizer: The optimizer to apply gradients.
    global_batch_size: The total number of examples, across all replicas used to
      compute the gradient.
    train_datum: A (multi-replica) minibatch from the training dataset.

  Returns:
    The total loss across all replicas.
  """
  kwargs = dict(
      keras_model=keras_model,
      loss=loss,
      optimizer=optimizer,
      global_batch_size=global_batch_size,
      train_datum=train_datum)
  per_replica_losses = strategy.run(_train_step, kwargs=kwargs)
  return strategy.reduce(
      tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


@tf.function
def _eval_step(keras_model: tf.keras.Model,
               metrics: Sequence[tf.keras.metrics.Metric], batch: Mapping[str,
                                                                          Any]):
  """Applies a single step of eval computation.

  Args:
    keras_model: The model being evaluated.
    metrics: The Eval metrics to compute.
    batch: A minibatch from the eval dataset.
  """
  model_input, target = batch['x'], batch['y']
  model_output = keras_model(model_input, training=False)
  for metric in metrics:
    metric.update_state(y_true=target, y_pred=model_output)


def _eval(keras_model: tf.keras.Model,
          metrics: Sequence[tf.keras.metrics.Metric], dataset: tf.data.Dataset,
          strategy: tf.distribute.Strategy,
          result_key_prefix: str) -> tuple[dict[str, float], tff.Type]:
  """Runs a custom eval loop.

  Args:
    keras_model: The model to evaluate.
    metrics: The set of metrics to compute for the evaluation.
    dataset: The data to evaluate on. Should have elements of the form
      `Ordereddict(x=..., y=...)`.
    strategy: A `tf.distribute.Strategy` dataset to parallelize the eval
      computations.
    result_key_prefix: Prefix appended to metric names in returned dictionary.

  Returns:
    A dictionary mapping the name of each metric to its result evaluated as a
    float scalar.
  """
  for metric in metrics:
    metric.reset_state()
  for batch in dataset:
    strategy.run(
        _eval_step,
        kwargs=dict(keras_model=keras_model, metrics=metrics, batch=batch))

  result_dict = {}
  for metric in metrics:
    key = result_key_prefix + '/' + metric.name
    result_dict[key] = float(metric.result())
  result_type = tff.StructType([(x, tf.float32) for x in result_dict.keys()])
  return result_dict, result_type


def run(
    model_fn: Callable[[], tf.keras.Model],
    loss_fn: Callable[[], tf.keras.losses.Loss],
    metrics_fn: Callable[[], Sequence[tf.keras.metrics.Metric]],
    optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer],
    train_dataset: tf.data.Dataset,
    validation_dataset: Optional[tf.data.Dataset],
    test_dataset: Optional[tf.data.Dataset],
    experiment_name: str,
    root_output_dir: str,
    metrics_managers: Optional[Sequence[tff.program.ReleaseManager]],
    per_replica_batch_size: int,
    num_steps: int,
    checkpoint_interval_steps: int = 100,
    eval_interval_steps: int = 100,
    inference_model_fn: Optional[Callable[[], tf.keras.Model]] = None,
    export_interval_steps: int = 10000,
    hparams_dict: Optional[dict[str, Any]] = None,
) -> tf.keras.Model:
  """Run centralized training for a given compiled `tf.keras.Model`.

  Args:
    model_fn: A callable that builds model to train.
    loss_fn: A loss on the model outputs and training labels.
    metrics_fn: A callable returning keras metrics to evaluate the model with.
    optimizer_fn: A callable to build the object to optimize loss over the model
      variables.
    train_dataset: The `tf.data.Dataset` to be used for training. It is expected
      to yield elements of the form `Ordereddict(x=..., y=...)`, where the key
      'x' is mapped to model inputs and the key 'y' is mapped to the label.
    validation_dataset: Dataset for validation, of the same format as
      `train_dataset`. Metrics are computed on this set every epoch.
    test_dataset: Dataset for test, of the same format as `train_dataset`.
      Metrics are computed after training.
    experiment_name: Name of the experiment, used as part of the name of the
      output directory.
    root_output_dir: The top-level output directory. The directory
      `root_output_dir/experiment_name` will contain CSVs and other outputs.
    metrics_managers: TFF release managers to report results.
    per_replica_batch_size: The number of examples used in a train step on each
      replica.
    num_steps: How many training steps to perform.
    checkpoint_interval_steps: The frequency to checkpoint model weights.
    eval_interval_steps: The interval at which to evaluate on the validation
      set.
    inference_model_fn: Callable to build the inference-time model, to be
      exported as a `SavedModel`. Will use `model_fn` if this is None.
    export_interval_steps: The interval at which to write a `SavedModel` to the
      export directory.
    hparams_dict: An optional dict specifying hyperparameters. If provided, the
      hyperparameters will be written to CSV.

  Returns:
    The trained model.
  """
  _check_positive(num_steps, 'num_steps')
  _check_positive(eval_interval_steps, 'eval_interval_steps')
  inference_model_fn = inference_model_fn or model_fn

  strategy = tf.distribute.MirroredStrategy()
  global_batch_size = per_replica_batch_size * strategy.num_replicas_in_sync
  logging.info('Global batch size: %d', global_batch_size)

  # Build the model and associated objects.
  with strategy.scope():
    keras_model = model_fn()
    inference_model = inference_model_fn()
    optimizer = optimizer_fn()
    loss = loss_fn()
    metrics = metrics_fn()

  loop = asyncio.get_event_loop()

  checkpoints_dir = os.path.join(root_output_dir, 'checkpoints',
                                 experiment_name)
  results_dir = os.path.join(root_output_dir, 'results', experiment_name)

  for path in [root_output_dir, results_dir]:
    tf.io.gfile.makedirs(path)

  if hparams_dict:
    hparams_file = os.path.join(results_dir, 'hparams.csv')
    logging.info('Saving hyper parameters to: [%s]', hparams_file)
    file_utils.atomic_write_series_to_csv(hparams_dict, hparams_file)

  checkpoint = tf.train.Checkpoint(
      optimizer=optimizer, model=keras_model)
  # Uses optimizer.iterations as a step counter.
  # Should use a counter variable if doing multiple-replica training.
  checkpoint_manager = tf.train.CheckpointManager(
      checkpoint,
      directory=checkpoints_dir,
      max_to_keep=5,
      step_counter=optimizer.iterations,
      checkpoint_interval=checkpoint_interval_steps)
  checkpoint_manager.restore_or_initialize()

  step = int(optimizer.iterations.read_value())
  logging.info('Starting at step %d', step)

  dist_train_dataset = strategy.experimental_distribute_dataset(
      train_dataset.repeat())
  if validation_dataset is not None:
    dist_validation_dataset = strategy.experimental_distribute_dataset(
        validation_dataset)
  if test_dataset is not None:
    dist_test_dataset = strategy.experimental_distribute_dataset(test_dataset)

  for train_datum in dist_train_dataset:
    loss_value = _distributed_train_step(
        strategy=strategy,
        keras_model=keras_model,
        loss=loss,
        optimizer=optimizer,
        global_batch_size=global_batch_size,
        train_datum=train_datum)

    step += 1
    checkpoint_manager.save(check_interval=True)

    if metrics_managers is not None:
      train_step_result = {'train/loss': float(loss_value)}
      train_result_type = tff.StructType([('train/loss', tf.float32)])
      loop.run_until_complete(
          asyncio.gather(*[
              m.release(train_step_result, train_result_type, step)
              for m in metrics_managers
          ]))

    if validation_dataset is not None and step % eval_interval_steps == 0:
      val_results, val_type = _eval(keras_model, metrics,
                                    dist_validation_dataset, strategy, 'val')
      if metrics_managers is not None:
        loop.run_until_complete(
            asyncio.gather(*[
                m.release(val_results, val_type, step) for m in metrics_managers
            ]))

    if step % export_interval_steps == 0:
      export_dir = os.path.join(root_output_dir, 'export', experiment_name,
                                'inference_%06d' % step)
      export.export_keras_model(
          train_model=keras_model,
          export_model=inference_model,
          export_dir=export_dir)

    if step >= num_steps:
      break

  # Evaluate on test dataset only once, at the end of training.
  if test_dataset is not None:
    test_results, test_results_type = _eval(keras_model, metrics,
                                            dist_test_dataset, strategy, 'test')
    if metrics_managers is not None:
      loop.run_until_complete(
          asyncio.gather(*[
              m.release(test_results, test_results_type, step)
              for m in metrics_managers
          ]))

  if num_steps % export_interval_steps != 0:
    export_dir = os.path.join(root_output_dir, 'export', experiment_name,
                              'inference_%06d' % num_steps)
    export.export_keras_model(
        train_model=keras_model,
        export_model=inference_model,
        export_dir=export_dir)

  return keras_model


def run_on_task(
    task: task_utils.EmbeddingTask,
    experiment_name: str,
    root_output_dir: str,
    metrics_managers: Optional[Sequence[tff.program.ReleaseManager]],
    per_replica_batch_size: int,
    num_steps: int,
    checkpoint_interval_steps: int = 100,
    eval_interval_steps: int = 100,
    export_interval_steps: int = 10000,
    hparams_dict: Optional[dict[str, Any]] = None,
    initial_lr: float = 0.01):
  """Run centralized training for a EmbeddingTask object.

  Args:
    task: The task that defines the model and dataset.
    experiment_name: Name of the experiment, used as part of the name of the
      output directory.
    root_output_dir: The top-level output directory. The directory
      `root_output_dir/experiment_name` will contain CSVs and other outputs.
    metrics_managers: TFF release managers to report results.
    per_replica_batch_size: The number of examples used in a train step on each
      replica.
    num_steps: How many training steps to perform.
    checkpoint_interval_steps: The frequency to checkpoint model weights.
    eval_interval_steps: The interval at which to evaluate on the validation
      set.
    export_interval_steps: The interval at which to write a `SavedModel` to the
      export directory.
    hparams_dict: An optional dict specifying hyperparameters. If provided, the
      hyperparameters will be written to CSV.
    initial_lr: Starting learning rate to initialize the optimizer.
  """

  def loss_fn():
    return task.get_loss(reduction=tf.keras.losses.Reduction.NONE)

  def optimizer_fn():
    return tf.keras.optimizers.SGD(learning_rate=initial_lr, momentum=0.9)

  def inference_model_fn():
    return task.inference_model

  train_dataset = task.datasets.get_centralized_train_data()
  validation_dataset = task.datasets.get_centralized_validation_data()
  test_dataset = task.datasets.get_centralized_test_data()

  run(model_fn=task.keras_model_fn,
      loss_fn=loss_fn,
      metrics_fn=task.get_metrics,
      optimizer_fn=optimizer_fn,
      train_dataset=train_dataset,
      validation_dataset=validation_dataset,
      test_dataset=test_dataset,
      experiment_name=experiment_name,
      root_output_dir=root_output_dir,
      metrics_managers=metrics_managers,
      per_replica_batch_size=per_replica_batch_size,
      num_steps=num_steps,
      checkpoint_interval_steps=checkpoint_interval_steps,
      eval_interval_steps=eval_interval_steps,
      inference_model_fn=inference_model_fn,
      export_interval_steps=export_interval_steps,
      hparams_dict=hparams_dict)
