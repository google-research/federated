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
"""Federated evaluation of a `ReconstructionModel`."""

import functools
from typing import Optional

import tensorflow as tf
import tensorflow_federated as tff

from reconstruction import evaluation_computation
from reconstruction import keras_utils
from reconstruction import reconstruction_utils

# Type aliases for readability.
LossFn = evaluation_computation.LossFn
MetricsFn = evaluation_computation.MetricsFn
ModelFn = evaluation_computation.ModelFn
OptimizerFn = evaluation_computation.OptimizerFn


def build_federated_finetune_evaluation(
    model_fn: ModelFn,
    *,  # Callers pass below args by name.
    loss_fn: LossFn,
    metrics_fn: Optional[MetricsFn] = None,
    finetune_optimizer_fn: OptimizerFn = functools.partial(
        tf.keras.optimizers.SGD, learning_rate=0.1),
    dataset_split_fn: Optional[reconstruction_utils.DatasetSplitFn] = None
) -> tff.Computation:
  """Builds a computation for evaluating a fully global `ReconstructionModel`.

  The input `model_fn` must return a `ReconstructionModel` that has only global
  variables. The returned computation proceeds in two stages on every client:
  (1) fine-tuning and (2) evaluation. During the fine-tuning stage, all global
  variables are fine-tuned on the first `tf.data.Dataset` returned by
  `dataset_split_fn` using finetune_optimizer_fn. During the evaluation stage,
  the fine-tuned model is evaluated on the second `tf.data.Dataset` returned by
  `dataset_split_fn`.

  Usage of returned computation:
    eval_comp = build_federated_finetune_evaluation(...)
    metrics = eval_comp(reconstruction_utils.get_global_variables(model),
                        federated_data)

  Args:
    model_fn: A no-arg function that returns a `ReconstructionModel`. The
      returned model must have only global variables. This method must *not*
      capture Tensorflow tensors or variables and use them. Must be constructed
      entirely from scratch on each invocation, returning the same model each
      call will result in an error.
    loss_fn: A no-arg function returning a `tf.keras.losses.Loss` to use to
      evaluate the model. The loss will be applied to the model's outputs during
      the evaluation stage. The final loss metric is the example-weighted mean
      loss across batches (and across clients).
    metrics_fn: A no-arg function returning a list of `tf.keras.metrics.Metric`s
      to evaluate the model. The metrics will be applied to the model's outputs
      during the evaluation stage. Final metric values are the example-weighted
      mean of metric values across batches (and across clients). If None, no
      metrics are applied.
    finetune_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer` used to fine-tune the global variables. A
      learning rate of zero means no fine-tuning.
    dataset_split_fn: A `reconstruction_utils.DatasetSplitFn` taking in a client
      dataset and round number (always 0 for evaluation) and producing two TF
      datasets. The first is iterated over during fine-tuning, and the second is
      iterated over during evaluation. If None, split client data in half for
      each user, using even-indexed entries for fine-tuning and odd-indexed
      entries for evaluation. See
      `federated_trainer_utils.build_dataset_split_fn` for options.

  Raises:
    ValueError: if `model_fn` returns a model with local variables.

  Returns:
    A `tff.Computation` that accepts model parameters and federated data and
    returns example-weighted evaluation loss and metrics.
  """
  # Construct the model first just to obtain the metadata and define all the
  # types needed to define the computations that follow.
  with tf.Graph().as_default():
    model = model_fn()
    global_weights = reconstruction_utils.get_global_variables(model)
    if not reconstruction_utils.has_only_global_variables(model):
      raise ValueError(
          '`model_fn` should return a model with only global variables.')
    model_weights_type = tff.framework.type_from_tensors(global_weights)
    batch_type = tff.to_type(model.input_spec)
    metrics = [keras_utils.MeanLossMetric(loss_fn())]
    if metrics_fn is not None:
      metrics.extend(metrics_fn())
    federated_output_computation = (
        keras_utils.federated_output_computation_from_metrics(metrics))
    # Remove unneeded variables to avoid polluting namespace.
    del model
    del global_weights
    del metrics

  if dataset_split_fn is None:
    dataset_split_fn = reconstruction_utils.build_dataset_split_fn(
        split_dataset=True)

  @tff.tf_computation(model_weights_type, tff.SequenceType(batch_type))
  def client_computation(incoming_model_weights, client_dataset):
    """Fine-tunes and evaluates with `incoming_model_weights`."""
    client_model = model_fn()
    client_global_weights = reconstruction_utils.get_global_variables(
        client_model)
    metrics = [keras_utils.MeanLossMetric(loss_fn())]
    if metrics_fn is not None:
      metrics.extend(metrics_fn())
    batch_loss_fn = loss_fn()
    finetune_optimizer = finetune_optimizer_fn()

    @tf.function
    def finetune_reduce_fn(num_examples_sum, batch):
      """Fine-tunes the model on local client batch."""
      with tf.GradientTape() as tape:
        output = client_model.forward_pass(batch, training=True)
        batch_loss = batch_loss_fn(
            y_true=output.labels, y_pred=output.predictions)

      gradients = tape.gradient(batch_loss, client_global_weights.trainable)
      finetune_optimizer.apply_gradients(
          zip(gradients, client_global_weights.trainable))

      return num_examples_sum + output.num_examples

    @tf.function
    def evaluation_reduce_fn(num_examples_sum, batch):
      """Runs evaluation on client batch without training."""
      output = client_model.forward_pass(batch, training=False)
      # Update each metric.
      for metric in metrics:
        metric.update_state(y_true=output.labels, y_pred=output.predictions)
      return num_examples_sum + output.num_examples

    @tf.function
    def tf_client_computation(incoming_model_weights, client_dataset):
      """Fine-tunes and evaluates with `incoming_model_weights`."""
      # Pass in fixed 0 round number during evaluation.
      finetune_dataset, eval_dataset = dataset_split_fn(
          client_dataset, tf.constant(0, dtype=tf.int64))

      # Assign incoming global weights to `client_model` before fine-tuning.
      tf.nest.map_structure(lambda v, t: v.assign(t), client_global_weights,
                            incoming_model_weights)

      finetune_dataset.reduce(tf.constant(0), finetune_reduce_fn)
      eval_dataset.reduce(tf.constant(0), evaluation_reduce_fn)

      eval_local_outputs = keras_utils.read_metric_variables(metrics)
      return eval_local_outputs

    return tf_client_computation(incoming_model_weights, client_dataset)

  @tff.federated_computation(
      tff.type_at_server(model_weights_type),
      tff.type_at_clients(tff.SequenceType(batch_type)))
  def server_eval(server_model_weights, federated_dataset):
    client_outputs = tff.federated_map(
        client_computation,
        [tff.federated_broadcast(server_model_weights), federated_dataset])
    return federated_output_computation(client_outputs)

  return server_eval
