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
"""Evaluation for `ReconstructionModel`s.

Since a trained `ReconstructionModel` consists of only global variables,
evaluation for models trained using Federated Reconstruction involves (1)
reconstructing local variables on client data and (2) evaluation of model global
variables and reconstructed local variables, computing loss and metrics.
Generally (1) and (2) should use disjoint parts of data for a given client.

`build_federated_reconstruction_evaluation`: builds a `tff.Computation` that
  performs (1) similarly to the Federated Reconstruction training algorithm and
  then (2) with the reconstructed local variables.

`build_federated_reconstruction_evaluation_process`: wraps
  `build_federated_reconstruction_evaluation` as a
  `tff.templates.IterativeProcess` that iteratively performs
  Federated Reconstruction evaluation across different clients for some number
  of rounds.
"""

import functools
from typing import Callable, List, Optional

import tensorflow as tf
import tensorflow_federated as tff

from reconstruction import keras_utils
from reconstruction import reconstruction_model
from reconstruction import reconstruction_utils

# Type aliases for readability.
LossFn = Callable[[], tf.keras.losses.Loss]
MetricsFn = Callable[[], List[tf.keras.metrics.Metric]]
ModelFn = Callable[[], reconstruction_model.ReconstructionModel]
OptimizerFn = Callable[[], tf.keras.optimizers.Optimizer]


def build_federated_reconstruction_evaluation(
    model_fn: ModelFn,
    *,  # Callers pass below args by name.
    loss_fn: LossFn,
    metrics_fn: Optional[MetricsFn],
    reconstruction_optimizer_fn: OptimizerFn = functools.partial(
        tf.keras.optimizers.SGD, 0.1),
    dataset_split_fn: Optional[reconstruction_utils.DatasetSplitFn] = None
) -> tff.Computation:
  """Builds a `tff.Computation` for evaluation of a `ReconstructionModel`.

  The returned computation proceeds in two stages: (1) reconstruction and (2)
  evaluation. During the reconstruction stage, local variables are reconstructed
  by freezing global variables and training using reconstruction_optimizer_fn.
  During the evaluation stage, the reconstructed local variables and global
  variables are evaluated using the provided loss_fn and metrics_fn.

  Usage of returned computation:
    eval_comp = build_federated_reconstruction_evaluation(...)
    metrics = eval_comp(reconstruction_utils.get_global_variables(model),
                        federated_data)

  Args:
    model_fn: A no-arg function that returns a `ReconstructionModel`. This
      method must *not* capture Tensorflow tensors or variables and use them.
      Must be constructed entirely from scratch on each invocation, returning
      the same pre-constructed model each call will result in an error.
    loss_fn: A no-arg function returning a `tf.keras.losses.Loss` to use to
      evaluate the model. The loss will be applied to the model's outputs during
      the evaluation stage. The final loss metric is the example-weighted mean
      loss across batches (and across clients).
    metrics_fn: A no-arg function returning a list of `tf.keras.metrics.Metric`s
      to evaluate the model. The metrics will be applied to the model's outputs
      during the evaluation stage. Final metric values are the example-weighted
      mean of metric values across batches (and across clients). If None, no
      metrics are applied.
    reconstruction_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer` used to reconstruct the local variables
      with the global ones frozen.
    dataset_split_fn: A `reconstruction_utils.DatasetSplitFn` taking in a client
      dataset and round number (always 0 for evaluation) and producing two TF
      datasets. The first is iterated over during reconstruction, and the second
      is iterated over during evaluation. This can be used to preprocess
      datasets to e.g. iterate over them for multiple epochs or use disjoint
      data for reconstruction and evaluation. If None, split client data in half
      for each user, using one half for reconstruction and the other for
      evaluation. See `reconstruction_utils.build_dataset_split_fn` for options.

  Raises:
    ValueError: if both `loss_fn` and `metrics_fn` are None.

  Returns:
    A `tff.Computation` that accepts model parameters and federated data and
    returns example-weighted evaluation loss and metrics.
  """
  # Construct the model first just to obtain the metadata and define all the
  # types needed to define the computations that follow.
  with tf.Graph().as_default():
    model = model_fn()
    global_weights = reconstruction_utils.get_global_variables(model)
    model_weights_type = tff.framework.type_from_tensors(global_weights)
    batch_type = tff.to_type(model.input_spec)
    metrics = [keras_utils.MeanLossMetric(loss_fn())]
    if metrics_fn is not None:
      metrics.extend(metrics_fn())
    if not metrics:
      raise ValueError(
          'One or both of metrics_fn and loss_fn should be provided.')
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
    """Reconstructs and evaluates with `incoming_model_weights`."""
    client_model = model_fn()
    client_global_weights = reconstruction_utils.get_global_variables(
        client_model)
    client_local_weights = reconstruction_utils.get_local_variables(
        client_model)
    metrics = [keras_utils.MeanLossMetric(loss_fn())]
    if metrics_fn is not None:
      metrics.extend(metrics_fn())
    batch_loss_fn = loss_fn()
    reconstruction_optimizer = reconstruction_optimizer_fn()

    @tf.function
    def reconstruction_reduce_fn(num_examples_sum, batch):
      """Runs reconstruction training on local client batch."""
      with tf.GradientTape() as tape:
        output = client_model.forward_pass(batch, training=True)
        batch_loss = batch_loss_fn(
            y_true=output.labels, y_pred=output.predictions)

      gradients = tape.gradient(batch_loss, client_local_weights.trainable)
      reconstruction_optimizer.apply_gradients(
          zip(gradients, client_local_weights.trainable))

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
      """Reconstructs and evaluates with `incoming_model_weights`."""
      # Pass in fixed 0 round number during evaluation, since global variables
      # aren't being iteratively updated as in training.
      recon_dataset, eval_dataset = dataset_split_fn(
          client_dataset, tf.constant(0, dtype=tf.int64))

      # Assign incoming global weights to `client_model` before reconstruction.
      tf.nest.map_structure(lambda v, t: v.assign(t), client_global_weights,
                            incoming_model_weights)

      recon_dataset.reduce(tf.constant(0), reconstruction_reduce_fn)
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


def build_federated_reconstruction_evaluation_process(
    model_fn: ModelFn,
    *,  # Callers pass below args by name.
    loss_fn: LossFn,
    metrics_fn: Optional[MetricsFn],
    reconstruction_optimizer_fn: OptimizerFn = functools.partial(
        tf.keras.optimizers.SGD, 0.1),
    dataset_split_fn: Optional[reconstruction_utils.DatasetSplitFn] = None
) -> tff.templates.IterativeProcess:
  """Builds an `IterativeProcess` for evaluation of `ReconstructionModel`s.

  The returned process wraps the `tff.Computation` returned by
  `build_federated_reconstruction_evaluation`, iteratively performing evaluation
  across clients for some number of rounds.

  Usage of returned process:
    eval_process = build_federated_reconstruction_evaluation_process(...)
    state = eval_process.initialize()
    state, metrics = eval_process(state, federated_data)

  Args:
    model_fn: A no-arg function that returns a `ReconstructionModel`. This
      method must *not* capture Tensorflow tensors or variables and use them.
      Must be constructed entirely from scratch on each invocation, returning
      the same pre-constructed model each call will result in an error.
    loss_fn: A no-arg function returning a `tf.keras.losses.Loss` to use to
      evaluate the model. The loss will be applied to the model's outputs during
      the evaluation stage. The final loss metric is the example-weighted mean
      loss across batches (and across clients).
    metrics_fn: A no-arg function returning a list of `tf.keras.metrics.Metric`s
      to evaluate the model. The metrics will be applied to the model's outputs
      during the evaluation stage. Final metric values are the example-weighted
      mean of metric values across batches (and across clients). If None, no
      metrics are applied.
    reconstruction_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer` used to reconstruct the local variables
      with the global ones frozen.
    dataset_split_fn: A `reconstruction_utils.DatasetSplitFn` taking in a client
      dataset and round number (always 0 for evaluation) and producing two TF
      datasets. The first is iterated over during reconstruction, and the second
      is iterated over during evaluation. This can be used to preprocess
      datasets to e.g. iterate over them for multiple epochs or use disjoint
      data for reconstruction and evaluation. If None, split client data in half
      for each user, using one half for reconstruction and the other for
      evaluation. See `reconstruction_utils.build_dataset_split_fn` for options.

  Returns:
    `tff.templates.IterativeProcess` constructed from the `tff.Computation`
    returned by `build_federated_reconstruction_evaluation`.
  """
  eval_comp = build_federated_reconstruction_evaluation(
      model_fn=model_fn,
      loss_fn=loss_fn,
      metrics_fn=metrics_fn,
      reconstruction_optimizer_fn=reconstruction_optimizer_fn,
      dataset_split_fn=dataset_split_fn)

  server_state_type = tff.type_at_server(
      reconstruction_utils.ServerState(
          model=eval_comp.type_signature.parameter[0].member,
          # There is no server optimizer in eval, so the optimizer_state is
          # empty.
          optimizer_state=(),
          round_num=tf.TensorSpec((), dtype=tf.int64),
          # Aggregations are stateless for evaluation.
          aggregator_state=(),
      ))
  batch_type = eval_comp.type_signature.parameter[1]

  @tff.tf_computation()
  def create_initial_state():
    return reconstruction_utils.ServerState(
        model=reconstruction_utils.get_global_variables(model_fn()),
        optimizer_state=(),
        round_num=tf.constant(0, dtype=tf.int64),
        aggregator_state=(),
    )

  @tff.federated_computation()
  def initialize():
    return tff.federated_value(create_initial_state(), tff.SERVER)

  @tff.federated_computation(server_state_type, batch_type)
  def eval_next(state, data):
    metrics = eval_comp(state.model, data)
    return state, metrics

  return tff.templates.IterativeProcess(initialize, eval_next)
