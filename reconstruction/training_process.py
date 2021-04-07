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
"""An implementation of Federated Reconstruction (FedRecon).

This is a modification of the standard Federated Averaging algorithm, designed
for `ReconstructionModel`s. `ReconstructionModel`s introduce a partition of
variables into global variables and local variables.

At a high level, local variables are reconstructed (via training) on client
devices at the beginning of each round and never sent to the server. Each
client's local variables are then used to update global variables. Global
variable deltas are aggregated normally on the server as in Federated Averaging
and sent to new clients at the beginning of the next round.

During each round:
1. A random subset of clients is selected.
2. Each client receives the latest global variables from the server.
3. Each client locally reconstructs its local variables.
4. Each client computes an update for the global variables.
5. The server aggregates global variables across users and updates them for the
   next round.

Note that clients are stateless since the local variables are not stored across
rounds.

Variations of this general approach enabled by this implementation include:
jointly training global and local variables in step (4), using different
optimizers/learning rates for different steps, doing multiple epochs over data
in steps (3) and (4), limiting the data used in each step, skipping step (3)
entirely, and others.

Original Federated Averaging algorithm is based on the paper:
Communication-Efficient Learning of Deep Networks from Decentralized Data
    H. Brendan McMahan, Eider Moore, Daniel Ramage,
    Seth Hampson, Blaise Aguera y Arcas. AISTATS 2017.
    https://arxiv.org/abs/1602.05629
"""

import functools
from typing import Callable, List, Optional

from absl import logging
import tensorflow as tf
import tensorflow_federated as tff

from reconstruction import keras_utils
from reconstruction import reconstruction_model
from reconstruction import reconstruction_utils
from utils import tensor_utils

# Type aliases for readability.
ClientWeightFn = Callable[..., float]
LossFn = Callable[[], tf.keras.losses.Loss]
MetricsFn = Callable[[], List[tf.keras.metrics.Metric]]
ModelFn = Callable[[], reconstruction_model.ReconstructionModel]
OptimizerFn = Callable[[], tf.keras.optimizers.Optimizer]
TFComputationFn = Callable[..., tff.tf_computation]


def build_server_init_fn(
    model_fn: ModelFn,
    server_optimizer_fn: OptimizerFn,
    aggregation_process: tff.templates.AggregationProcess,
) -> tff.Computation:
  """Builds a `tff.tf_computation` that returns an initial `ServerState`.

  Args:
    model_fn: A no-arg function that returns a `ReconstructionModel`.
    server_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer`.
    aggregation_process: Instance of `tff.templates.AggregationProcess` which
      will perform aggregation from clients to server.

  Returns:
    A `tff.Computation` that returns an initial `ServerState`.
  """

  @tff.tf_computation
  def server_init_tf():
    """Initialize the TensorFlow-only portions of the server state."""
    # Round number can be used to parameterize client behavior, e.g. clients
    # can do more local iterations for reconstruction for later rounds.
    round_num = tf.constant(1, dtype=tf.int64)
    model = model_fn()
    server_optimizer = server_optimizer_fn()
    # Create optimizer variables so we have a place to assign the optimizer's
    # state.
    server_optimizer_vars = reconstruction_utils.create_optimizer_vars(
        model, server_optimizer)
    return reconstruction_utils.get_global_variables(
        model), server_optimizer_vars, round_num

  @tff.federated_computation()
  def server_init_tff():
    """Returns a `reconstruction_utils.ServerState` placed at `tff.SERVER`."""
    tf_init_tuple = tff.federated_eval(server_init_tf, tff.SERVER)
    aggregation_process_init = aggregation_process.initialize()
    return tff.federated_zip(
        reconstruction_utils.ServerState(
            model=tf_init_tuple[0],
            optimizer_state=tf_init_tuple[1],
            round_num=tf_init_tuple[2],
            aggregator_state=aggregation_process_init))

  return server_init_tff


def build_server_update_fn(
    model_fn: ModelFn, server_optimizer_fn: OptimizerFn,
    server_state_type: tff.Type, model_weights_type: tff.Type,
    aggregator_state_type: tff.Type) -> tff.tf_computation:
  """Builds a `tff.tf_computation` that updates `ServerState`.

  Args:
    model_fn: A no-arg function that returns a `ReconstructionModel`.
    server_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer`.
    server_state_type: type_signature of server state.
    model_weights_type: type_signature of model weights.
    aggregator_state_type: type signature of the `state` element of the
      `tff.templates.AggregationProcess` used to perform aggregation.

  Returns:
    A `tff.tf_computation` that updates `ServerState`.
  """

  @tf.function
  def server_update(model, server_optimizer, server_optimizer_vars,
                    server_state, weights_delta, aggregator_state):
    """Updates `server_state` based on `weights_delta`.

    Args:
      model: A `ReconstructionModel`.
      server_optimizer: A `tf.keras.optimizers.Optimizer`.
      server_optimizer_vars: A list of variables of server_optimizer.
      server_state: A `ServerState`, the state to be updated.
      weights_delta: An update to the trainable variables of the model.
      aggregator_state: The state of the aggregator after performing
        aggregation.

    Returns:
      An updated `ServerState`.
    """
    global_model_weights = reconstruction_utils.get_global_variables(model)
    # Initialize the model with the current state.
    tf.nest.map_structure(lambda a, b: a.assign(b),
                          (global_model_weights, server_optimizer_vars),
                          (server_state.model, server_state.optimizer_state))

    weights_delta, has_non_finite_weight = (
        tensor_utils.zero_all_if_any_non_finite(weights_delta))
    # We ignore the update if the weights_delta is non finite.
    if has_non_finite_weight > 0:
      return tff.structure.update_struct(
          server_state,
          model=global_model_weights,
          optimizer_state=server_optimizer_vars,
          round_num=server_state.round_num + 1,
          aggregator_state=aggregator_state)

    # Apply the update to the model.
    grads_and_vars = tf.nest.map_structure(
        lambda x, v: (-1.0 * x, v), tf.nest.flatten(weights_delta),
        tf.nest.flatten(global_model_weights.trainable))
    server_optimizer.apply_gradients(grads_and_vars, name='server_update')

    # Create a new state based on the updated model.
    return tff.structure.update_struct(
        server_state,
        model=global_model_weights,
        optimizer_state=server_optimizer_vars,
        round_num=server_state.round_num + 1,
        aggregator_state=aggregator_state,
    )

  @tff.tf_computation(server_state_type, model_weights_type.trainable,
                      aggregator_state_type)
  def server_update_tf(server_state, model_delta, aggregator_state):
    """Updates the `server_state`.

    Args:
      server_state: The `ServerState`.
      model_delta: The model delta in global trainable variables from clients.
      aggregator_state: The state of the aggregator after performing
        aggregation.

    Returns:
      The updated `ServerState`.
    """
    model = model_fn()
    server_optimizer = server_optimizer_fn()
    # Create optimizer variables so we have a place to assign the optimizer's
    # state.
    server_optimizer_vars = reconstruction_utils.create_optimizer_vars(
        model, server_optimizer)

    return server_update(model, server_optimizer, server_optimizer_vars,
                         server_state, model_delta, aggregator_state)

  return server_update_tf


def build_client_update_fn(
    model_fn: ModelFn,
    *,  # Callers should use keyword args for below.
    loss_fn: LossFn,
    metrics_fn: Optional[MetricsFn],
    tf_dataset_type: tff.SequenceType,
    model_weights_type: tff.Type,
    client_optimizer_fn: OptimizerFn,
    reconstruction_optimizer_fn: OptimizerFn,
    dataset_split_fn: reconstruction_utils.DatasetSplitFn,
    evaluate_reconstruction: bool,
    jointly_train_variables: bool,
    client_weight_fn: Optional[ClientWeightFn] = None,
) -> tff.tf_computation:
  """Builds a `tff.tf_computation` for local model reconstruction and training.

  Args:
    model_fn: A no-arg function that returns a `ReconstructionModel`.
    loss_fn: A no-arg function returning a `tf.keras.losses.Loss` to use to
      compute local model updates during reconstruction and post-reconstruction
      and evaluate the model during training. The final loss metric is the
      example-weighted mean loss across batches and across clients. Depending on
      whether `evaluate_reconstruction` is True, the loss metric may or may not
      include reconstruction batches in the loss.
    metrics_fn: A no-arg function returning a list of `tf.keras.metrics.Metric`s
      to evaluate the model. Metrics results are computed locally as described
      by the metric, and are aggregated across clients as in
      `federated_aggregate_keras_metric`. If None, no metrics are applied.
      Depending on whether evaluate_reconstruction is True, metrics may or may
      not be computed on reconstruction batches as well.
    tf_dataset_type: type_signature of dataset.
    model_weights_type: type_signature of model weights.
    client_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer` for training the model weights on the
      client post-reconstruction.
    reconstruction_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer` for reconstructing the local variables
      with global variables frozen. This optimizer is used before the one given
      by client_optimizer_fn.
    dataset_split_fn: A `reconstruction_utils.DatasetSplitFn` taking in a client
      dataset and training round number (1-indexed) and producing two TF
      datasets. The first is iterated over during reconstruction, and the second
      is iterated over post-reconstruction. This can be used to preprocess
      datasets to e.g. iterate over them for multiple epochs or use disjoint
      data for reconstruction and post-reconstruction. If None,
      `reconstruction_utils.simple_dataset_split_fn` is used, which results in
      iterating over the original client data for both phases of training. See
      `reconstruction_utils.build_dataset_split_fn` for options.
    evaluate_reconstruction: If True, metrics (including loss) are computed on
      batches during reconstruction and post-reconstruction. If False, metrics
      are computed on batches only post-reconstruction, when global weights are
      being updated. Note that metrics are aggregated across batches as given by
      the metric (example-weighted mean for the loss). Setting this to True
      includes all local batches in metric calculations. Setting this to False
      brings the interpretation of these metrics closer to the interpretation of
      metrics in FedAvg. Note that this does not affect training at all: losses
        for individual batches are calculated and used to update variables
        regardless.
    jointly_train_variables: Whether to train local variables after the
      reconstruction stage. If True, global and local variables are trained
      jointly after reconstruction of local variables, using the optimizer given
      by `client_optimizer_fn`. If False, only global variables are trained
      after the reconstruction stage with local variables frozen, similar to
      alternating minimization.
    client_weight_fn: Optional function that takes the local model's output, and
      returns a tensor that provides the weight in the federated average of
      model deltas. If not provided, the default is the total number of examples
      processed on device during post-reconstruction phase.

  Returns:
    A `tff.tf_computation` for local model optimization.
  """

  @tf.function
  def client_update(model, metrics, batch_loss_fn, dataset, initial_weights,
                    client_optimizer, reconstruction_optimizer, round_num):
    """Updates client model.

    Outputted weight deltas represent the difference between final global
    variables and initial ones. The client weight (used in aggregation across
    clients) is the sum of the number of examples across all batches
    post-reconstruction (that is, only the local steps that involve updating
    global variables).

    Args:
      model: A `ReconstructionModel`.
      metrics: A List of `tf.keras.metrics.Metric`s containing metrics to be
        computed and aggregated across clients.
      batch_loss_fn: A `tf.keras.losses.Loss` used to compute batch loss on
        `BatchOutput.predictions` (y_pred) and `BatchOutput.labels` (y_true) for
        each batch during and after reconstruction.
      dataset: A 'tf.data.Dataset'.
      initial_weights: A `tff.learning.ModelWeights` containing global trainable
        and non-trainable weights from the server.
      client_optimizer: a `tf.keras.optimizers.Optimizer` for training after the
        reconstruction step.
      reconstruction_optimizer: a `tf.keras.optimizers.Optimizer` for
        reconstruction of local trainable variables.
      round_num: the federated training round number, 1-indexed.

    Returns:
      A 'reconstruction_utils.ClientOutput`.
    """
    global_model_weights = reconstruction_utils.get_global_variables(model)
    local_model_weights = reconstruction_utils.get_local_variables(model)
    tf.nest.map_structure(lambda a, b: a.assign(b), global_model_weights,
                          initial_weights)

    @tf.function
    def reconstruction_reduce_fn(num_examples_sum, batch):
      """Runs reconstruction training on local client batch."""
      with tf.GradientTape() as tape:
        output = model.forward_pass(batch, training=True)
        batch_loss = batch_loss_fn(
            y_true=output.labels, y_pred=output.predictions)

      gradients = tape.gradient(batch_loss, local_model_weights.trainable)
      reconstruction_optimizer.apply_gradients(
          zip(gradients, local_model_weights.trainable))

      # Update metrics if needed.
      if evaluate_reconstruction:
        for metric in metrics:
          metric.update_state(y_true=output.labels, y_pred=output.predictions)

      return num_examples_sum + output.num_examples

    @tf.function
    def train_reduce_fn(num_examples_sum, batch):
      """Runs one step of client optimizer on local client batch."""
      if jointly_train_variables:
        all_trainable_variables = (
            global_model_weights.trainable + local_model_weights.trainable)
      else:
        all_trainable_variables = global_model_weights.trainable
      with tf.GradientTape() as tape:
        output = model.forward_pass(batch, training=True)
        batch_loss = batch_loss_fn(
            y_true=output.labels, y_pred=output.predictions)

      gradients = tape.gradient(batch_loss, all_trainable_variables)
      client_optimizer.apply_gradients(zip(gradients, all_trainable_variables))

      # Update each metric.
      for metric in metrics:
        metric.update_state(y_true=output.labels, y_pred=output.predictions)

      return num_examples_sum + output.num_examples

    recon_dataset, post_recon_dataset = dataset_split_fn(dataset, round_num)

    # If needed, do reconstruction, training the local variables while keeping
    # the global ones frozen.
    if local_model_weights.trainable:
      # Ignore output number of examples used in reconstruction, since this
      # isn't included in `client_weight`.
      recon_dataset.reduce(
          initial_state=tf.constant(0), reduce_func=reconstruction_reduce_fn)

    # Train the global variables, possibly jointly with local variables if
    # jointly_train_variables is True.
    num_examples_sum = post_recon_dataset.reduce(
        initial_state=tf.constant(0), reduce_func=train_reduce_fn)

    weights_delta = tf.nest.map_structure(lambda a, b: a - b,
                                          global_model_weights.trainable,
                                          initial_weights.trainable)

    # We ignore the update if the weights_delta is non finite.
    weights_delta, has_non_finite_weight = (
        tensor_utils.zero_all_if_any_non_finite(weights_delta))

    model_local_outputs = keras_utils.read_metric_variables(metrics)

    if has_non_finite_weight > 0:
      client_weight = tf.constant(0, dtype=tf.float32)
    elif client_weight_fn is None:
      client_weight = tf.cast(num_examples_sum, dtype=tf.float32)
    else:
      client_weight = client_weight_fn(model_local_outputs)

    return reconstruction_utils.ClientOutput(weights_delta, client_weight,
                                             model_local_outputs)

  @tff.tf_computation(tf_dataset_type, model_weights_type, tf.int64)
  def client_delta_tf(tf_dataset, initial_model_weights, round_num):
    """Performs client local model optimization.

    Args:
      tf_dataset: a `tf.data.Dataset` that provides training examples.
      initial_model_weights: a `tff.learning.ModelWeights` containing the
        starting global trainable and non_trainable weights.
      round_num: the federated training round number, 1-indexed.

    Returns:
      A `ClientOutput`.
    """
    model = model_fn()
    client_optimizer = client_optimizer_fn()
    reconstruction_optimizer = reconstruction_optimizer_fn()

    metrics = []
    if metrics_fn is not None:
      metrics.extend(metrics_fn())
    # To be used to calculate example-weighted mean across batches and clients.
    metrics.append(keras_utils.MeanLossMetric(loss_fn()))
    # To be used to calculate batch loss for model updates.
    batch_loss_fn = loss_fn()

    return client_update(model, metrics, batch_loss_fn, tf_dataset,
                         initial_model_weights, client_optimizer,
                         reconstruction_optimizer, round_num)

  return client_delta_tf


def build_run_one_round_fn(
    server_update_fn: TFComputationFn,
    client_update_fn: TFComputationFn,
    federated_output_computation: tff.federated_computation,
    federated_server_state_type: tff.Type,
    federated_dataset_type: tff.SequenceType,
    aggregation_process: tff.templates.AggregationProcess,
) -> tff.federated_computation:
  """Builds a `tff.federated_computation` for a round of training.

  Args:
    server_update_fn: A function for updates in the server.
    client_update_fn: A function for updates in the clients.
    federated_output_computation: A `tff.federated_computation` for aggregating
      local model outputs across clients.
    federated_server_state_type: type_signature of federated server state.
    federated_dataset_type: type_signature of federated dataset.
    aggregation_process: Instance of `tff.templates.AggregationProcess` to
      perform aggregation during the round.

  Returns:
    A `tff.federated_computation` for a round of training.
  """

  @tff.federated_computation(federated_server_state_type,
                             federated_dataset_type)
  def run_one_round(server_state, federated_dataset):
    """Orchestration logic for one round of computation.

    Args:
      server_state: A `ServerState`.
      federated_dataset: A federated `tf.Dataset` with placement `tff.CLIENTS`.

    Returns:
      A tuple of updated `ServerState` and aggregated metrics.
    """
    client_model = tff.federated_broadcast(server_state.model)
    client_round_number = tff.federated_broadcast(server_state.round_num)

    client_outputs = tff.federated_map(
        client_update_fn,
        (federated_dataset, client_model, client_round_number))

    if aggregation_process.is_weighted:
      aggregation_output = aggregation_process.next(
          server_state.aggregator_state,
          client_outputs.weights_delta,
          weight=client_outputs.client_weight)
    else:
      aggregation_output = aggregation_process.next(
          server_state.aggregator_state, client_outputs.weights_delta)

    round_model_delta = aggregation_output.result

    server_state = tff.federated_map(
        server_update_fn,
        (server_state, round_model_delta, aggregation_output.state))

    aggregated_model_outputs = federated_output_computation(
        client_outputs.model_output)

    # We drop the `measurements` portion of the aggregation_output here, as it
    # is not necessary for our experiments.

    return server_state, aggregated_model_outputs

  return run_one_round


def _instantiate_aggregation_process(
    aggregation_factory, model_weights_type,
    client_weight_fn) -> tff.templates.AggregationProcess:
  """Constructs aggregation process given factory, checking compatibilty."""
  if aggregation_factory is None:
    aggregation_factory = tff.aggregators.MeanFactory()
    aggregation_process = aggregation_factory.create(
        model_weights_type.trainable, tff.TensorType(tf.float32))
  else:
    # We give precedence to unweighted aggregation.
    if isinstance(aggregation_factory,
                  tff.aggregators.UnweightedAggregationFactory):
      if client_weight_fn is not None:
        logging.warning(
            'When using an unweighted aggregation, '
            '`client_weight_fn` should not be specified; found '
            '`client_weight_fn` %s', client_weight_fn)
      aggregation_process = aggregation_factory.create(
          model_weights_type.trainable)
    elif isinstance(aggregation_factory,
                    tff.aggregators.WeightedAggregationFactory):
      aggregation_process = aggregation_factory.create(
          model_weights_type.trainable, tff.TensorType(tf.float32))
    else:
      raise ValueError('Unknown type of aggregation factory: {}'.format(
          type(aggregation_factory)))
  return aggregation_process


def build_federated_reconstruction_process(
    model_fn: ModelFn,
    *,  # Callers pass below args by name.
    loss_fn: LossFn,
    metrics_fn: Optional[MetricsFn] = None,
    server_optimizer_fn: OptimizerFn = functools.partial(
        tf.keras.optimizers.SGD, 1.0),
    client_optimizer_fn: OptimizerFn = functools.partial(
        tf.keras.optimizers.SGD, 0.1),
    reconstruction_optimizer_fn: OptimizerFn = functools.partial(
        tf.keras.optimizers.SGD, 0.1),
    dataset_split_fn: Optional[reconstruction_utils.DatasetSplitFn] = None,
    evaluate_reconstruction: bool = False,
    jointly_train_variables: bool = False,
    client_weight_fn: Optional[ClientWeightFn] = None,
    aggregation_factory: Optional[
        tff.aggregators.WeightedAggregationFactory] = None,
) -> tff.templates.IterativeProcess:
  """Builds the IterativeProcess for optimization using FedRecon.

  Returns a `tff.templates.IterativeProcess` for Federated Reconstruction. On
  the client, computation can be divided into two stages: (1) reconstruction of
  local variables and (2) training of global variables (possibly jointly with
  reconstructed local variables).

  Args:
    model_fn: A no-arg function that returns a `ReconstructionModel`. This
      method must *not* capture Tensorflow tensors or variables and use them.
      must be constructed entirely from scratch on each invocation, returning
      the same pre-constructed model each call will result in an error.
    loss_fn: A no-arg function returning a `tf.keras.losses.Loss` to use to
      compute local model updates during reconstruction and post-reconstruction
      and evaluate the model during training. The final loss metric is the
      example-weighted mean loss across batches and across clients. Depending on
      whether `evaluate_reconstruction` is True, the loss metric may or may not
      include reconstruction batches in the loss.
    metrics_fn: A no-arg function returning a list of `tf.keras.metrics.Metric`s
      to evaluate the model. Metrics results are computed locally as described
      by the metric, and are aggregated across clients as in
      `federated_aggregate_keras_metric`. If None, no metrics are applied.
      Depending on whether evaluate_reconstruction is True, metrics may or may
      not be computed on reconstruction batches as well.
    server_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer` for applying updates to the global model
      on the server.
    client_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer` for local client training after
      reconstruction.
    reconstruction_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer` used to reconstruct the local variables,
      with the global ones frozen, or the first stage described above.
    dataset_split_fn: A `reconstruction_utils.DatasetSplitFn` taking in a client
      dataset and training round number (1-indexed) and producing two TF
      datasets. The first is iterated over during reconstruction, and the second
      is iterated over post-reconstruction. This can be used to preprocess
      datasets to e.g. iterate over them for multiple epochs or use disjoint
      data for reconstruction and post-reconstruction. If None,
      `reconstruction_utils.simple_dataset_split_fn` is used, which results in
      iterating over the original client data for both phases of training. See
      `reconstruction_utils.build_dataset_split_fn` for options.
    evaluate_reconstruction: If True, metrics (including loss) are computed on
      batches during reconstruction and post-reconstruction. If False, metrics
      are computed on batches only post-reconstruction, when global weights are
      being updated. Note that metrics are aggregated across batches as given by
      the metric (example-weighted mean for the loss). Setting this to True
      includes all local batches in metric calculations. Setting this to False
      brings the interpretation of these metrics closer to the interpretation of
      metrics in FedAvg. Note that this does not affect training at all: losses
        for individual batches are calculated and used to update variables
        regardless.
    jointly_train_variables: Whether to train local variables during the second
      stage described above. If True, global and local variables are trained
      jointly after reconstruction of local variables using the optimizer given
      by client_optimizer_fn. If False, only global variables are trained during
      the second stage with local variables frozen, similar to alternating
      minimization.
    client_weight_fn: Optional function that takes the local model's output, and
      returns a tensor that provides the weight in the federated average of
      model deltas. If not provided, the default is the total number of examples
      processed on device during post-reconstruction phase.
    aggregation_factory: An optional instance of
      `tff.aggregators.WeightedAggregationFactory` determining the method of
      aggregation to perform. If unspecified, uses a default
      `tff.aggregators.MeanFactory` which computes a stateless weighted mean
      across clients.

  Returns:
    A `tff.templates.IterativeProcess`.
  """
  with tf.Graph().as_default():
    throwaway_model_for_metadata = model_fn()

  model_weights_type = tff.framework.type_from_tensors(
      reconstruction_utils.get_global_variables(throwaway_model_for_metadata))

  aggregation_process = _instantiate_aggregation_process(
      aggregation_factory, model_weights_type, client_weight_fn)
  aggregator_state_type = (
      aggregation_process.initialize.type_signature.result.member)

  server_init_tff = build_server_init_fn(model_fn, server_optimizer_fn,
                                         aggregation_process)
  server_state_type = server_init_tff.type_signature.result.member

  server_update_fn = build_server_update_fn(
      model_fn,
      server_optimizer_fn,
      server_state_type,
      server_state_type.model,
      aggregator_state_type=aggregator_state_type)

  tf_dataset_type = tff.SequenceType(throwaway_model_for_metadata.input_spec)
  if dataset_split_fn is None:
    dataset_split_fn = reconstruction_utils.simple_dataset_split_fn
  client_update_fn = build_client_update_fn(
      model_fn,
      loss_fn=loss_fn,
      metrics_fn=metrics_fn,
      tf_dataset_type=tf_dataset_type,
      model_weights_type=server_state_type.model,
      client_optimizer_fn=client_optimizer_fn,
      reconstruction_optimizer_fn=reconstruction_optimizer_fn,
      dataset_split_fn=dataset_split_fn,
      evaluate_reconstruction=evaluate_reconstruction,
      jointly_train_variables=jointly_train_variables,
      client_weight_fn=client_weight_fn)

  federated_server_state_type = tff.type_at_server(server_state_type)
  federated_dataset_type = tff.type_at_clients(tf_dataset_type)
  # Create placeholder metrics to produce a corresponding federated output
  # computation.
  metrics = []
  if metrics_fn is not None:
    metrics.extend(metrics_fn())
  metrics.append(keras_utils.MeanLossMetric(loss_fn()))
  federated_output_computation = (
      keras_utils.federated_output_computation_from_metrics(metrics))

  run_one_round_tff = build_run_one_round_fn(
      server_update_fn,
      client_update_fn,
      federated_output_computation,
      federated_server_state_type,
      federated_dataset_type,
      aggregation_process=aggregation_process,
  )

  iterative_process = tff.templates.IterativeProcess(
      initialize_fn=server_init_tff, next_fn=run_one_round_tff)

  @tff.tf_computation(server_state_type)
  def get_model_weights(server_state):
    return server_state.model

  iterative_process.get_model_weights = get_model_weights
  return iterative_process
