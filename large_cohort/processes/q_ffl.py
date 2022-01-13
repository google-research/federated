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
"""An implementation of the q-Fair Federated Learning (q-FFL) algorithm.

Based on the paper:

Fair Resource Allocation in Federated Learning.
    Tian Li, Maziar Sanjabi, Ahmad Beirami, Virginia Smith. ICLR 2020.
    https://arxiv.org/abs/1602.05629

Note that the primary distinction between this implementation and the algorithm
described in the paper above is that the paper weights each client by their loss
after training. This requires an extra pass over each client's dataset. In order
to reduce training time on clients, we use the loss computed as the client
trains to do the weighting in q-FFL.
"""

from typing import Any, Callable, Optional

import tensorflow as tf
import tensorflow_federated as tff

DEFAULT_SERVER_OPTIMIZER_FN = lambda: tf.keras.optimizers.SGD(learning_rate=1.0)


def build_keras_output_to_loss_fn(
    metric_builder=Callable[[], tf.keras.metrics.Metric]):
  """Creates a function that computes the result of a `tf.keras` metric."""

  def output_to_loss_fn(output):
    loss_variables = output['loss']
    metric = metric_builder()
    tf.nest.map_structure(lambda a, b: a.assign(b), metric.variables,
                          loss_variables)
    return metric.result()

  return output_to_loss_fn


def build_q_ffl_process(
    model_fn: Callable[[], tff.learning.Model],
    fairness_parameter: tf.Tensor,
    client_optimizer_fn: Callable[[], tf.keras.optimizers.Optimizer],
    server_optimizer_fn: Callable[
        [], tf.keras.optimizers.Optimizer] = DEFAULT_SERVER_OPTIMIZER_FN,
    broadcast_process: Optional[tff.templates.MeasuredProcess] = None,
    model_update_aggregation_factory: Optional[
        tff.aggregators.WeightedAggregationFactory] = None,
    use_experimental_simulation_loop: bool = False,
    output_to_loss_fn: Optional[Callable[[Any], tf.Tensor]] = None,
) -> tff.templates.IterativeProcess:
  """Builds an iterative process that performs q-FFL.

  This function creates a `tff.templates.IterativeProcess` that performs
  a variant of federated averaging on client models, where client updates are
  weighted according by their loss raised to the power `fairness_parameter`.

  The iterative process has the following methods inherited from
  `tff.templates.IterativeProcess`:

  *   `initialize`: A `tff.Computation` with the functional type signature
      `( -> S@SERVER)`, where `S` is a `tff.learning.framework.ServerState`
      representing the initial state of the server.
  *   `next`: A `tff.Computation` with the functional type signature
      `(<S@SERVER, {B*}@CLIENTS> -> <S@SERVER, T@SERVER>)` where `S` is a
      `tff.learning.framework.ServerState` whose type matches that of the output
      of `initialize`, and `{B*}@CLIENTS` represents the client datasets, where
      `B` is the type of a single batch. This computation returns a
      `tff.learning.framework.ServerState` representing the updated server state
      and metrics computed during training.

  The iterative process also has the following method not inherited from
  `tff.templates.IterativeProcess`:

  *   `get_model_weights`: A `tff.Computation` that takes as input the
      a `tff.learning.framework.ServerState`, and returns a
      `tff.learning.ModelWeights` containing the state's model weights.

  The internal logic of the resulting iterative process is the same as
  `tff.learning.build_federated_averaging_process`, but with a custom weighting
  function.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`. This method
      must *not* capture TensorFlow tensors or variables and use them. The model
      must be constructed entirely from scratch on each invocation, returning
      the same pre-constructed model each call will result in an error.
    fairness_parameter: A scalar tensor governing the exponent in the client
      weights. Must be convertible to a scalar `tf.float32`.
    client_optimizer_fn: A no-arg callable that returns a `tf.keras.Optimizer`.
    server_optimizer_fn: A no-arg callable that returns a `tf.keras.Optimizer`.
      By default, this uses `tf.keras.optimizers.SGD` with a learning rate of
      1.0.
    broadcast_process: a `tff.templates.MeasuredProcess` that broadcasts the
      model weights on the server to the clients. It must support the signature
      `(input_values@SERVER -> output_values@CLIENT)`. If set to default None,
      the server model is broadcast to the clients using the default
      tff.federated_broadcast.
    model_update_aggregation_factory: An optional
      `tff.aggregators.WeightedAggregationFactory` that constructs
      `tff.templates.AggregationProcess` for aggregating the client model
      updates on the server. If `None`, uses `tff.aggregators.MeanFactory`.
    use_experimental_simulation_loop: Controls the reduce loop function for
      input dataset. An experimental reduce loop is used for simulation. It is
      currently necessary to set this flag to True for performant GPU
      simulations.
    output_to_loss_fn: An optional callable that takes the result of
      `model_fn().report_local_unfinalized_metrics()` and returns a scalar
      tensor representing the loss of the model. If set to `None`, this method
      will assume that the loss will attempt to be extracted
      `model_fn().report_local_unfinalized_metrics()['loss']`.

  Returns:
    A `tff.templates.IterativeProcess`.
  """
  if output_to_loss_fn is None:
    output_to_loss_fn = lambda x: x['loss']

  def client_weighting(client_output):
    loss = output_to_loss_fn(client_output)
    return tf.math.pow(loss, fairness_parameter)

  return tff.learning.build_federated_averaging_process(
      model_fn=model_fn,
      client_optimizer_fn=client_optimizer_fn,
      server_optimizer_fn=server_optimizer_fn,
      client_weighting=client_weighting,
      broadcast_process=broadcast_process,
      model_update_aggregation_factory=model_update_aggregation_factory,
      use_experimental_simulation_loop=use_experimental_simulation_loop)
