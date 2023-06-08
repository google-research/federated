# Copyright 2023, Google LLC.
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
"""An implementation of the Federated Averaging algorithm.

This is forked from TFF/simple_fedavg with the following changes for DP:
(1) clip the norm of the model delta from clients;
(2) aggregate the model delta from clients with uniform weighting.
"""
import collections
from collections.abc import Callable
from typing import Optional

import tensorflow as tf
import tensorflow_federated as tff

DEFAULT_CLIENT_OPTIMIZER_FN = lambda: tf.keras.optimizers.SGD(learning_rate=0.1)


def _unpack_data_label(batch):
  if isinstance(batch, collections.abc.Mapping):
    return batch['x'], batch['y']
  elif isinstance(batch, (tuple, list)):
    if len(batch) < 2:
      raise ValueError('Expecting both data and label from a batch.')
    return batch[0], batch[1]
  else:
    raise ValueError('Unrecognized batch data.')


@tf.function
def keras_evaluate(model, test_data, metrics):
  """Evaluates a Keras model against `test_data`, for each `metric`."""
  for metric in metrics:
    metric.reset_states()
  # Force autograph to generate a while+scan pattern, which TF may be able to
  # more easily partition across multiple GPUs.
  for batch in iter(test_data):
    batch_x, batch_y = _unpack_data_label(batch)
    preds = model(batch_x, training=False)
    for metric in metrics:
      metric.update_state(y_true=batch_y, y_pred=preds)
  return tf.nest.map_structure(lambda x: x.result(), metrics)


def build_dpftrl_fedavg_process(
    model_fn: Callable[[], tff.learning.models.VariableModel],
    client_optimizer_fn: Callable[
        [], tf.keras.optimizers.Optimizer
    ] = DEFAULT_CLIENT_OPTIMIZER_FN,
    *,  # Require named (non-positional) parameters for the following kwargs:
    server_learning_rate: float = 0.1,
    server_momentum: float = 0.9,
    server_nesterov: bool = False,
    use_experimental_simulation_loop: bool = False,
    dp_aggregator_factory: Optional[
        tff.aggregators.DifferentiallyPrivateFactory
    ] = None,
) -> tff.learning.templates.LearningProcess:
  """Builds an iterative process that performs federated averaging with differential privacy.

  This function creates a `tff.learning.templates.LearningProcess`. The server
  optimizer is DP-FTRL, as described in:

  "Practical and Private (Deep) Learning without Sampling or Shuffling".

  Args:
    model_fn: A no-arg function that returns a
      `tff.learning.models.VariableModel`. This method must *not* capture
      TensorFlow tensors or variables and use them. The model must be
      constructed entirely from scratch on each invocation, returning the same
      pre-constructed model each call will result in an error.
    client_optimizer_fn: A no-arg callable that returns a `tf.keras.Optimizer`.
    server_learning_rate: The learning rate of server DP-FTRL optimizer.
    server_momentum: The momentum of server DP-FTRL optimizer.
    server_nesterov: If true, use Nesterov momentum instead of heavyball.
    use_experimental_simulation_loop: Controls the reduce loop function for
      input dataset. An experimental reduce loop is used for simulation. It is
      currently necessary to set this flag to True for performant GPU
      simulations.
    dp_aggregator_factory: Optional instance of
      `tff.aggregators.DifferentiallyPrivateFactory` to use as aggregator. If
      `None`, TFF's default aggregator (weighted mean, with no privacy) is used.

  Returns:
    A `tff.templates.IterativeProcess`.
  """

  def server_optimizer_fn():
    return tf.keras.optimizers.SGD(
        learning_rate=server_learning_rate,
        momentum=server_momentum,
        nesterov=server_nesterov,
    )

  return tff.learning.algorithms.build_unweighted_fed_avg(
      model_fn,
      client_optimizer_fn,
      server_optimizer_fn,
      model_aggregator=dp_aggregator_factory,
      use_experimental_simulation_loop=use_experimental_simulation_loop,
  )
