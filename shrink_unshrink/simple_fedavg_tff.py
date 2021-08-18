# Copyright 2020, The TensorFlow Federated Authors.
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

This is intended to be a minimal stand-alone implementation of Federated
Averaging, suitable for branching as a starting point for algorithm
modifications; see `tff.learning.build_federated_averaging_process` for a
more full-featured implementation.

Based on the paper:

Communication-Efficient Learning of Deep Networks from Decentralized Data
    H. Brendan McMahan, Eider Moore, Daniel Ramage,
    Seth Hampson, Blaise Aguera y Arcas. AISTATS 2017.
    https://arxiv.org/abs/1602.05629
"""

from typing import Union

from absl import logging
import tensorflow as tf
import tensorflow_federated as tff

from shrink_unshrink.shrink_unshrink_tff import make_identity_shrink
from shrink_unshrink.shrink_unshrink_tff import make_identity_unshrink
from shrink_unshrink.simple_fedavg_tf import build_server_broadcast_message
from shrink_unshrink.simple_fedavg_tf import client_update
from shrink_unshrink.simple_fedavg_tf import get_model_weights
from shrink_unshrink.simple_fedavg_tf import KerasModelWrapper
from shrink_unshrink.simple_fedavg_tf import server_update
from shrink_unshrink.simple_fedavg_tf import ServerState


def _initialize_optimizer_vars(model: Union[tff.learning.Model,
                                            KerasModelWrapper],
                               optimizer: tf.keras.optimizers.Optimizer):
  """Creates optimizer variables to assign the optimizer's state."""
  # Create zero gradients to force an update that doesn't modify.
  # Force eagerly constructing the optimizer variables. Normally Keras lazily
  # creates the variables on first usage of the optimizer. Optimizers such as
  # Adam, Adagrad, or using momentum need to create a new set of variables shape
  # like the model weights.
  model_weights = get_model_weights(model)
  zero_gradient = [tf.zeros_like(t) for t in model_weights.trainable]
  optimizer.apply_gradients(zip(zero_gradient, model_weights.trainable))
  assert optimizer.variables()


def build_federated_shrink_unshrink_process(
    *,
    server_model_fn,
    client_model_fn,
    make_shrink=make_identity_shrink,
    make_unshrink=make_identity_unshrink,
    shrink_unshrink_info=None,
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1),
    debugging=False):
  """Builds the TFF computations for optimization using federated averaging.

  Args:
    server_model_fn: A no-arg function that returns a
      `simple_fedavg_tf.KerasModelWrapper`.
    client_model_fn: A no-arg function that returns a
      `simple_fedavg_tf.KerasModelWrapper`.
    make_shrink: A function which creates and returns a `shrink` function based
      on passed in parameters.
    make_unshrink: A function which creates and returns an `unshrink` function
      based on passed in parameters.
    shrink_unshrink_info: Context needed by the algorithm containing information
      about how shrinking and unshrinking operations are performed.
    server_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer` for server update.
    client_optimizer_fn: A no-arg function that returns a
      `tf.keras.optimizers.Optimizer` for client update.
    debugging: A boolean which if True returns the shrink and unshrink functions
      for testing/debugging purposes

  Returns:
    A `tff.templates.IterativeProcess`.
  """
  whimsy_client_model = client_model_fn()

  @tff.tf_computation
  def server_init_tf():
    model = server_model_fn()
    model_weights = get_model_weights(model)
    server_optimizer = server_optimizer_fn()
    _initialize_optimizer_vars(model, server_optimizer)
    return ServerState(
        model_weights=model_weights,
        optimizer_state=server_optimizer.variables(),
        round_num=0)

  server_state_type = server_init_tf.type_signature.result

  # model_weights_type = server_state_type.model_weights

  @tff.tf_computation
  def server_update_fn(
      server_state,
      model_delta):  # (server_state_type, model_weights_type.trainable)
    model = server_model_fn()
    server_optimizer = server_optimizer_fn()
    _initialize_optimizer_vars(model, server_optimizer)
    return server_update(model, server_optimizer, server_state, model_delta)

  tf_client_dataset_type = tff.SequenceType(whimsy_client_model.input_spec)

  @tff.tf_computation
  def server_message_fn(server_state):  # (server_state_type)
    return build_server_broadcast_message(server_state)

  shrink = make_shrink(
      server_state_type=server_state_type,
      tf_client_dataset_type=tf_client_dataset_type,
      server_message_fn=server_message_fn,
      server_model_fn=server_model_fn,
      client_model_fn=client_model_fn,
      shrink_unshrink_info=shrink_unshrink_info)

  server_message_type = shrink.type_signature.result.member

  @tff.tf_computation(tf_client_dataset_type, server_message_type)
  def client_update_fn(tf_dataset, server_message):
    model = client_model_fn()
    client_optimizer = client_optimizer_fn()
    return client_update(model, tf_dataset, server_message, client_optimizer)

  client_update_output_type = client_update_fn.type_signature.result

  unshrink = make_unshrink(
      server_state_type=server_state_type,
      client_update_output_type=client_update_output_type,
      server_update_fn=server_update_fn,
      server_model_fn=server_model_fn,
      client_model_fn=client_model_fn,
      shrink_unshrink_info=shrink_unshrink_info)

  federated_server_state_type = tff.type_at_server(server_state_type)
  federated_dataset_type = tff.type_at_clients(tf_client_dataset_type)

  @tff.federated_computation(federated_server_state_type,
                             federated_dataset_type)
  def run_one_round(server_state, federated_dataset):
    """Orchestration logic for one round of computation.

    Args:
      server_state: A `ServerState`.
      federated_dataset: A federated `tf.data.Dataset` with placement
        `tff.CLIENTS`.

    Returns:
      A tuple of updated `ServerState` and `tf.Tensor` of average loss.
    """
    logging.info("start of round")

    logging.info("shrink")
    server_message_at_client = shrink(server_state, federated_dataset)

    logging.info("client_update")
    client_outputs = tff.federated_map(
        client_update_fn, (federated_dataset, server_message_at_client))

    logging.info("unshrink")
    server_state = unshrink(server_state, client_outputs)

    round_loss_metric = tff.federated_mean(
        client_outputs.model_output, weight=client_outputs.client_weight)
    logging.info("end of round")

    return server_state, {"metric": round_loss_metric}

  @tff.federated_computation
  def server_init_tff():
    """Orchestration logic for server model initialization."""
    return tff.federated_value(server_init_tf(), tff.SERVER)

  if debugging:
    return tff.templates.IterativeProcess(
        initialize_fn=server_init_tff, next_fn=run_one_round), shrink, unshrink

  return tff.templates.IterativeProcess(
      initialize_fn=server_init_tff, next_fn=run_one_round)
