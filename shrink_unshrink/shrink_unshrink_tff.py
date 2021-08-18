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
"""Contains TFF wrappers to make various shrink and unshrink operations.

Specifically, build_shrink_unshrink_process usese these functions and its
parameters to construct an iterative process which utilizes the shrink unshrink
functionality. This file contains multiple implementations of ways one could do
this shrink and unshrink procedure.
"""

from absl import logging
import tensorflow as tf
import tensorflow_federated as tff

from shrink_unshrink.simple_fedavg_tf import create_left_maskval_to_projmat_dict
from shrink_unshrink.simple_fedavg_tf import get_model_weights
from shrink_unshrink.simple_fedavg_tf import project_server_weights
from shrink_unshrink.simple_fedavg_tf import unproject_client_weights


def make_identity_shrink(*, server_state_type, tf_client_dataset_type,
                         server_message_fn, server_model_fn, client_model_fn,
                         shrink_unshrink_info):
  """Creates an identity shrink function.

  This version does not do anything different from the original code, it
  just changed the structural organization.

  Args:
    server_state_type: the type of server_state.
    tf_client_dataset_type: the type of the client dataset.
    server_message_fn: a function which converts the server_state into a
      `BroadcastMessage` object to be sent to clients.
    server_model_fn: a `tf.keras.Model' which specifies the server-side model.
    client_model_fn: a `tf.keras.Model' which specifies the client-side model.
    shrink_unshrink_info: an object specifying how the shrink and unshrink
      operations are performed.

  Returns:
    A corresponding shrink and unshrink functions.
  """
  del shrink_unshrink_info
  del server_model_fn
  del client_model_fn

  federated_server_state_type = tff.type_at_server(server_state_type)
  federated_dataset_type = tff.type_at_clients(tf_client_dataset_type)

  @tff.federated_computation(federated_server_state_type,
                             federated_dataset_type)
  def shrink(server_state, federated_dataset):
    del federated_dataset
    server_message = tff.federated_map(server_message_fn, server_state)
    server_message_at_client = tff.federated_broadcast(server_message)
    return server_message_at_client

  return shrink


def make_identity_unshrink(*, server_state_type, client_update_output_type,
                           server_update_fn, server_model_fn, client_model_fn,
                           shrink_unshrink_info):
  """Creates an identity unshrink function.

  This version does not do anything different from the original code, it
  just changed the structural organization.

  Args:
    server_state_type: the type of server_state.
    client_update_output_type: the type of client_outputs.
    server_update_fn: a function which evolves the server_state.
    server_model_fn: a `tf.keras.Model' which specifies the server-side model.
    client_model_fn: a `tf.keras.Model' which specifies the client-side model.
    shrink_unshrink_info: an object specifying how the shrink and unshrink
      operations are performed.

  Returns:
    A corresponding shrink and unshrink functions.
  """
  del shrink_unshrink_info
  del server_model_fn
  del client_model_fn

  federated_server_state_type = tff.type_at_server(server_state_type)
  federated_client_outputs_type = tff.type_at_clients(client_update_output_type)

  @tff.federated_computation(federated_server_state_type,
                             federated_client_outputs_type)
  def unshrink(server_state, client_outputs):
    round_model_delta = tff.federated_mean(
        client_outputs.weights_delta, weight=client_outputs.client_weight)

    return tff.federated_map(server_update_fn,
                             (server_state, round_model_delta))

  return unshrink


def make_layerwise_projection_shrink(*, server_state_type,
                                     tf_client_dataset_type, server_message_fn,
                                     server_model_fn, client_model_fn,
                                     shrink_unshrink_info):
  """Creates a shrink function which shrink by projecting weight matrices.

  Args:
    server_state_type: the type of server_state.
    tf_client_dataset_type: the type of the client dataset.
    server_message_fn: a function which converts the server_state into a
      `BroadcastMessage` object to be sent to clients.
    server_model_fn: a `tf.keras.Model' which specifies the server-side model.
    client_model_fn: a `tf.keras.Model' which specifies the client-side model.
    shrink_unshrink_info: an object specifying how the shrink and unshrink
      operations are performed.

  Returns:
    A corresponding shrink and unshrink functions.
  """
  left_mask = shrink_unshrink_info.left_mask
  right_mask = shrink_unshrink_info.right_mask
  tf.debugging.assert_equal(len(left_mask), len(right_mask))
  tf.debugging.assert_equal(
      len(left_mask), len(get_model_weights(server_model_fn()).trainable))
  tf.debugging.assert_equal(
      len(left_mask), len(get_model_weights(client_model_fn()).trainable))
  build_projection_matrix = shrink_unshrink_info.build_projection_matrix

  federated_server_state_type = tff.type_at_server(server_state_type)
  federated_dataset_type = tff.type_at_clients(tf_client_dataset_type)

  @tff.tf_computation(server_state_type)
  def project_server_weights_fn(server_state):
    whimsy_server_weights = get_model_weights(server_model_fn()).trainable
    whimsy_client_weights = get_model_weights(client_model_fn()).trainable
    left_maskval_to_projmat_dict = create_left_maskval_to_projmat_dict(
        server_state.round_num,
        whimsy_server_weights,
        whimsy_client_weights,
        left_mask,
        right_mask,
        build_projection_matrix=build_projection_matrix)
    return project_server_weights(server_state, left_maskval_to_projmat_dict,
                                  left_mask, right_mask)

  @tff.federated_computation(federated_server_state_type,
                             federated_dataset_type)
  def shrink(server_state, federated_dataset):
    del federated_dataset
    projected_server_state = tff.federated_map(project_server_weights_fn,
                                               server_state)
    server_message = tff.federated_map(server_message_fn,
                                       projected_server_state)
    server_message_at_client = tff.federated_broadcast(server_message)
    return server_message_at_client

  return shrink


def make_layerwise_projection_unshrink(*, server_state_type,
                                       client_update_output_type,
                                       server_update_fn, server_model_fn,
                                       client_model_fn, shrink_unshrink_info):
  """Creates an unshrink function which unshrinks by unprojecting weight matrices.

  Args:
    server_state_type: the type of server_state.
    client_update_output_type: the type of client_outputs.
    server_update_fn: a function which evolves the server_state.
    server_model_fn: a `tf.keras.Model' which specifies the server-side model.
    client_model_fn: a `tf.keras.Model' which specifies the client-side model.
    shrink_unshrink_info: an object specifying how the shrink and unshrink
      operations are performed.

  Returns:
    A corresponding shrink and unshrink functions.
  """
  left_mask = shrink_unshrink_info.left_mask
  right_mask = shrink_unshrink_info.right_mask
  tf.debugging.assert_equal(len(left_mask), len(right_mask))
  tf.debugging.assert_equal(
      len(left_mask), len(get_model_weights(server_model_fn()).trainable))
  tf.debugging.assert_equal(
      len(left_mask), len(get_model_weights(client_model_fn()).trainable))
  build_projection_matrix = shrink_unshrink_info.build_projection_matrix

  federated_server_state_type = tff.type_at_server(server_state_type)
  federated_client_outputs_type = tff.type_at_clients(client_update_output_type)

  @tff.tf_computation(client_update_output_type)
  def unproject_client_weights_fn(client_output):
    whimsy_server_weights = get_model_weights(server_model_fn()).trainable
    # gTODO should this be get_model_weights(...).trainable_variables?
    whimsy_client_weights = get_model_weights(client_model_fn()).trainable
    left_maskval_to_projmat_dict = create_left_maskval_to_projmat_dict(
        client_output.round_num,
        whimsy_server_weights,
        whimsy_client_weights,
        left_mask,
        right_mask,
        build_projection_matrix=build_projection_matrix)
    return unproject_client_weights(client_output, left_maskval_to_projmat_dict,
                                    left_mask, right_mask)

  @tff.tf_computation
  def reshape_a(client_ouput_weight_delta):
    whimsy_server_weights = get_model_weights(server_model_fn()).trainable
    return tf.nest.map_structure(lambda a, b: tf.reshape(a, tf.shape(b)),
                                 client_ouput_weight_delta,
                                 whimsy_server_weights)

  @tff.federated_computation(federated_server_state_type,
                             federated_client_outputs_type)
  def unshrink(server_state, client_outputs):
    # If we ever needed to track state,
    # we would likely need to put shrink_unshrink_info in server_state

    client_outputs = tff.federated_map(unproject_client_weights_fn,
                                       client_outputs)
    logging.info("going to compute mean")
    my_weights_delta = tff.federated_map(reshape_a,
                                         client_outputs.weights_delta)
    round_model_delta = tff.federated_mean(
        my_weights_delta, weight=client_outputs.client_weight
    )  # in practice, averaging would happen on server.
    logging.info("finished computing mean")

    return tff.federated_map(server_update_fn,
                             (server_state, round_model_delta))

  return unshrink
