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

import collections

from absl import logging
import tensorflow as tf
import tensorflow_federated as tff

from shrink_unshrink.simple_fedavg_tf import BroadcastMessage
from shrink_unshrink.simple_fedavg_tf import build_learned_sparse_projection_matrix
from shrink_unshrink.simple_fedavg_tf import create_left_maskval_to_projmat_dict
from shrink_unshrink.simple_fedavg_tf import get_model_weights
from shrink_unshrink.simple_fedavg_tf import left_sample_covariance_helper
from shrink_unshrink.simple_fedavg_tf import project_server_weights
from shrink_unshrink.simple_fedavg_tf import right_sample_covariance_helper
from shrink_unshrink.simple_fedavg_tf import ServerState
from shrink_unshrink.simple_fedavg_tf import ShrinkUnshrinkServerInfo
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

  A single set of random matrices are generated using
  shrink_unshrink_info.build_projection_matrix
  for any given cohort (i.e., each client within each cohort uses the same set
  of random matrices). The round number is used as a seed to achieve this end.

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
        server_state.round_num //
        shrink_unshrink_info.new_projection_dict_decimate,
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
  """Creates an unshrink function which unshrinks by unprojecting weight matrices corresponding to make_layerwise_projection_shrink.

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
    whimsy_client_weights = get_model_weights(client_model_fn()).trainable
    left_maskval_to_projmat_dict = create_left_maskval_to_projmat_dict(
        client_output.round_num //
        shrink_unshrink_info.new_projection_dict_decimate,
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
    client_outputs = tff.federated_map(unproject_client_weights_fn,
                                       client_outputs)
    my_weights_delta = tff.federated_map(reshape_a,
                                         client_outputs.weights_delta)
    round_model_delta = tff.federated_mean(
        my_weights_delta, weight=client_outputs.client_weight)
    logging.info("finished computing mean")

    return tff.federated_map(server_update_fn,
                             (server_state, round_model_delta))

  return unshrink


def make_learnedv2_layerwise_projection_shrink(*, server_state_type,
                                               tf_client_dataset_type,
                                               server_message_fn,
                                               server_model_fn, client_model_fn,
                                               shrink_unshrink_info):
  """Creates a shrink function which shrink by projecting weight matrices.

  The learning algorithm is the same proposed by
  make_learned_layerwise_projection_shrink;
  however, unlike the aforementioned algorithm, this function generates a
  different random projection matrix for each user selected within a cohort on
  "explore" (i.e., random_project_weights_and_create_broadcast_message_fn) steps

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
    A corresponding shrink functions.
  """
  del server_message_fn
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

  @tff.tf_computation(server_state_type, tf_client_dataset_type)
  def random_project_weights_and_create_broadcast_message_fn(
      server_state, federated_dataset):
    del federated_dataset
    gen = tf.random.Generator.from_non_deterministic_state()
    my_seed = gen.uniform_full_int(())
    whimsy_server_weights = get_model_weights(server_model_fn()).trainable
    whimsy_client_weights = get_model_weights(client_model_fn()).trainable
    left_maskval_to_projmat_dict = create_left_maskval_to_projmat_dict(
        my_seed,
        whimsy_server_weights,
        whimsy_client_weights,
        left_mask,
        right_mask,
        build_projection_matrix=build_projection_matrix)
    projected_server_state = project_server_weights(
        server_state, left_maskval_to_projmat_dict, left_mask, right_mask)
    return BroadcastMessage(
        model_weights=projected_server_state.model_weights,
        round_num=projected_server_state.round_num,
        shrink_unshrink_dynamic_info=left_maskval_to_projmat_dict)

  @tff.tf_computation(server_state_type, tf_client_dataset_type)
  def learned_project_weights_and_create_broadcast_message_fn(
      server_state, federated_dataset):
    del federated_dataset
    left_maskval_to_projmat_dict = server_state.shrink_unshrink_server_info.oja_left_maskval_to_projmat_dict
    projected_server_state = project_server_weights(
        server_state, left_maskval_to_projmat_dict, left_mask, right_mask)
    return BroadcastMessage(
        model_weights=projected_server_state.model_weights,
        round_num=projected_server_state.round_num,
        shrink_unshrink_dynamic_info=left_maskval_to_projmat_dict)

  @tff.tf_computation(server_state_type, tf_client_dataset_type)
  def logical_helper(server_state, federated_dataset):
    # pyformat: disable
    # pylint:disable=g-long-lambda
    return tf.cond(
        tf.equal(tf.math.floormod(server_state.round_num, 2), 0),
        lambda: random_project_weights_and_create_broadcast_message_fn(
            server_state, federated_dataset),
        lambda: learned_project_weights_and_create_broadcast_message_fn(
            server_state, federated_dataset))
    # pylint:enable=g-long-lambda
    # pyformat: enable

  @tff.federated_computation(federated_server_state_type,
                             federated_dataset_type)
  def shrink(server_state, federated_dataset):
    server_state_at_client = tff.federated_broadcast(server_state)
    server_message_at_client = tff.federated_map(
        logical_helper, (server_state_at_client, federated_dataset))
    return server_message_at_client

  return shrink


def make_learned_layerwise_projection_shrink(*, server_state_type,
                                             tf_client_dataset_type,
                                             server_message_fn, server_model_fn,
                                             client_model_fn,
                                             shrink_unshrink_info):
  """Creates a shrink function which shrink by projecting weight matrices.

  This algorithm alternates between executing two different types of shrink
  steps. In explore steps when
  random_project_weights_and_create_broadcast_message_fn is called
  the server weights are shrinked a la make_layerwise_projection_shrink (i.e., a
  single set of random projection matrices is used. On exploit steps, the
  matrices defined in
  server_state.shrink_unshrink_server_info.oja_left_maskval_to_projmat_dict are
  used.

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
    A corresponding shrink function.
  """
  del server_message_fn
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
  def random_project_weights_and_create_broadcast_message_fn(server_state):
    whimsy_server_weights = get_model_weights(server_model_fn()).trainable
    whimsy_client_weights = get_model_weights(client_model_fn()).trainable
    left_maskval_to_projmat_dict = create_left_maskval_to_projmat_dict(
        server_state.round_num,
        whimsy_server_weights,
        whimsy_client_weights,
        left_mask,
        right_mask,
        build_projection_matrix=build_projection_matrix)
    projected_server_state = project_server_weights(
        server_state, left_maskval_to_projmat_dict, left_mask, right_mask)
    return BroadcastMessage(
        model_weights=projected_server_state.model_weights,
        round_num=projected_server_state.round_num,
        shrink_unshrink_dynamic_info=left_maskval_to_projmat_dict)

  @tff.tf_computation(server_state_type)
  def learned_project_weights_and_create_broadcast_message_fn(server_state):
    left_maskval_to_projmat_dict = server_state.shrink_unshrink_server_info.oja_left_maskval_to_projmat_dict
    projected_server_state = project_server_weights(
        server_state, left_maskval_to_projmat_dict, left_mask, right_mask)
    return BroadcastMessage(
        model_weights=projected_server_state.model_weights,
        round_num=projected_server_state.round_num,
        shrink_unshrink_dynamic_info=left_maskval_to_projmat_dict)

  @tff.tf_computation(server_state_type)
  def logical_helper(server_state):
    # pyformat: disable
    # pylint:disable=g-long-lambda
    return tf.cond(
        tf.equal(tf.math.floormod(server_state.round_num, 2), 0), lambda:
        random_project_weights_and_create_broadcast_message_fn(server_state),
        lambda: learned_project_weights_and_create_broadcast_message_fn(
            server_state))  # pyformat: enable # pylint:enable=g-long-lambda

  @tff.federated_computation(federated_server_state_type,
                             federated_dataset_type)
  def shrink(server_state, federated_dataset):
    del federated_dataset
    server_message = tff.federated_map(logical_helper, (server_state))
    server_message_at_client = tff.federated_broadcast(server_message)
    return server_message_at_client

  return shrink


def make_learned_layerwise_projection_unshrink(*, server_state_type,
                                               client_update_output_type,
                                               server_update_fn,
                                               server_model_fn, client_model_fn,
                                               shrink_unshrink_info):
  """Creates an unshrink function which unshrinks by unprojecting weight matrices.

  This algorithm alternates between executing two different types of shrink
  steps. In explore steps, this function unshrinks the client weight matrices
  and constructs a 'sample_covariance' type matrix. A k-PCA streaming algorithm
  step (Oja's algorithm) is then used to compute the new projection matrices;
  the server state is not updated on these steps. In exploit steps, the client
  weight matrices are unshrunk and used to update the server model.

  Args:
    server_state_type: the type of server_state.
    client_update_output_type: the type of client_outputs.
    server_update_fn: a function which evolves the server_state.
    server_model_fn: a `tf.keras.Model' which specifies the server-side model.
    client_model_fn: a `tf.keras.Model' which specifies the client-side model.
    shrink_unshrink_info: an object specifying how the shrink and unshrink
      operations are performed.

  Returns:
    A corresponding unshrink function.
  """
  left_mask = shrink_unshrink_info.left_mask
  right_mask = shrink_unshrink_info.right_mask
  tf.debugging.assert_equal(len(left_mask), len(right_mask))
  tf.debugging.assert_equal(
      len(left_mask), len(get_model_weights(server_model_fn()).trainable))
  tf.debugging.assert_equal(
      len(left_mask), len(get_model_weights(client_model_fn()).trainable))

  federated_server_state_type = tff.type_at_server(server_state_type)
  federated_client_outputs_type = tff.type_at_clients(client_update_output_type)

  @tff.tf_computation(client_update_output_type)
  def unproject_client_weights_fn(client_output):
    left_maskval_to_projmat_dict = client_output.shrink_unshrink_dynamic_info
    return unproject_client_weights(client_output, left_maskval_to_projmat_dict,
                                    left_mask, right_mask)

  @tff.tf_computation
  def reshape_a(client_ouput_weight_delta):
    whimsy_server_weights = get_model_weights(server_model_fn()).trainable
    return tf.nest.map_structure(lambda a, b: tf.reshape(a, tf.shape(b)),
                                 client_ouput_weight_delta,
                                 whimsy_server_weights)

  @tff.tf_computation
  def logical_helper(server_state, round_model_delta):
    return tf.cond(
        tf.equal(tf.math.floormod(server_state.round_num, 2), 0),
        lambda: update_server_state_explore(server_state, round_model_delta),
        lambda: server_update_fn(server_state, round_model_delta))

  @tff.tf_computation
  def update_server_state_explore(server_state, round_model_delta):
    oja_dict = server_state.shrink_unshrink_server_info.oja_left_maskval_to_projmat_dict
    new_oja_dict = collections.OrderedDict()
    for k, v in oja_dict.items():
      if k != "-1":
        v_shape = tf.shape(v)
        new_oja_dict[str(k)] = tf.zeros((v_shape[1], v_shape[1]),
                                        dtype=tf.float32)

    # populate with sample covariance-type matrices
    for idx, mask_val in enumerate(left_mask):
      if mask_val % 1000 == 0 and mask_val > 0:
        new_oja_dict[str(mask_val // 1000)] += left_sample_covariance_helper(
            oja_dict, mask_val, round_model_delta[idx])
      elif mask_val != -1:
        new_oja_dict[str(mask_val)] += left_sample_covariance_helper(
            oja_dict, mask_val, round_model_delta[idx])

    for idx, mask_val in enumerate(right_mask):
      if mask_val % 1000 == 0 and mask_val > 0:
        raise ValueError(
            "special convolational support is only used in left multiplies.")
      elif mask_val != -1:
        new_oja_dict[str(mask_val)] += right_sample_covariance_helper(
            round_model_delta[idx])

    # run oja step
    final_oja_dict = collections.OrderedDict()
    for k, v in oja_dict.items():
      if k == "-1":
        final_oja_dict[str(k)] = oja_dict[str(k)]
        tf.debugging.assert_equal(final_oja_dict[str(k)], 1.0)
      else:
        my_shape = tf.shape(new_oja_dict[str(k)])
        tf.debugging.assert_equal(my_shape[0], my_shape[1])
        q_matrix, _ = tf.linalg.qr(
            (tf.eye(num_rows=tf.shape(new_oja_dict[str(k)])[0]) +
             server_state.shrink_unshrink_server_info.lmbda *
             new_oja_dict[str(k)]) @ tf.transpose(v))
        final_oja_dict[str(k)] = tf.transpose(q_matrix)
        tf.debugging.assert_equal(tf.shape(v), tf.shape(final_oja_dict[str(k)]))

    new_shrink_unshrink_server_info = ShrinkUnshrinkServerInfo(
        lmbda=server_state.shrink_unshrink_server_info.lmbda,
        oja_left_maskval_to_projmat_dict=final_oja_dict,
        memory_dict=server_state.shrink_unshrink_server_info.memory_dict)

    return ServerState(
        model_weights=server_state.model_weights,
        optimizer_state=server_state.optimizer_state,
        round_num=server_state.round_num + 1,
        shrink_unshrink_server_info=new_shrink_unshrink_server_info)

  @tff.federated_computation(federated_server_state_type,
                             federated_client_outputs_type)
  def unshrink(server_state, client_outputs):
    unprojected_client_outputs = tff.federated_map(unproject_client_weights_fn,
                                                   client_outputs)
    my_weights_delta = tff.federated_map(
        reshape_a, unprojected_client_outputs.weights_delta)
    round_model_delta = tff.federated_mean(
        my_weights_delta, weight=client_outputs.client_weight)
    logging.info("finished computing mean")

    return tff.federated_map(logical_helper, (server_state, round_model_delta))

  return unshrink


def make_learned_sparse_layerwise_projection_unshrink(
    *, server_state_type, client_update_output_type, server_update_fn,
    server_model_fn, client_model_fn, shrink_unshrink_info):
  """Creates an unshrink function which unshrinks by unprojecting weight matrices.

  This algorithm alternates between executing two different types of shrink
  steps. In explore steps, this function unshrinks the client weight matrices
  and constructs a 'sample_covariance' type matrix. A dropout matrix which
  corresponding to selecting the k largest row norms of the sample_covariance is
  constructed to be used as the projection matrix;
  the server state is not updated on these steps. In exploit steps, the client
  weight matrices are unshrunk and used to update the server model.

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

  federated_server_state_type = tff.type_at_server(server_state_type)
  federated_client_outputs_type = tff.type_at_clients(client_update_output_type)

  @tff.tf_computation(client_update_output_type)
  def unproject_client_weights_fn(client_output):
    left_maskval_to_projmat_dict = client_output.shrink_unshrink_dynamic_info
    return unproject_client_weights(client_output, left_maskval_to_projmat_dict,
                                    left_mask, right_mask)

  @tff.tf_computation
  def reshape_a(client_ouput_weight_delta):
    whimsy_server_weights = get_model_weights(server_model_fn()).trainable
    return tf.nest.map_structure(lambda a, b: tf.reshape(a, tf.shape(b)),
                                 client_ouput_weight_delta,
                                 whimsy_server_weights)

  @tff.tf_computation
  def logical_helper(server_state, round_model_delta):
    return tf.cond(
        tf.equal(tf.math.floormod(server_state.round_num, 2), 0),
        lambda: update_server_state_explore(server_state, round_model_delta),
        lambda: server_update_fn(server_state, round_model_delta))

  @tff.tf_computation
  def update_server_state_explore(server_state, round_model_delta):
    oja_dict = server_state.shrink_unshrink_server_info.oja_left_maskval_to_projmat_dict
    new_oja_dict = collections.OrderedDict()
    for k, v in oja_dict.items():
      if k != "-1":
        v_shape = tf.shape(v)
        new_oja_dict[str(k)] = tf.zeros((v_shape[1], v_shape[1]),
                                        dtype=tf.float32)

    # populate with sample covariance-type matrices
    for idx, mask_val in enumerate(left_mask):
      if mask_val % 1000 == 0 and mask_val > 0:
        new_oja_dict[str(mask_val // 1000)] += left_sample_covariance_helper(
            oja_dict, mask_val, round_model_delta[idx])
      elif mask_val != -1:
        new_oja_dict[str(mask_val)] += left_sample_covariance_helper(
            oja_dict, mask_val, round_model_delta[idx])

    for idx, mask_val in enumerate(right_mask):
      if mask_val % 1000 == 0 and mask_val > 0:
        raise ValueError(
            "special convolational support is only used in left multiplies.")
      elif mask_val != -1:
        new_oja_dict[str(mask_val)] += right_sample_covariance_helper(
            round_model_delta[idx])

    lmbda = server_state.shrink_unshrink_server_info.lmbda
    old_memory_dict = server_state.shrink_unshrink_server_info.memory_dict
    new_memory_dict = collections.OrderedDict()
    # run oja step
    final_oja_dict = collections.OrderedDict()
    for k, v in oja_dict.items():
      if k == "-1":
        final_oja_dict[str(k)] = oja_dict[str(k)]
        tf.debugging.assert_equal(final_oja_dict[str(k)], 1.0)
      else:
        my_shape = tf.shape(new_oja_dict[str(k)])
        tf.debugging.assert_equal(my_shape[0], my_shape[1])
        new_memory_dict[str(k)] = old_memory_dict[str(k)] * lmbda + (
            1 - lmbda) * new_oja_dict[str(k)]
        final_oja_dict[str(k)] = build_learned_sparse_projection_matrix(
            new_memory_dict[str(k)], tf.shape(v), is_left_multiply=True)

    new_shrink_unshrink_server_info = ShrinkUnshrinkServerInfo(
        lmbda=server_state.shrink_unshrink_server_info.lmbda,
        oja_left_maskval_to_projmat_dict=final_oja_dict,
        memory_dict=new_memory_dict)

    return ServerState(
        model_weights=server_state.model_weights,
        optimizer_state=server_state.optimizer_state,
        round_num=server_state.round_num + 1,
        shrink_unshrink_server_info=new_shrink_unshrink_server_info)

  @tff.federated_computation(federated_server_state_type,
                             federated_client_outputs_type)
  def unshrink(server_state, client_outputs):
    unprojected_client_outputs = tff.federated_map(unproject_client_weights_fn,
                                                   client_outputs)
    my_weights_delta = tff.federated_map(
        reshape_a, unprojected_client_outputs.weights_delta)
    round_model_delta = tff.federated_mean(
        my_weights_delta, weight=client_outputs.client_weight)

    return tff.federated_map(logical_helper, (server_state, round_model_delta))

  return unshrink


def make_static_client_specific_layerwise_projection_shrink(
    *,
    server_state_type,
    tf_client_id_type,
    server_message_fn,
    server_model_fn,
    client_model_fn,
    shrink_unshrink_info,
    static_client_layerwise_num_buckets):
  """Creates a shrink function which shrink by projecting weight matrices.

  Args:
    server_state_type: the type of server_state.
    tf_client_id_type: the type of the client id.
    server_message_fn: a function which converts the server_state into a
      `BroadcastMessage` object to be sent to clients.
    server_model_fn: a `tf.keras.Model' which specifies the server-side model.
    client_model_fn: a `tf.keras.Model' which specifies the client-side model.
    shrink_unshrink_info: an object specifying how the shrink and unshrink
      operations are performed.
    static_client_layerwise_num_buckets: an integer corresponding to the total
      number of hashbuckets a client id could be hashed to. Used to generate
      seeds for clients.

  Returns:
    A corresponding shrink and unshrink functions.
  """
  del server_message_fn
  left_mask = shrink_unshrink_info.left_mask
  right_mask = shrink_unshrink_info.right_mask
  tf.debugging.assert_equal(len(left_mask), len(right_mask))
  tf.debugging.assert_equal(
      len(left_mask), len(get_model_weights(server_model_fn()).trainable))
  tf.debugging.assert_equal(
      len(left_mask), len(get_model_weights(client_model_fn()).trainable))
  build_projection_matrix = shrink_unshrink_info.build_projection_matrix

  federated_server_state_type = tff.type_at_server(server_state_type)
  federated_client_id_type = tff.type_at_clients(tf_client_id_type)

  @tff.tf_computation(server_state_type, tf_client_id_type)
  def project_weights_and_create_broadcast_message_fn(server_state, client_id):
    my_seed = tf.strings.to_hash_bucket(
        client_id, num_buckets=static_client_layerwise_num_buckets)
    whimsy_server_weights = get_model_weights(server_model_fn()).trainable
    whimsy_client_weights = get_model_weights(client_model_fn()).trainable
    left_maskval_to_projmat_dict = create_left_maskval_to_projmat_dict(
        my_seed,
        whimsy_server_weights,
        whimsy_client_weights,
        left_mask,
        right_mask,
        build_projection_matrix=build_projection_matrix)
    new_server_state = project_server_weights(server_state,
                                              left_maskval_to_projmat_dict,
                                              left_mask, right_mask)
    return BroadcastMessage(
        model_weights=new_server_state.model_weights,
        round_num=new_server_state.round_num,
        shrink_unshrink_dynamic_info=my_seed)

  @tff.federated_computation(federated_server_state_type,
                             federated_client_id_type)
  def shrink(server_state, federated_client_id):
    server_state_at_client = tff.federated_broadcast(server_state)
    server_message_at_client = tff.federated_map(
        project_weights_and_create_broadcast_message_fn,
        (server_state_at_client, federated_client_id))
    return server_message_at_client

  return shrink


def make_client_specific_layerwise_projection_shrink(
    *, server_state_type, tf_client_dataset_type, server_message_fn,
    server_model_fn, client_model_fn, shrink_unshrink_info):
  """Creates a shrink function which shrink by projecting weight matrices.

  A distinct set of random matrices are generated using
  shrink_unshrink_info.build_projection_matrix
  for every client in any given cohort (i.e., each client within each cohort
  uses a different set of random matrices). A random seed is generated for every
  user using tf.random.Generator.from_non_deterministic_state() (seed generation
  is not reproducible).

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
  del server_message_fn
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

  @tff.tf_computation(server_state_type, tf_client_dataset_type)
  def project_weights_and_create_broadcast_message_fn(server_state,
                                                      federated_dataset):
    del federated_dataset
    gen = tf.random.Generator.from_non_deterministic_state()
    my_seed = gen.uniform_full_int(())
    whimsy_server_weights = get_model_weights(server_model_fn()).trainable
    whimsy_client_weights = get_model_weights(client_model_fn()).trainable
    left_maskval_to_projmat_dict = create_left_maskval_to_projmat_dict(
        my_seed,
        whimsy_server_weights,
        whimsy_client_weights,
        left_mask,
        right_mask,
        build_projection_matrix=build_projection_matrix)
    new_server_state = project_server_weights(server_state,
                                              left_maskval_to_projmat_dict,
                                              left_mask, right_mask)
    return BroadcastMessage(
        model_weights=new_server_state.model_weights,
        round_num=new_server_state.round_num,
        shrink_unshrink_dynamic_info=my_seed)

  @tff.federated_computation(federated_server_state_type,
                             federated_dataset_type)
  def shrink(server_state, federated_dataset):
    server_state_at_client = tff.federated_broadcast(server_state)
    server_message_at_client = tff.federated_map(
        project_weights_and_create_broadcast_message_fn,
        (server_state_at_client, federated_dataset)
    )
    return server_message_at_client

  return shrink


def make_client_specific_layerwise_projection_unshrink(
    *, server_state_type, client_update_output_type, server_update_fn,
    server_model_fn, client_model_fn, shrink_unshrink_info):
  """Creates an unshrink function which unshrinks by unprojecting weight matrices which corresponds to make_client_specific_layerwise_projection_shrink.

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
    whimsy_client_weights = get_model_weights(client_model_fn()).trainable
    # left_maskval_to_projmat_dict = client_output.shrink_unshrink_dynamic_info
    # could try to pass the whole dictionary

    my_seed = client_output.shrink_unshrink_dynamic_info
    left_maskval_to_projmat_dict = create_left_maskval_to_projmat_dict(
        my_seed,
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
