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
"""Bandits iterative process for TFF simulation."""
import collections
from collections.abc import Callable
from typing import Any, Optional

import attr
import tensorflow as tf
import tensorflow_federated as tff

from bandits import bandits_utils


# The following keys are used to track bandits information on clients.
# The returned values are "averaged" rewards and probabilities of the inference
# actions over the sampled data on each client.
_REWARD_KEY = 'rewards'
_PROB_KEY = 'prob'
_WEIGHT_KEY = 'num_examples_weights'


@attr.s(eq=False, frozen=True, slots=True)
class ServerState(object):
  """Structure for state on the server.

  Attributes:
    delayed_inference_model: Model used for bandits inference to generate
      trainig data. The models are delayed from some rounds to simulate the
      previously deployed models in a practical system.
    train_state: The state of a TFF iterative process for model training.
    round_num: The number of completed training rounds, which is tracked so that
      the model from `train_state` can be deployed/copied to
      `delayed_inference_model` every a few rounds.
    average_reward: Tacks the average of the inference rewards. Round_reward are
      first averaged for all samples with inference time results from clients
      participating in the current traning round, and then average_rewards is
      computed by taking the average of round_reward across all rounds so far.
  """

  delayed_inference_model = attr.ib()
  train_state = attr.ib()
  round_num = attr.ib()
  average_reward = attr.ib()


# TODO(b/215566681): the `data_element_spec` is manully set for bandits.
# it may be automated by using
# `tff.simulation.compose_dataset_computation_with_computation`.
# TODO(b/215566681): verify TFF polymorphic status and cleanup the unnecessary
# types.
def build_bandits_iterative_process(
    *,
    model_fn: Callable[[], tff.learning.models.VariableModel],
    training_process: tff.learning.templates.LearningProcess,
    train2infer_frequency: int,
    data_element_spec: Any,
    bandit_data_fn: bandits_utils.BanditFnType,
    initial_model_path: Optional[str] = None,
) -> tff.templates.IterativeProcess:
  """Returns an iterative process for bandits simulation in TFF.

  The bandits process add the bandits inference functionality to online generate
  bandits data by `bandit_data_fn` for training with `training_process`.
  Every `train2infer_frequency` rounds, the model in `training_process` will be
  deployed (copied to `ServerState.delayed_inference_model`) for bandits
  inference.

  Args:
    model_fn: A no-arg function returns a `tff.learning.models.VariableModel`.
    training_process: A `tff.learning.templates.LearningProcess` for model
      training.
    train2infer_frequency: The frequency of deploying the model for inference.
    data_element_spec: The `data_element_spec` of the original dataset, i.e. the
      input of `bandit_data_fn`. It may be different from
      `model_fn().input_spec`, which should match the output dataset of
      `bandit_data_fn`.
    bandit_data_fn: A function that use a inference model to generate bandits
      data for training.
    initial_model_path: A path to load a saved model for initialization.
  """
  bandits_utils.check_data_element_spec(data_element_spec)

  tf_dataset_type = tff.SequenceType(data_element_spec)
  federated_dataset_type = tff.type_at_clients(tf_dataset_type)

  train_state_type = training_process.initialize.type_signature.result.member
  model_type = training_process.get_model_weights.type_signature.result
  server_state_type = ServerState(
      delayed_inference_model=model_type,
      train_state=train_state_type,
      round_num=tf.int32,
      average_reward=tf.float32,
  )
  federated_server_state_type = tff.type_at_server(server_state_type)

  @tff.tf_computation(model_type, tf_dataset_type)
  def bandit_data_comp(model_weights, tf_dataset):
    # TODO(b/222111606): explicitly pins the variables on CPU to avoid error in
    # TFF GPU simulation.
    with tf.device('/device:cpu:0'):
      model = model_fn()
    return bandit_data_fn(model, model_weights, tf_dataset)

  @tff.tf_computation(bandit_data_comp.type_signature.result)
  @tf.function
  def track_bandit_reward(bandits_dataset):
    num_examples = tf.constant(0, tf.int32)
    sum_rewards = tf.constant(0, tf.float32)
    sum_prob = tf.constant(0, tf.float32)
    for batch in bandits_dataset:
      batch_rewards = batch['y'][bandits_utils.BanditsKeys.reward]
      num_examples += tf.shape(batch_rewards)[0]
      sum_rewards += tf.math.reduce_sum(batch_rewards)
      sum_prob += tf.math.reduce_sum(batch['y'][bandits_utils.BanditsKeys.prob])
    num_examples_weights = tf.cast(num_examples, dtype=tf.float32)
    return collections.OrderedDict([
        (_REWARD_KEY, sum_rewards / num_examples_weights),
        (_PROB_KEY, sum_prob / num_examples_weights),
        (_WEIGHT_KEY, num_examples_weights),
    ])

  @tff.tf_computation(server_state_type, train_state_type, tf.int32, tf.float32)
  @tf.function
  def update_server_state(
      server_state, train_state, train2infer_frequency, round_reward
  ):
    round_num = server_state.round_num + 1
    round_num_float = tf.cast(round_num, tf.float32)
    average_reward = (
        server_state.average_reward * (round_num_float - 1.0) / round_num_float
        + round_reward / round_num_float
    )
    if tf.equal(tf.math.mod(round_num, train2infer_frequency), 0):
      # TODO(b/215566681): uses a single model for inference. Potentially
      # extends to multiple historical models, and different number of bandits
      # data can be generated depending on the deployment time.
      delayed_inference_model = train_state.global_model_weights
    else:
      delayed_inference_model = server_state.delayed_inference_model
    return ServerState(
        delayed_inference_model=delayed_inference_model,
        train_state=train_state,
        round_num=round_num,
        average_reward=average_reward,
    )

  @tff.federated_computation(
      federated_server_state_type, federated_dataset_type
  )
  def run_one_round(server_state, federated_dataset):
    """One round of bandits simulation.

    Online generates bandits data using `server_state.delayed_inference_model`
    and the input client datasets, and then runs one round of training_process

    Args:
      server_state: A `ServerState` with attributes (delayed_inference_model,
        train_state, round_num)
      federated_dataset: A set of client datasets for training.

    Returns:
      A tuple of (updated server_state, results such as metrics).
    """
    bandit_dataset = tff.federated_map(
        bandit_data_comp,
        (
            tff.federated_broadcast(server_state.delayed_inference_model),
            federated_dataset,
        ),
    )
    client_rewards = tff.federated_map(track_bandit_reward, bandit_dataset)
    round_reward = tff.federated_mean(
        value=client_rewards[_REWARD_KEY], weight=client_rewards[_WEIGHT_KEY]
    )
    round_prob = tff.federated_mean(
        value=client_rewards[_PROB_KEY], weight=client_rewards[_WEIGHT_KEY]
    )
    train_state, train_metrics = training_process.next(
        server_state.train_state, bandit_dataset
    )
    server_state = tff.federated_map(
        update_server_state,
        (
            server_state,
            train_state,
            tff.federated_value(train2infer_frequency, tff.SERVER),
            round_reward,
        ),
    )
    bandits_metric = collections.OrderedDict(
        round_rewards=round_reward,
        average_rewards=server_state.average_reward,
        round_prob=round_prob,
    )
    metrics = collections.OrderedDict(
        optimization=train_metrics, bandits=bandits_metric
    )
    return server_state, metrics

  if initial_model_path:

    @tff.tf_computation
    def load_model_weights(inner_state):
      keras_model = tf.keras.models.load_model(initial_model_path)
      model_weights = tff.learning.models.ModelWeights.from_model(keras_model)
      # TODO(b/227775900): explicit control_dependencies is necessary for TFF
      # computation without `tf.function`.
      with tf.control_dependencies(
          model_weights.trainable + model_weights.non_trainable
      ):
        return training_process.set_model_weights(inner_state, model_weights)

  else:

    @tff.tf_computation
    def load_model_weights(inner_state):
      return inner_state

  @tff.federated_computation
  def server_init_tff():
    """Orchestration logic for server model initialization."""
    inner_state = training_process.initialize()
    inner_state = tff.federated_map(load_model_weights, inner_state)
    model_weights = tff.federated_map(
        training_process.get_model_weights, inner_state
    )
    return tff.federated_zip(
        ServerState(
            delayed_inference_model=model_weights,
            train_state=inner_state,
            round_num=tff.federated_value(0, tff.SERVER),
            average_reward=tff.federated_value(0.0, tff.SERVER),
        )
    )

  return tff.templates.IterativeProcess(
      initialize_fn=server_init_tff, next_fn=run_one_round
  )
