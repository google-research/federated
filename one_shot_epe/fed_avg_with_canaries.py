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
"""Custom client work for generating canary updates."""

import collections
from collections.abc import Callable
from typing import Any, NamedTuple, Optional, Union

import tensorflow as tf
import tensorflow_federated as tff

from one_shot_epe import dot_product_utils

_ReduceFnCallable = Callable[[Any, tf.Tensor], Any]

_CANARY_SCALING_FACTOR = 10000


def _dataset_reduce(
    reduce_fn: _ReduceFnCallable,
    dataset: tf.data.Dataset,
    initial_state_fn: Callable[[], Any] = lambda: tf.constant(0),
) -> Any:
  """Reduces dataset by calling `tf.data.Dataset.reduce`."""
  return dataset.reduce(initial_state=initial_state_fn(), reduce_func=reduce_fn)


def _iter_reduce(
    reduce_fn: _ReduceFnCallable,
    dataset: tf.data.Dataset,
    initial_state_fn: Callable[[], Any] = lambda: tf.constant(0),
) -> Any:
  """Reduces dataset with python iterator for performant GPU simulation."""
  update_state = initial_state_fn()
  for batch in iter(dataset):
    update_state = reduce_fn(update_state, batch)
  return update_state


def _build_client_update(
    model_fn: Callable[[], tff.learning.models.VariableModel],
    use_experimental_simulation_loop: bool = False,
) -> ...:
  """Returns client update function that branches logic based on client type.

  On real clients, the returned client update is the typical FedAvg update. On
  canary clients, it is that canary's update, which is sampled from the unit
  sphere deterministically based on the canary ID.

  Args:
    model_fn: Function to instantiate the model.
    use_experimental_simulation_loop: Controls the reduce loop function for
      input dataset. An experimental reduce loop is used for simulation. It is
      currently necessary to set this flag to True for performant GPU
      simulations.

  Returns:
    A tf.function.
  """
  model = model_fn()
  reduce = _iter_reduce if use_experimental_simulation_loop else _dataset_reduce

  @tf.function
  def client_update(optimizer, initial_weights, data, client_id, canary_seed):
    model_weights = tff.learning.models.ModelWeights.from_model(model).trainable
    split_id = tf.strings.split(client_id, ':', 1)
    client_type = split_id[0]

    if tf.math.equal(client_type, 'real'):
      tf.nest.map_structure(
          lambda a, b: a.assign(b), model_weights, initial_weights.trainable
      )

      def reduce_fn(num_examples_sum, batch):
        """Trains a `tff.learning.models.VariableModel` on a batch of data."""
        with tf.GradientTape() as tape:
          output = model.forward_pass(batch, training=True)

        gradients = tape.gradient(output.loss, model_weights)
        grads_and_vars = zip(gradients, model_weights)
        optimizer.apply_gradients(grads_and_vars)

        if output.num_examples is None:
          num_examples_sum += tf.shape(output.predictions, out_type=tf.in64)[0]
        else:
          num_examples_sum += tf.cast(output.num_examples, tf.int64)

        return num_examples_sum

      def initial_state_for_reduce_fn():
        return tf.zeros(shape=[], dtype=tf.int64)

      reduce(reduce_fn, data, initial_state_for_reduce_fn)
      client_update = tf.nest.map_structure(
          tf.subtract, initial_weights.trainable, model_weights
      )

    else:
      canary_id = tf.strings.to_number(split_id[1], out_type=tf.int32)
      canary_update = dot_product_utils.packed_canary(
          initial_weights.trainable, canary_id, canary_seed
      )

      # Scale by large value to ensure that the update is clipped.
      client_update = tf.nest.map_structure(
          lambda x: tf.multiply(x, _CANARY_SCALING_FACTOR), canary_update
      )

    model_output = model.report_local_unfinalized_metrics()
    client_weight = tf.constant(1.0, tf.float32)

    return (
        tff.learning.templates.ClientResult(
            update=client_update, update_weight=client_weight
        ),
        model_output,
    )

  return client_update


def _build_canary_client_work(
    model_fn: Callable[[], tff.learning.models.VariableModel],
    optimizer_fn: Callable[[], tf.keras.optimizers.legacy.Optimizer],
    canary_seed: int,
    use_experimental_simulation_loop: bool = False,
) -> tff.learning.templates.ClientWorkProcess:
  """Creates a `ClientWorkProcess` for federated averaging with canaries.

    The `initialize_fn` and `next_fn` of the returned `ClientWorkProcess` have
    the following type signatures:
    ```
      - initialize_fn: ( -> S@SERVER)
      - next_fn: (<S@SERVER,
                   (A,I)@CLIENTS,
                   {D*}@CLIENTS>
                  ->
                  <state=S@SERVER,
                   result=ClientResult(B, C)@CLIENTS,
                   measurements=M@SERVER>)
    ```
    Here:
    - `S` is the server state,
    - `A` is a client's model weights,
    - `I` is the client ID as a string,
    - `D*` is a `tff.SequenceType` of client data,
    - `B` is the client update, and
    - `C` is the client weight.

  Args:
    model_fn: A no-arg function that returns a
      `tff.learning.models.VariableModel`. This method must *not* capture
      TensorFlow tensors or variables and use them. The model must be
      constructed entirely from scratch on each invocation, returning the same
      pre-constructed model each call will result in an error.
    optimizer_fn: A no-arg callable that returns a `tf.keras.legacy.Optimizer`.
    canary_seed: Global seed for canaries.
    use_experimental_simulation_loop: Controls the reduce loop function for
      input dataset. An experimental reduce loop is used for simulation. It is
      currently necessary to set this flag to True for performant GPU
      simulations.

  Returns:
    A `ClientWorkProcess` for federated averaging with canaries.
  """
  if not callable(model_fn):
    raise TypeError(f'model_fn must be a callable. Found {type(model_fn)}.')
  if not callable(optimizer_fn):
    raise TypeError(
        'optimizer_fn must be a no-arg callable returning a '
        f'tf.keras.optimizers.legacy.Optimizer. Found {type(optimizer_fn)}.'
    )

  metrics_aggregator = tff.learning.metrics.sum_then_finalize

  with tf.Graph().as_default():
    # Wrap model construction in a graph to avoid polluting the global context
    # with variables created for this model.
    model = model_fn()
    unfinalized_metrics_type = tff.types.type_from_tensors(
        model.report_local_unfinalized_metrics()
    )
    metrics_aggregation_fn = metrics_aggregator(
        model.metric_finalizers(),
        unfinalized_metrics_type,  # pytype: disable=wrong-arg-types
    )

  data_type = tff.SequenceType(model.input_spec)
  client_id_type = tff.TensorType(tf.string)
  model_weights_type = tff.learning.models.weights_type_from_model(model)
  canary_seed_type = tff.TensorType(tf.int64)

  @tff.tf_computation
  def canary_seed_fn():
    return tf.constant(canary_seed, tf.int64)

  @tff.federated_computation
  def init_fn():
    # State is simply the canary seed.
    return tff.federated_zip(
        collections.OrderedDict(
            canary_seed=tff.federated_eval(canary_seed_fn, tff.SERVER)
        )
    )

  @tff.tf_computation(
      model_weights_type, data_type, client_id_type, canary_seed_type
  )
  def client_update_computation(model_weights, dataset, client_id, canary_seed):
    optimizer = optimizer_fn()
    client_update = _build_client_update(
        model_fn=model_fn,
        use_experimental_simulation_loop=use_experimental_simulation_loop,
    )
    return client_update(
        optimizer, model_weights, dataset, client_id, canary_seed
    )

  @tff.federated_computation(
      init_fn.type_signature.result,
      tff.type_at_clients((model_weights_type, client_id_type)),
      tff.type_at_clients(data_type),
  )
  def next_fn(state, model_weights_and_client_ids, client_data):
    model_weights, client_ids = model_weights_and_client_ids
    canary_seed_on_clients = tff.federated_broadcast(state['canary_seed'])
    client_result, model_outputs = tff.federated_map(
        client_update_computation,
        (model_weights, client_data, client_ids, canary_seed_on_clients),
    )
    train_metrics = metrics_aggregation_fn(model_outputs)
    measurements = tff.federated_zip(
        collections.OrderedDict(train=train_metrics)
    )
    return tff.templates.MeasuredProcessOutput(
        state, client_result, measurements
    )

  return tff.learning.templates.ClientWorkProcess(init_fn, next_fn)


class CanaryLearningProcess(tff.templates.IterativeProcess):
  """A `tff.templates.IterativeProcess` for training with canary clients.

  This is similar to `tff.learning.templates.LearningProcess` except the `next`
  function takes three arguments for (state, client_ids, client_data).
  """

  def __init__(
      self,
      initialize_fn: tff.Computation,
      next_fn: tff.Computation,
      get_model_weights: tff.Computation,
      set_model_weights: tff.Computation,
      next_is_multi_arg: Optional[bool] = None,
  ):
    super().__init__(initialize_fn, next_fn, next_is_multi_arg)
    self._get_model_weights = get_model_weights
    self._set_model_weights = set_model_weights

  @property
  def get_model_weights(self) -> tff.Computation:
    return self._get_model_weights

  @property
  def set_model_weights(self) -> tff.Computation:
    return self._set_model_weights


class CanaryLearningProcessState(NamedTuple):
  global_model_weights: Any
  distributor: Any
  client_work: Any
  aggregator: Any
  finalizer: Any
  max_canary_model_delta_cosines: Any
  max_unseen_canary_model_delta_cosines: Any


def build_canary_learning_process(
    model_fn: Callable[[], tff.learning.models.VariableModel],
    dataset_computation: tff.Computation,
    canary_seed: int,
    num_canaries: int,
    num_unseen_canaries: int,
    update_aggregator_factory: tff.aggregators.UnweightedAggregationFactory,
    client_optimizer_fn: Callable[[], tf.keras.optimizers.legacy.Optimizer],
    server_optimizer_fn: Union[
        tff.learning.optimizers.Optimizer,
        Callable[[], tf.keras.optimizers.Optimizer],
    ],
    use_experimental_simulation_loop: bool = False,
) -> CanaryLearningProcess:
  """Creates a learning process using Canary client work.

  Args:
    model_fn: A no-arg function that returns a
      `tff.learning.models.VariableModel`.
    dataset_computation: A `tff.Computation` that accepts client_ids and returns
      preprocessed data.
    canary_seed: The global seed to used to generate canary updates. If None,
      randomness is seeded nondeterministically.
    num_canaries: The number of canaries.
    num_unseen_canaries: The number of unseen canaries.
    update_aggregator_factory: A `tff.aggregartors.UnweightedAggregationFectory`
      for aggregating updates.
    client_optimizer_fn: A callable returning a
      `tf.keras.optimizers.legacy.Optimizer` to use for optimization on clients.
    server_optimizer_fn: A callable returning a
      `tf.keras.optimizers.legacy.Optimizer` to use for optimization on the
      server.
    use_experimental_simulation_loop: Controls the reduce loop function for
      input dataset. An experimental reduce loop is used for simulation. It is
      currently necessary to set this flag to True for performant GPU
      simulations.

  Returns:
    A `CanaryLearningProcess` for federated averaging with canary clients.
  """

  @tff.tf_computation()
  def initial_model_weights_fn():
    return tff.learning.models.ModelWeights.from_model(model_fn())

  client_work = _build_canary_client_work(
      model_fn,
      client_optimizer_fn,
      canary_seed,
      use_experimental_simulation_loop,
  )
  client_work_output_type = client_work.next.type_signature.result.result.member  # pytype: disable=attribute-error
  aggregator = tff.aggregators.as_weighted_aggregator(
      update_aggregator_factory
  ).create(
      client_work_output_type.update, client_work_output_type.update_weight
  )
  model_weights_type = initial_model_weights_fn.type_signature.result
  finalizer = tff.learning.templates.build_apply_optimizer_finalizer(
      server_optimizer_fn, model_weights_type
  )

  @tff.tf_computation()
  def initial_max_cosines_fn():
    return tf.constant(-1.0, tf.float32, (num_canaries,))

  @tff.tf_computation()
  def initial_max_unseen_cosines_fn():
    return tf.constant(-1.0, tf.float32, (num_unseen_canaries,))

  @tff.federated_computation()
  def init_fn():
    initial_model_weights = tff.federated_eval(
        initial_model_weights_fn, tff.SERVER
    )
    unused_distributor_state = tff.federated_value((), tff.SERVER)
    max_cosines = tff.federated_eval(initial_max_cosines_fn, tff.SERVER)
    max_unseen_cosines = tff.federated_eval(
        initial_max_unseen_cosines_fn, tff.SERVER
    )
    return tff.federated_zip(
        CanaryLearningProcessState(
            initial_model_weights,
            unused_distributor_state,
            client_work.initialize(),
            aggregator.initialize(),
            finalizer.initialize(),
            max_cosines,
            max_unseen_cosines,
        )
    )

  state_type = init_fn.type_signature.result
  client_id_type = tff.types.at_clients(tff.TensorType(tf.string))

  @tff.tf_computation()
  def model_delta_cosines(weights, new_weights, canary_seed):
    """Computes cosines with observed and unobserved canaries."""
    model_deltas = tf.nest.map_structure(tf.subtract, new_weights, weights)
    return [
        dot_product_utils.compute_negative_cosines_with_all_canaries(
            model_deltas, num_canaries, canary_seed, offset=0
        ),
        dot_product_utils.compute_negative_cosines_with_all_canaries(
            model_deltas, num_unseen_canaries, canary_seed, offset=num_canaries
        ),
    ]

  @tff.tf_computation()
  def get_model_delta_norm(weights, new_weights):
    model_delta = tf.nest.map_structure(tf.subtract, new_weights, weights)
    return tf.linalg.global_norm(tf.nest.flatten(model_delta))

  @tff.tf_computation()
  def max_cosines(old_max_cosines, new_cosines):
    return tf.maximum(old_max_cosines, new_cosines)

  @tff.federated_computation(state_type, client_id_type)
  def next_fn(state, client_ids):
    client_data = tff.federated_map(dataset_computation, client_ids)
    weights_and_client_ids = tff.federated_zip(
        (tff.federated_broadcast(state.global_model_weights), client_ids)
    )
    client_work_output = client_work.next(
        state.client_work, weights_and_client_ids, client_data
    )
    aggregator_output = aggregator.next(
        state.aggregator,
        client_work_output.result.update,
        client_work_output.result.update_weight,
    )
    finalizer_output = finalizer.next(
        state.finalizer, state.global_model_weights, aggregator_output.result
    )
    new_cosines = tff.federated_map(
        model_delta_cosines,
        (
            state.global_model_weights.trainable,
            finalizer_output.result.trainable,
            state.client_work['canary_seed'],
        ),
    )
    new_max_cosines = tff.federated_map(
        max_cosines,
        (state.max_canary_model_delta_cosines, new_cosines[0]),
    )
    new_unseen_max_cosines = tff.federated_map(
        max_cosines,
        (state.max_unseen_canary_model_delta_cosines, new_cosines[1]),
    )

    model_delta_norm = tff.federated_map(
        get_model_delta_norm,
        (
            state.global_model_weights.trainable,
            finalizer_output.result.trainable,
        ),
    )

    new_state = tff.federated_zip(
        CanaryLearningProcessState(
            finalizer_output.result,
            state.distributor,
            client_work_output.state,
            aggregator_output.state,
            finalizer_output.state,
            new_max_cosines,
            new_unseen_max_cosines,
        )
    )
    metrics = tff.federated_zip(
        collections.OrderedDict(
            client_work=client_work_output.measurements,
            aggregator=aggregator_output.measurements,
            finalizer=finalizer_output.measurements,
            model_delta_norm=model_delta_norm,
        )
    )

    return tff.learning.templates.LearningProcessOutput(new_state, metrics)

  @tff.tf_computation(state_type.member)
  def get_model_weights(state):
    return state.global_model_weights

  @tff.tf_computation(state_type.member, state_type.member.global_model_weights)
  def set_model_weights(state, model_weights):
    return tff.learning.templates.LearningAlgorithmState(
        global_model_weights=model_weights,
        distributor=state.distributor,
        client_work=state.client_work,
        aggregator=state.aggregator,
        finalizer=state.finalizer,
    )

  return CanaryLearningProcess(
      init_fn, next_fn, get_model_weights, set_model_weights, True
  )
