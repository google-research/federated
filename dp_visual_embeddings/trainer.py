# Copyright 2022, Google LLC.
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
"""Library of embedding models in TFF simulation."""
from collections.abc import Callable
import enum
import functools
from typing import Any, Optional, Union

import tensorflow as tf
import tensorflow_federated as tff

from dp_visual_embeddings.algorithms import federated_partial
from dp_visual_embeddings.algorithms import process_with_pretrained
from dp_visual_embeddings.models import embedding_model
from dp_visual_embeddings.models import keras_utils
from dp_visual_embeddings.tasks.task_utils import EmbeddingTask
from dp_visual_embeddings.utils import export


_OptimizerType = Callable[[], tf.keras.optimizers.Optimizer]
_ScheduleOptimizerType = Callable[[float], tf.keras.optimizers.Optimizer]
_ModelFnType = Callable[[], Union[tff.learning.Model, embedding_model.Model]]
_ADAM_EPS = 1e-3


class ProcessType(enum.Enum):
  FEDAVG = enum.auto()
  FEDPARTIAL = enum.auto()


def get_process_types():
  return list(ProcessType)


class AggregatorType(enum.Enum):
  """Aggregator type for (differentially private) federated algorithms.

  Attributes:
    DPFTRL: Tree aggregation based DP-FTRL https://arxiv.org/abs/2103.00039.
    DPSGD: DP-FedAvg https://arxiv.org/abs/1710.06963
  """
  DPFTRL = enum.auto()
  DPSGD = enum.auto()


def get_aggregator_types():
  return list(AggregatorType)


def _check_nonnegative(value: Union[int, float], name: str = ''):
  if value < 0:
    raise ValueError(f'Got {value} for non negative input {name}.')


def _check_positive(value: Union[int, float], name: str = ''):
  if value <= 0:
    raise ValueError(f'Got {value} for positive input {name}.')


def _check_momentum(value: float):
  if value < 0 or value >= 1:
    raise ValueError(f'Got {value} for input momentum in [0, 1).')


def configure_optimizers(
    server_learning_rate: float = 1.,
    server_momentum: float = 0.,
    client_learning_rate: float = .1,
    client_momentum: float = 0.,
    client_opt: str = 'sgd',
) -> tuple[_OptimizerType, _OptimizerType]:
  """Configures server and client optimizers for Generalized FedAvg algorithms.

  For differentially private algorithms, default server optimzier to momentum
  SGD, and client optimizer to SGD.

  Args:
    server_learning_rate: A positive float for server learning rate.
    server_momentum: A positive float for server momentum.
    client_learning_rate: A positive float for client learning rate.
    client_momentum: A positive float for client momentum.
    client_opt: The type of client optimizer, 'sgd' or 'adam'.

  Returns:
    A tuple of no-arg callables that return a `tf.keras.optimizers.SGD`.
  """
  _check_nonnegative(server_learning_rate)
  _check_momentum(server_momentum)
  _check_nonnegative(client_learning_rate)
  _check_momentum(client_momentum)
  server_optimizer_fn = lambda: tf.keras.optimizers.SGD(  # pylint:disable=g-long-lambda,line-too-long
      server_learning_rate,
      momentum=server_momentum)
  if client_opt == 'sgd':
    client_optimizer_fn = lambda: tf.keras.optimizers.SGD(  # pylint:disable=g-long-lambda,line-too-long
        client_learning_rate,
        momentum=client_momentum)
  elif client_opt == 'adam':
    client_optimizer_fn = lambda: tf.keras.optimizers.Adam(  # pylint:disable=g-long-lambda,line-too-long
        client_learning_rate,
        beta_1=client_momentum,
        beta_2=0.99,
        epsilon=_ADAM_EPS)
  else:
    raise ValueError(f'Unknown client_opt: {client_opt}')
  return server_optimizer_fn, client_optimizer_fn


def configure_client_scheduled_optimizers(
    server_learning_rate: float = 1.,
    server_momentum: float = 0.,
    client_learning_rate: float = .1,
    client_momentum: float = 0.,
    client_opt: str = 'sgd',
) -> tuple[_OptimizerType, _ScheduleOptimizerType, Callable[[int], float]]:
  """Configures server and client optimizers for Generalized FedAvg algorithms.

  For differentially private algorithms, default server optimizer to momentum
  SGD, client optimizer to SGD, client optimizer learning rate to constant.

  Args:
    server_learning_rate: A positive float for server learning rate.
    server_momentum: A positive float for server learning rate.
    client_learning_rate: A positive float for client learning rate.
    client_momentum: A positive float for client momentum.
    client_opt: The type of client optimizer, 'sgd' or 'adam'.

  Returns:
    A tuple of (server_optimizer_fn, client_optimizer_fn, learning_rate_fn),
      where `server_optimizer_fn` is a no-arg function returning server
      optimizer; `learning_rate_fn` returns learning rate based on the round
      index, which can be used in `client_optimizer_fn`.
  """
  _check_nonnegative(server_learning_rate)
  _check_momentum(server_momentum)
  _check_nonnegative(client_learning_rate)
  _check_momentum(client_momentum)
  server_optimizer_fn = lambda: tf.keras.optimizers.SGD(  # pylint:disable=g-long-lambda,line-too-long
      server_learning_rate,
      momentum=server_momentum)
  learning_rate_fn = lambda round_num: client_learning_rate
  if client_opt == 'sgd':
    client_optimizer_fn = functools.partial(
        tf.keras.optimizers.SGD, momentum=client_momentum)
  elif client_opt == 'adam':
    client_optimizer_fn = functools.partial(
        tf.keras.optimizers.Adam,
        beta_1=client_momentum,
        beta_2=0.99,
        epsilon=_ADAM_EPS)
  else:
    raise ValueError(f'Unknown client_opt: {client_opt}')
  return server_optimizer_fn, client_optimizer_fn, learning_rate_fn


def configure_aggregator(
    *,
    aggregator_type: Optional[AggregatorType] = None,
    model_fn: Optional[_ModelFnType] = None,
    clip_norm: Optional[float] = None,
    noise_multiplier: Optional[float] = None,
    report_goal: Optional[int] = None,
    target_unclipped_quantile: Optional[float] = None,
    clip_learning_rate: float = 0.2,
    clipped_count_stddev: Optional[float] = None,
    noise_seed: Optional[int] = None,
) -> tff.aggregators.AggregationFactory:
  """Returns (differentially private) aggregator for federated algorithms.

  A few options:
    If `target_unclipped_quantile` is not `None`: DP-FTRL/DP-SGD with adaptive
      clipping described in
      "Differentially Private Learning with Adaptive Clipping".
    If `noise_multiplier` is not `None`: DP-FTRL/DP-SGD with fixed clip norm.
    If `clip_norm` is not `None`: use constant client-side clipping and
      unweighted aggregation.
    Else: a default `tff.learning.robust_aggregator` for clipping.

  Args:
    aggregator_type: (Optional) DPFTRL or DPSGD for differentially private
      federated algorithms.
    model_fn: (Optional) A no-arg function that returns a `tff.learning.Model`
      or a `embedding_model.Model`.
    clip_norm: The l2 clip norm of client delta for differential privacy. Must
      be positive when `noise_multiplier` is not `None`.
    noise_multiplier: The noise multiplier for differential privacy. The noise
      std for the sum of client deltas is equal to `clip_norm*noise_multiplier`.
      If `None`, no differential privacy mechanism is applied.
    report_goal: The report goal/minimum expected clients per round. Must be
      positive when `noise_multiplier` is not `None`.
    target_unclipped_quantile: The desired quantile of updates which should be
      unclipped.
    clip_learning_rate: The learning rate for the clipping norm adaptation. With
      geometric updating, a rate of r means that the clipping norm will change
      by a maximum factor of exp(r) at each round.
    clipped_count_stddev: The stddev of the noise added to the clipped_count. If
      `None`, set to `clients_per_round / 20`.
    noise_seed: Seed for random noise generation. If `None` and
      `noise_multiplier` is not `None`, non-deterministic noise will be used.
  """
  if noise_multiplier is not None:
    _check_nonnegative(noise_multiplier, 'noise_multiplier')
  if clip_norm is not None:
    _check_positive(clip_norm, 'clip_norm')
  if report_goal is not None:
    _check_positive(report_goal, 'report_goal')
  if target_unclipped_quantile is not None:
    if aggregator_type is None:
      raise ValueError('Please specify `aggregator_type`.')
    elif aggregator_type == AggregatorType.DPSGD:
      return tff.aggregators.DifferentiallyPrivateFactory.gaussian_adaptive(
          noise_multiplier=noise_multiplier,
          clients_per_round=report_goal,
          initial_l2_norm_clip=clip_norm,
          target_unclipped_quantile=target_unclipped_quantile,
          learning_rate=clip_learning_rate,
          clipped_count_stddev=clipped_count_stddev)
    elif aggregator_type == AggregatorType.DPFTRL:
      raise ValueError('DP-FTRL with adaptive clipping is not supported.')
    else:
      raise ValueError(f'Unknown aggregator type {aggregator_type}')
  elif noise_multiplier is not None:
    if aggregator_type is None:
      raise ValueError('Please specify `aggregator_type`.')
    elif aggregator_type == AggregatorType.DPFTRL:
      model_weight_specs = tff.framework.type_to_tf_tensor_specs(
          tff.learning.models.weights_type_from_model(model_fn).trainable)
      return tff.aggregators.DifferentiallyPrivateFactory.tree_aggregation(
          noise_multiplier=noise_multiplier,
          clients_per_round=report_goal,
          l2_norm_clip=clip_norm,
          record_specs=model_weight_specs,
          noise_seed=noise_seed,
          use_efficient=True)
    elif aggregator_type == AggregatorType.DPSGD:
      return tff.aggregators.DifferentiallyPrivateFactory.gaussian_fixed(
          noise_multiplier=noise_multiplier,
          clients_per_round=report_goal,
          clip=clip_norm)
    else:
      raise ValueError(f'Unknown aggregator type {aggregator_type}')
  elif clip_norm is not None:
    if aggregator_type is not None:
      raise ValueError('Clipping only, set `aggregator_type=None`.')
    return tff.aggregators.robust.clipping_factory(
        clipping_norm=clip_norm,
        inner_agg_factory=tff.aggregators.UnweightedMeanFactory())
  else:
    if aggregator_type is not None:
      raise ValueError('Clipping only, set `aggregator_type=None`.')
    return tff.learning.robust_aggregator(
        zeroing=False,
        clipping=True,
        weighted=False,
        debug_measurements_fn=None)


def build_train_process(
    process_type: ProcessType,
    *,
    aggregator: tff.aggregators.AggregationFactory,
    task: EmbeddingTask,
    server_optimizer: _OptimizerType,
    client_optimizer: Union[_OptimizerType, _ScheduleOptimizerType],
    client_learning_rate_fn: Optional[Callable[[int], float]] = None,
    pretrained_model_path: Optional[str] = None,
    head_lr_scale: float = 1.,
    reconst_iters: Optional[int] = None,
) -> tff.learning.templates.LearningProcess:
  """Returns federated training iterative process.

  Args:
    process_type: A `ProcessType` indicates process type.
    aggregator: A `tff.aggregators.AggregationFactory` for (differentially
      private) federated algorithms.
    task: A `EmbeddingTask` to provide model function for iterative process.
    server_optimizer: A no-arg callable that returns a `tf.keras.Optimizer`.
    client_optimizer: A no-arg callable that returns a `tf.keras.Optimizer`, or
      a callable accepts learning rate as input and returns a
      `tf.keras.Optimizer`.
    client_learning_rate_fn: A callable accepts `round_num` as input and returns
      a learning rate for that round.
    pretrained_model_path: Optional path to model saved by Keras `model.save`.
      If None, no saved model will be loaded.
    head_lr_scale: Use head_lr_scale to scale the learning rate for updating
      the local variables (head of the model).
    reconst_iters: If not `None`, first optimize the head of the model
      for `reconst_iters` iterations before training the Backbone.
  """
  if process_type == ProcessType.FEDAVG:
    if client_learning_rate_fn is not None:
      raise ValueError('`client_learning_rate_fn` is not supported for FedAvg.')
    if head_lr_scale != 1.:
      raise ValueError('`head_lr_scale` is not supported for FedAvg.')
    process = tff.learning.algorithms.build_unweighted_fed_avg(
        task.federated_model_fn,
        server_optimizer_fn=server_optimizer,
        client_optimizer_fn=client_optimizer,
        model_aggregator=aggregator,
        use_experimental_simulation_loop=True)
    return process_with_pretrained.build_fedavg_process_with_pretrained(
        process, pretrained_model_path=pretrained_model_path)

  elif process_type == ProcessType.FEDPARTIAL:
    if head_lr_scale is None:
      head_lr_scale = 1
    process = federated_partial.build_unweighted_averaging_with_optimizer_schedule(
        model_fn=task.embedding_model_fn,
        client_learning_rate_fn=client_learning_rate_fn,
        client_optimizer_fn=client_optimizer,
        server_optimizer_fn=server_optimizer,
        model_aggregator=aggregator,
        head_lr_scale=head_lr_scale,
        reconst_iters=reconst_iters)

    def pretrained_model_fn():
      model = task.inference_model
      return keras_utils.EmbeddingModel(
          model=model,
          global_variables=tff.learning.ModelWeights.from_model(model),
          client_variables=tff.learning.ModelWeights([], []))

    return process_with_pretrained.build_process_with_pretrained(
        process,
        pretrained_model_fn=pretrained_model_fn,
        pretrained_model_path=pretrained_model_path)

  else:
    raise ValueError(f'Unsupported process type:{process_type}')


def build_eval_fn(
    process_type: ProcessType, *, task: EmbeddingTask,
    train_process: tff.learning.templates.LearningProcess) -> tff.Computation:
  """Returns evaluation function, which is used for federated evaluation.

  Args:
    process_type: A `ProcessType` indicates process type.
    task: A `EmbeddingTask` to provide model function for iterative process.
    train_process: The TFF iterative process for training.
  """
  if process_type == ProcessType.FEDAVG:
    federated_eval = tff.learning.build_federated_evaluation(
        task.federated_model_fn, use_experimental_simulation_loop=True)
  elif process_type == ProcessType.FEDPARTIAL:
    federated_eval = tff.learning.build_federated_evaluation(
        task.embedding_model_fn, use_experimental_simulation_loop=True)
  else:
    raise ValueError(f'Unsupported process type:{process_type}')

  @tff.federated_computation(train_process.state_type,
                             federated_eval.type_signature.parameter[1])
  def evaluation_fn(state, evaluation_data):
    return federated_eval(
        tff.federated_map(train_process.get_model_weights, state),
        evaluation_data)

  return evaluation_fn


def build_export_fn(
    process_type: ProcessType, *, task: EmbeddingTask,
    train_process: tff.learning.templates.LearningProcess
) -> Callable[[Any, str], None]:
  """Returns export function, which is used for centralized evaluation.

  Args:
    process_type: A `ProcessType` indicates process type.
    task: A `EmbeddingTask` to provide model function for iterative process.
    train_process: The TFF iterative process for training.
  """
  if process_type == ProcessType.FEDAVG:
    model_fn = task.federated_model_fn
  elif process_type == ProcessType.FEDPARTIAL:
    model_fn = task.embedding_model_fn
  else:
    raise ValueError(f'Unsupported process type:{process_type}')

  def export_fn(state, export_dir):
    model_weights = train_process.get_model_weights(state)
    export.export_state(model_fn, model_weights,
                        task.inference_model,
                        export_dir)

  return export_fn


def train_and_eval(
    task: EmbeddingTask,
    *,
    train_process: tff.learning.templates.LearningProcess,
    evaluation_fn: Optional[tff.Computation],
    export_fn: Callable[[Any], None],
    total_rounds: int,
    clients_per_round: int,
    rounds_per_eval: int,
    program_state_manager: Optional[tff.program.FileProgramStateManager] = None,
    rounds_per_saving_program_state: int = 50,
    metrics_managers: Optional[list[tff.program.ReleaseManager]] = None) -> Any:
  """Train and evaluate embedding models in TFF simulation.

  Args:
    task: A task that defines data and model for TFF simulation.
    train_process: A TFF iterative process for training.
    evaluation_fn: A function evaluates iterative process state on given data.
    export_fn: When provided a directory location, a function exports model with
      trained weights provided by `train_process` state.
    total_rounds: A positive integer for the total number of training rounds.
    clients_per_round: A positive integer for clients per round.
    rounds_per_eval: A positive integer for the frequency of evaluation.
    program_state_manager: An optional `tff.program.FileProgramStateManager`
      used to periodically save program state of the iterative process state.
    rounds_per_saving_program_state: How often to write program state.
    metrics_managers: An optional list of `tff.program.ReleaseManager` objects
      used to save training metrics throughout the simulation.

  Returns:
    The final state of `train_process` after training.
  """
  _check_nonnegative(total_rounds)
  _check_nonnegative(clients_per_round)
  _check_nonnegative(rounds_per_eval)

  train_process = tff.simulation.compose_dataset_computation_with_learning_process(
      task.datasets.train_dataset_computation, train_process)
  if evaluation_fn is not None:
    evaluation_fn = tff.simulation.compose_dataset_computation_with_computation(
        task.datasets.validation_dataset_computation, evaluation_fn)

  def client_selection_fn(round_num):
    del round_num
    return task.datasets.sample_train_client_ids(num_clients=clients_per_round)

  def evaluation_selection_fn(round_num):
    del round_num
    # Use `replace=True` to handle the case when the validation set is small.
    return task.datasets.sample_validation_client_ids(
        num_clients=clients_per_round, replace=True)

  server_state = tff.simulation.run_training_process(
      training_process=train_process,
      training_selection_fn=client_selection_fn,
      total_rounds=total_rounds,
      evaluation_fn=evaluation_fn,
      evaluation_selection_fn=evaluation_selection_fn,
      rounds_per_evaluation=rounds_per_eval,
      program_state_manager=program_state_manager,
      rounds_per_saving_program_state=rounds_per_saving_program_state,
      metrics_managers=metrics_managers)

  export_fn(server_state)
  return server_state
