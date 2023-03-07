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
"""Federated trainer."""
from collections.abc import Callable
import enum
import functools
import os
from typing import Any, Optional, Union
from absl import logging

import tensorflow as tf
import tensorflow_federated as tff

from bandits import bandits_process
from bandits import bandits_utils
from bandits.algs import epsilon_greedy
from bandits.algs import falcon
from bandits.algs import softmax_sampling
from bandits.tasks import emnist
from bandits.tasks import stackoverflow
from bandits.tasks import task_utils

_OptimizerType = Callable[[], tf.keras.optimizers.Optimizer]


class TaskType(enum.Enum):
  EMNIST_DIGITS = enum.auto()
  EMNIST_CHARS = enum.auto()
  EMNIST10_LINEAR = enum.auto()
  EMNIST62_LINEAR = enum.auto()
  STACKOVERFLOW_TAG = enum.auto()


def get_task_types():
  return list(TaskType)


class BanditsType(enum.Enum):
  """Type of bandits algorithms.

  Attributes:
    EPSILON_GREEDY: Epsilon greedy algorithm with epsilon probability for
      exploration (random action selection). Use mean square regression loss.
    EPSILON_GREEDY_CE: Epsilon greedy algorithm with epsilon probability for
      exploration (random action selection). Use binary cross entropy loss.
    EPSILON_GREEDY_UNWEIGHT: Epsilon greedy algorithm with epsilon probability
      for exploration (random action selection). Use mean square regression loss
      without weighting by probability.
    EPSILON_GREEDY_CE_UNW: Epsilon greedy algorithm with epsilon probability for
      exploration (random action selection). Use binary cross entropy loss
      without weighting by probability.
    FALCON: FALCON algorithm with mean square regression loss.
    SUPERVISED_MSE: Supervised mean square regression loss where the rewards for
      all actions of a given context can be observed.
    SUPERVISED_CE: Supervised cross entropy loss where the rewards for all
      actions of a given context can be observed.
    SOFTMAX: Sampling probability for exploration is based on the softmax
      function.
  """

  EPSILON_GREEDY = enum.auto()
  EPSILON_GREEDY_CE = enum.auto()
  EPSILON_GREEDY_UNWEIGHT = enum.auto()
  EPSILON_GREEDY_CE_UNW = enum.auto()
  FALCON = enum.auto()
  SUPERVISED_MSE = enum.auto()
  SUPERVISED_CE = enum.auto()
  SOFTMAX = enum.auto()


def get_bandits_types():
  return list(BanditsType)


class DistShiftType(enum.Enum):
  """Type of distribution shift.

  For EMNIST and StackOverflow, we consider different data distribution for the
  initial model training, and the later bandits training.

  For EMNIST-62, the `INIT` distribution assigns inaccurate labels 10-35
  (represent A-Z) to the images of a-z (original integer labels 36-61). The
  `BANDITS` distirbution will use the following rewards structure: 1 for correct
  actions, 0.5 for inaccurately recognize a-z as 10-35, 0 for other incorrect
  actions.

  For StackOverflow, the `INIT` distribution can only observe the top 30 tags,
  while the `BANDITS` distribtuion will observe the top 50 tags, and use the
  rewards inverse propotional to the frequency of the tags.
  """

  INIT = enum.auto()
  BANDITS = enum.auto()


def get_distribution_types():
  return list(DistShiftType)


class AggregatorType(enum.Enum):
  """Aggregator type for (differentially private) federated algorithms.

  Attributes:
    DPFTRL: Tree aggregation based DP-FTRL https://arxiv.org/abs/2103.00039.
    DPSGD: DP-FedAvg https://arxiv.org/abs/1710.06963
    NONPRIVATE: Non-private aggregator, usually a tff.learning.
  """

  DPFTRL = enum.auto()
  DPSGD = enum.auto()


def get_aggregator_types():
  return list(AggregatorType)


def _check_positive(value: Union[int, float], name: str):
  if value <= 0:
    raise ValueError(f'Got {value} for positive input {name}.')


def _check_nonnegative(value: Union[int, float], name: str):
  if value < 0:
    raise ValueError(f'Got {value} for nonnegative input {name}.')


def _check_momentum(value: float, name: str):
  if value < 0 or value >= 1:
    raise ValueError(f'Got {value} for input {name} in [0, 1).')


def _get_task_data(
    task: TaskType,
    *,
    train_client_batch_size: int = 16,
    test_client_batch_size: int = 128,
    train_client_epochs_per_round: int = 1,
    train_shuffle_buffer_size: Optional[int] = None,
    train_client_max_elements: Optional[int] = None,
    use_synthetic_data: bool = False,
    stackoverflow_vocab_size: int = stackoverflow.DEFAULT_WORD_VOCAB_SIZE,
    stackoverflow_tag_size: int = stackoverflow.DEFAULT_TAG_VOCAB_SIZE,
    distribution_shift: Optional[DistShiftType] = None,
    population_client_selection: Optional[str] = None,
) -> tff.simulation.baselines.BaselineTaskDatasets:
  """Returns `BaselineTaskDatasets` for bandits inference simulation."""
  if task in [TaskType.EMNIST_DIGITS, TaskType.EMNIST10_LINEAR]:
    if distribution_shift is not None:
      raise ValueError(
          f'EMNIST-10 does not support distribution shift {distribution_shift} '
          'Use None or EMNIST-62.'
      )
    task_dataset = emnist.create_emnist_preprocessed_datasets(
        train_client_batch_size=train_client_batch_size,
        test_client_batch_size=test_client_batch_size,
        train_client_epochs_per_round=train_client_epochs_per_round,
        train_shuffle_buffer_size=train_shuffle_buffer_size,
        train_client_max_elements=train_client_max_elements,
        only_digits=True,
        use_synthetic_data=use_synthetic_data,
        population_client_selection=population_client_selection,
    )
  elif task in [TaskType.EMNIST_CHARS, TaskType.EMNIST62_LINEAR]:
    if (
        distribution_shift is not None
        and distribution_shift == DistShiftType.INIT
    ):
      label_distribution_shift = True
    else:
      label_distribution_shift = False
    task_dataset = emnist.create_emnist_preprocessed_datasets(
        train_client_batch_size=train_client_batch_size,
        test_client_batch_size=test_client_batch_size,
        train_client_epochs_per_round=train_client_epochs_per_round,
        train_shuffle_buffer_size=train_shuffle_buffer_size,
        train_client_max_elements=train_client_max_elements,
        only_digits=False,
        use_synthetic_data=use_synthetic_data,
        label_distribution_shift=label_distribution_shift,
        population_client_selection=population_client_selection,
    )
  elif task == TaskType.STACKOVERFLOW_TAG:
    if (
        distribution_shift is not None
        and distribution_shift == DistShiftType.INIT
    ):
      label_distribution_shift = True
    else:
      label_distribution_shift = False
    task_dataset = stackoverflow.create_stackoverflow_preprocessed_datasets(
        train_client_batch_size=train_client_batch_size,
        test_client_batch_size=test_client_batch_size,
        train_client_epochs_per_round=train_client_epochs_per_round,
        train_shuffle_buffer_size=train_shuffle_buffer_size,
        train_client_max_elements=train_client_max_elements,
        word_vocab_size=stackoverflow_vocab_size,
        tag_vocab_size=stackoverflow_tag_size,
        use_synthetic_data=use_synthetic_data,
        label_distribution_shift=label_distribution_shift,
        population_client_selection=population_client_selection,
    )
  else:
    raise ValueError(f'Unrecongnized task type {task} for data.')
  return task_dataset


def _get_model_fn(
    task: TaskType,
    bandits: BanditsType,
    bandits_data_spec: Any,
    stackoverflow_vocab_size: int = stackoverflow.DEFAULT_WORD_VOCAB_SIZE,
    stackoverflow_tag_size: int = stackoverflow.DEFAULT_TAG_VOCAB_SIZE,
) -> Callable[[], tff.learning.models.VariableModel]:
  """Returns `model_fn` for bandits training simulation."""
  if task == TaskType.EMNIST_DIGITS:
    num_arms = 10
    create_model_fn = functools.partial(
        emnist.create_emnist_bandits_model_fn, only_digits=True
    )
  elif task == TaskType.EMNIST_CHARS:
    num_arms = 62
    create_model_fn = functools.partial(
        emnist.create_emnist_bandits_model_fn, only_digits=False
    )
  elif task == TaskType.EMNIST10_LINEAR:
    num_arms = 10
    create_model_fn = functools.partial(
        emnist.create_emnist_bandits_model_fn,
        only_digits=True,
        model=emnist.ModelType.LINEAR,
    )
  elif task == TaskType.EMNIST62_LINEAR:
    num_arms = 62
    create_model_fn = functools.partial(
        emnist.create_emnist_bandits_model_fn,
        only_digits=False,
        model=emnist.ModelType.LINEAR,
    )
  elif task == TaskType.STACKOVERFLOW_TAG:
    num_arms = stackoverflow_tag_size
    create_model_fn = functools.partial(
        stackoverflow.create_stackoverflow_bandits_model_fn,
        input_size=stackoverflow_vocab_size,
        output_size=num_arms,
    )
  else:
    raise ValueError(f'Unrecongnized task type {task} for model.')

  if bandits == BanditsType.EPSILON_GREEDY:
    loss = task_utils.BanditsMSELoss()
  elif bandits == BanditsType.EPSILON_GREEDY_CE:
    loss = task_utils.BanditsCELoss()
  elif bandits in [
      BanditsType.EPSILON_GREEDY_UNWEIGHT,
      BanditsType.FALCON,
      BanditsType.SOFTMAX,
  ]:
    loss = task_utils.BanditsMSELoss(importance_weighting=False)
  elif bandits == BanditsType.EPSILON_GREEDY_CE_UNW:
    loss = task_utils.BanditsCELoss(importance_weighting=False)
  elif bandits == BanditsType.SUPERVISED_MSE:
    if task == TaskType.STACKOVERFLOW_TAG:
      loss = task_utils.MultiLabelMSELoss()
    else:
      loss = task_utils.SupervisedMSELoss(num_arms)
  elif bandits == BanditsType.SUPERVISED_CE:
    if task == TaskType.STACKOVERFLOW_TAG:
      loss = task_utils.MultiLabelCELoss()
    else:
      loss = task_utils.SupervisedCELoss()
  else:
    raise ValueError(f'Unrecongnized bandits type {bandits} for model.')
  return create_model_fn(bandits_data_spec, loss=loss)


def _get_bandits_data_fn_and_spec(
    task: TaskType,
    bandits: BanditsType,
    distribution_shift: Optional[DistShiftType],
    data_element_spec: Any,
    epsilon: float,
    mu: Optional[float],
    gamma: float,
    temperature: float,
    tag_size: int,
    use_synthetic_data: bool = False,
) -> tuple[bandits_utils.BanditFnType, Any]:
  """Returns `bandit_data_fn` and `bandits_data_spec` for bandits simulation."""
  reward_fn = None
  if (
      distribution_shift is not None
      and distribution_shift == DistShiftType.BANDITS
  ):
    if task in [TaskType.EMNIST_CHARS, TaskType.EMNIST62_LINEAR]:
      reward_fn = bandits_utils.get_emnist_dist_shift_reward_fn()
    elif task == TaskType.STACKOVERFLOW_TAG:
      reward_fn = bandits_utils.get_stackoverflow_dist_shift_reward_fn(
          tag_size=tag_size, use_synthetic_tag=use_synthetic_data
      )
  if bandits in [
      BanditsType.EPSILON_GREEDY,
      BanditsType.EPSILON_GREEDY_CE,
      BanditsType.EPSILON_GREEDY_UNWEIGHT,
      BanditsType.EPSILON_GREEDY_CE_UNW,
  ]:
    return epsilon_greedy.build_epsilon_greedy_bandit_data_fn(
        data_element_spec=data_element_spec,
        epsilon=epsilon,
        reward_fn=reward_fn,
    )
  elif bandits == BanditsType.FALCON:
    if mu is None:
      if task in [TaskType.EMNIST_DIGITS, TaskType.EMNIST10_LINEAR]:
        mu = 10
      elif task in [TaskType.EMNIST_CHARS, TaskType.EMNIST62_LINEAR]:
        mu = 62
      elif task == TaskType.STACKOVERFLOW_TAG:
        mu = tag_size
      else:
        raise ValueError(f'Unrecognized task {task}.')
    return falcon.build_falcon_bandit_data_fn(
        data_element_spec=data_element_spec,
        mu=mu,
        gamma=gamma,
        reward_fn=reward_fn,
    )
  elif bandits in [BanditsType.SUPERVISED_MSE, BanditsType.SUPERVISED_CE]:
    # For supervised training baselines, track the reward by taking greedy
    # actions.
    return epsilon_greedy.build_epsilon_greedy_bandit_data_fn(
        data_element_spec=data_element_spec, epsilon=0, reward_fn=None
    )
  elif bandits == BanditsType.SOFTMAX:
    return softmax_sampling.build_softmax_bandit_data_fn(
        data_element_spec, temperature=temperature, reward_fn=reward_fn
    )
  else:
    raise ValueError(f'Unrecongnized bandits type {bandits}.')


def _get_eval_fn(model_fn: Callable[[], tff.learning.models.VariableModel]):
  """Returns the evaluation function.

  The (federated) evaluation depends on the definition of metrics in
  `tff.learning.models.VariableModel`. For example, if
  `task_utils.WrapCategoricalAccuracy` is
  used, the trained model will be evaluated for its prediction performance for
  the corresponding supervised task.

  Args:
    model_fn: A no-arg function returns the model definition.
  """
  federated_eval = tff.learning.build_federated_evaluation(
      model_fn, use_experimental_simulation_loop=True
  )

  def evaluation_fn(state, evaluation_data):
    model_weights = state.train_state.global_model_weights
    return federated_eval(model_weights, evaluation_data)

  return evaluation_fn


def _get_eval_selection_fn(
    task_dataset: tff.simulation.baselines.BaselineTaskDatasets,
    max_num_samples: Optional[int] = None,
) -> Callable[[int], list[tf.data.Dataset]]:
  """Returns the function for selecting evaluation client per round.

  A centralized evaluation dataset is used. The data format is transformed to
  be compatible with the TFF type checking for bandits data.

  Args:
    task_dataset: The task dataset.
    max_num_samples: The maximum number of samples for the dataset.
  """
  test_dataset = bandits_utils.dataset_format_map(
      task_dataset.get_centralized_test_data()
  )

  if max_num_samples is not None:
    test_dataset = test_dataset.take(max_num_samples)

  def evaluation_selection_fn(round_num):
    del round_num
    return [test_dataset]

  return evaluation_selection_fn


def configure_managers(
    root_output_dir: str,
    experiment_name: str,
) -> tuple[
    tff.program.FileProgramStateManager, list[tff.program.ReleaseManager]
]:
  """Configures checkpoint and metrics managers.

  Args:
    root_output_dir: A string representing the root output directory for the
      training simulation. All metrics and checkpoints will be logged to
      subdirectories of this directory.
    experiment_name: A unique identifier for the current training simulation,
      used to create appropriate subdirectories of `root_output_dir`.

  Returns:
    A `tff.program.FileProgramStateManager`, and a list of
    `tff.program.ReleaseManager` instances.
  """
  program_state_dir = os.path.join(
      root_output_dir, 'checkpoints', experiment_name
  )
  program_state_manager = tff.program.FileProgramStateManager(program_state_dir)

  results_dir = os.path.join(root_output_dir, 'results', experiment_name)
  csv_file_path = os.path.join(results_dir, 'experiment.metrics.csv')
  csv_manager = tff.program.CSVFileReleaseManager(
      file_path=csv_file_path,
      save_mode=tff.program.CSVSaveMode.WRITE,
      key_fieldname='round_num',
  )

  summary_dir = os.path.join(root_output_dir, 'logdir', experiment_name)
  tensorboard_manager = tff.program.TensorBoardReleaseManager(summary_dir)

  logging_manager = tff.program.LoggingReleaseManager()

  logging.info('Writing...')
  logging.info('    checkpoints to: %s', program_state_dir)
  logging.info('    CSV metrics to: %s', csv_file_path)
  logging.info('    TensorBoard summaries to: %s', summary_dir)
  return program_state_manager, [
      logging_manager,
      csv_manager,
      tensorboard_manager,
  ]


def _export_model(
    model_fn: Callable[[], tff.learning.models.VariableModel],
    state: tff.learning.templates.LearningAlgorithmState,
    export_dir: str,
):
  model = model_fn()
  state.train_state.global_model_weights.assign_weights_to(model)  # pytype: disable=attribute-error
  model._keras_model.save(export_dir)  # pylint: disable=protected-access  # pytype: disable=attribute-error


def _configure_aggregator(
    *,
    aggregator_type: Optional[AggregatorType] = None,
    model_fn: Optional[Callable[[], tff.learning.models.VariableModel]] = None,
    clip_norm: Optional[float] = None,
    noise_multiplier: Optional[float] = None,
    report_goal: Optional[int] = None,
    target_unclipped_quantile: Optional[float] = None,
    noise_seed: Optional[int] = None,
    default_adaptive_clipping: bool = False,
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
      federated algorithms. Must be `None` for clipping only aggregators.
    model_fn: (Optional) A no-arg function that returns a
      `tff.learning.models.VariableModel` or a `embedding_model.Model`.
    clip_norm: The l2 clip norm of client delta for differential privacy. Must
      be positive when `noise_multiplier` is not `None`. If
      `target_unclipped_quantile` is not `None`, this is the initial clip norm
      for adaptive clipping.
    noise_multiplier: The noise multiplier for differential privacy. The noise
      std for the sum of client deltas is equal to `clip_norm*noise_multiplier`.
      If `None`, no differential privacy mechanism is applied.
    report_goal: The report goal/minimum expected clients per round. Must be
      positive when `noise_multiplier` is not `None`.
    target_unclipped_quantile: The desired quantile of updates which should be
      unclipped, following "Differentially Private Learning with Adaptive
      Clipping"
      https://arxiv.org/abs/1905.03871. If `None`, use fixed `clip_norm` instead
        of adaptive clipping.
    noise_seed: Seed for random noise generation. If `None` and
      `noise_multiplier` is not `None`, non-deterministic noise will be used.
    default_adaptive_clipping: Whether to use the default adaptive clipping with
      target quantile 0.8 following "Differentially Private Learning with
      Adaptive Clipping"
      https://arxiv.org/abs/1905.03871.
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
      )
    elif aggregator_type == AggregatorType.DPFTRL:
      raise ValueError('DP-FTRL with adaptive clipping is not supported.')
    else:
      raise ValueError(f'Unknown aggregator type {aggregator_type}')
  elif noise_multiplier is not None:
    if aggregator_type is None:
      raise ValueError('Please specify `aggregator_type`.')
    elif aggregator_type == AggregatorType.DPFTRL:
      model_weight_specs = tff.types.type_to_tf_tensor_specs(
          tff.learning.models.weights_type_from_model(model_fn).trainable
      )
      return tff.aggregators.DifferentiallyPrivateFactory.tree_aggregation(
          noise_multiplier=noise_multiplier,
          clients_per_round=report_goal,
          l2_norm_clip=clip_norm,
          record_specs=model_weight_specs,
          noise_seed=noise_seed,
          use_efficient=True,
      )
    elif aggregator_type == AggregatorType.DPSGD:
      return tff.aggregators.DifferentiallyPrivateFactory.gaussian_fixed(
          noise_multiplier=noise_multiplier,
          clients_per_round=report_goal,
          clip=clip_norm,
      )
    else:
      raise ValueError(f'Unknown aggregator type {aggregator_type}')
  elif clip_norm is not None:
    if aggregator_type is not None:
      raise ValueError(
          'Clipping only without noise, set `aggregator_type=None`.'
      )
    return tff.aggregators.robust.clipping_factory(
        clipping_norm=clip_norm,
        inner_agg_factory=tff.aggregators.UnweightedMeanFactory(),
    )
  else:
    if aggregator_type is not None:
      raise ValueError(
          'Clipping only without noise,, adaptive clipping of target quantile '
          '0.8, set `aggregator_type=None`.'
      )
    return tff.learning.robust_aggregator(
        zeroing=False,
        clipping=default_adaptive_clipping,
        weighted=False,
        debug_measurements_fn=None,
    )


def train_and_eval(
    task: TaskType,
    bandits: BanditsType,
    *,
    distribution_shift: Optional[DistShiftType] = None,
    population_client_selection: Optional[str] = None,
    # training
    total_rounds: int,
    clients_per_round: int,
    rounds_per_eval: int,
    server_optimizer: _OptimizerType,
    client_optimizer: _OptimizerType,
    # model and dataset
    use_synthetic_data: bool = False,
    train_client_batch_size: int = 16,
    test_client_batch_size: int = 128,
    train_client_epochs_per_round: int = 1,
    train_shuffle_buffer_size: Optional[int] = None,
    train_client_max_elements: Optional[int] = None,
    max_validation_samples: Optional[int] = None,
    # bandits
    bandits_epsilon: float = 0.2,
    bandits_mu: Optional[float] = None,
    bandits_gamma: float = 100,
    bandits_temperature: float = 1,
    bandits_deployment_frequency: int = 32,
    # aggregator/DP
    aggregator_type: Optional[AggregatorType] = None,
    clip_norm: Optional[float] = None,
    noise_multiplier: Optional[float] = None,
    target_unclipped_quantile: Optional[float] = None,
    adaptive_clipping: bool = False,
    # metrics/managers
    program_state_manager: Optional[tff.program.FileProgramStateManager] = None,
    rounds_per_saving_program_state: int = 50,
    metrics_managers: Optional[list[tff.program.ReleaseManager]] = None,
    # task specific
    stackoverflow_vocab_size: int = stackoverflow.DEFAULT_WORD_VOCAB_SIZE,
    stackoverflow_tag_size: int = stackoverflow.DEFAULT_TAG_VOCAB_SIZE,
    # saving and loading
    initial_model_path: Optional[str] = None,
    export_dir: Optional[str] = None,
) -> bandits_process.ServerState:
  """Train and evaluate bandits process in TFF simulation.

  Args:
    task: A task that defines data and model for TFF simulation.
    bandits: The bandits algorithm for simulation.
    distribution_shift: The type of distribution shift for experiments.
    population_client_selection: Use a subset of clients to form the training
      population; can be useful for distribution shift settings. Should be in
      the format of "start_index-end_index" for [start_index, end_index) of all
      the sorted clients in a simulation dataset. for example "0-1000" will
      select the first 1000 clients.
    total_rounds: A positive integer for the total number of training rounds.
    clients_per_round: A positive integer for clients per round.
    rounds_per_eval: A positive integer for the frequency of evaluation.
    server_optimizer: A no-arg callable that returns a `tf.keras.Optimizer`.
    client_optimizer: A a no-arg callable that returns a `tf.keras.Optimizer`.
    use_synthetic_data: If True, use synthetic data. Useful in unit test.
    train_client_batch_size: The batch size for train clients.
    test_client_batch_size: The batch size for test clients. A centralized test
      dataset can be considered a single client.
    train_client_epochs_per_round: The number of epochs each train client should
      iterate over their local dataset, via `tf.data.Dataset.repeat`. Must be
      set to a positive integer.
    train_shuffle_buffer_size: An integer representing the shuffle buffer size
      (as in `tf.data.Dataset.shuffle`) for each train client's dataset. By
      default, this is set to the largest dataset size among all clients. If set
      to some integer less than or equal to 1, no shuffling occurs. If set to
      None, a value will be chosen based on `tff.simulation.baselines` default.
    train_client_max_elements: The maximum number of samples per client via
      `tf.data.Dataset.take`. If `None`, all the client data will be used.
    max_validation_samples: The maximum number of samples for (centralized)
      validation. If `None`, all samples in the validation/test dataset will be
      used.
    bandits_epsilon: The exploration parameter for epsilon-greedy bandits.
    bandits_mu: The exploration parameter mu for FALCON algorithm. If `None`,
      defaults to the number of possible actions.
    bandits_gamma: The exploration parameter gamma for FALCON algorithm.
    bandits_temperature: Parameter for softmax exploration.
    bandits_deployment_frequency: Deploys the online training model for bandits
      inference every `bandits_deployment_frequency` rounds.
    aggregator_type: (Optional) DPFTRL or DPSGD for differentially private
      federated algorithms. Must be `None` for clipping only aggregators.
    clip_norm: The l2 clip norm of client delta for differential privacy. Must
      be positive when `noise_multiplier` is not `None`. If
      `target_unclipped_quantile` is not `None`, this is the initial clip norm
      for adaptive clipping.
    noise_multiplier: The noise multiplier for differential privacy. The noise
      std for the sum of client deltas is equal to `clip_norm*noise_multiplier`.
      If `None`, no differential privacy mechanism is applied.
    target_unclipped_quantile: The desired quantile of updates which should be
      unclipped, following "Differentially Private Learning with Adaptive
      Clipping"
      https://arxiv.org/abs/1905.03871. If `None`, use fixed `clip_norm` instead
        of adaptive clipping.
    adaptive_clipping: For the default aggregator, whether to use adaptive
      clipping with target quantile 0.8 following "Differentially Private
      Learning with Adaptive Clipping"
      https://arxiv.org/abs/1905.03871.
    program_state_manager: An optional `tff.program.FileProgramStateManager`
      used to periodically save program state of the iterative process state.
    rounds_per_saving_program_state: How often to write program state.
    metrics_managers: An optional list of `tff.program.ReleaseManager` objects
      used to save training metrics throughout the simulation.
    stackoverflow_vocab_size: The vocabulary size for StackOverflow tag
      prediction.
    stackoverflow_tag_size: The number of tags for StackOverflow tag prediction.
    initial_model_path: A path to load a saved model for initialization.
    export_dir: A path to save a trained model.

  Returns:
    `bandits_process.ServerState` after training.
  """
  _check_nonnegative(total_rounds, 'total rounds')
  _check_positive(clients_per_round, 'clients per round')
  _check_positive(rounds_per_eval, 'rounds_per_eval')
  _check_positive(train_client_batch_size, 'train_client_batch_size')

  task_dataset = _get_task_data(
      task=task,
      train_client_batch_size=train_client_batch_size,
      test_client_batch_size=test_client_batch_size,
      train_client_epochs_per_round=train_client_epochs_per_round,
      train_shuffle_buffer_size=train_shuffle_buffer_size,
      train_client_max_elements=train_client_max_elements,
      use_synthetic_data=use_synthetic_data,
      stackoverflow_vocab_size=stackoverflow_vocab_size,
      stackoverflow_tag_size=stackoverflow_tag_size,
      distribution_shift=distribution_shift,
      population_client_selection=population_client_selection,
  )
  bandits_data_fn, bandits_data_spec = _get_bandits_data_fn_and_spec(
      task=task,
      bandits=bandits,
      distribution_shift=distribution_shift,
      data_element_spec=task_dataset.element_type_structure,
      epsilon=bandits_epsilon,
      mu=bandits_mu,
      gamma=bandits_gamma,
      temperature=bandits_temperature,
      tag_size=stackoverflow_tag_size,
      use_synthetic_data=use_synthetic_data,
  )
  model_fn = _get_model_fn(
      task,
      bandits,
      bandits_data_spec=bandits_data_spec,
      stackoverflow_vocab_size=stackoverflow_vocab_size,
      stackoverflow_tag_size=stackoverflow_tag_size,
  )

  aggregator = _configure_aggregator(
      aggregator_type=aggregator_type,
      model_fn=model_fn,
      clip_norm=clip_norm,
      noise_multiplier=noise_multiplier,
      report_goal=clients_per_round,
      target_unclipped_quantile=target_unclipped_quantile,
      default_adaptive_clipping=adaptive_clipping,
  )
  train_process = tff.learning.algorithms.build_unweighted_fed_avg(
      model_fn,
      server_optimizer_fn=server_optimizer,
      client_optimizer_fn=client_optimizer,
      model_aggregator=aggregator,
      use_experimental_simulation_loop=True,
  )
  bandit_process = bandits_process.build_bandits_iterative_process(
      model_fn=model_fn,
      training_process=train_process,
      train2infer_frequency=bandits_deployment_frequency,
      data_element_spec=task_dataset.element_type_structure,
      bandit_data_fn=bandits_data_fn,
      initial_model_path=initial_model_path,
  )

  def client_selection_fn(round_num):
    del round_num
    return task_dataset.sample_train_clients(num_clients=clients_per_round)

  evaluation_fn = _get_eval_fn(model_fn)
  evaluation_selection_fn = _get_eval_selection_fn(
      task_dataset, max_num_samples=max_validation_samples
  )

  # TODO(b/215566681): probably wants a final evaluation on all test data
  # explicitly set `max_validation_samples=None` for StackOverflow.
  state = tff.simulation.run_training_process(
      training_process=bandit_process,
      training_selection_fn=client_selection_fn,
      total_rounds=total_rounds,
      evaluation_fn=evaluation_fn,
      evaluation_selection_fn=evaluation_selection_fn,
      rounds_per_evaluation=rounds_per_eval,
      program_state_manager=program_state_manager,
      rounds_per_saving_program_state=rounds_per_saving_program_state,
      metrics_managers=metrics_managers,
  )

  # TODO(b/215566681): might consider also save and load optimizer states. It is
  # not particularly difficult to implement, but requires relatively strong
  # conditions to work, e.g., no distribution shift, same loss and optimizers.
  if export_dir:
    _export_model(model_fn, state, export_dir)

  return state
