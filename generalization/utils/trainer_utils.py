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
"""Utils function for configuring a federated or centralized task."""

import collections
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import tensorflow as tf
import tensorflow_federated as tff

from generalization.tasks import training_specs
from generalization.utils import client_data_utils
from generalization.utils import eval_metric_distribution


SamplerType = Iterator[List[tf.data.Dataset]]
MetricsDictType = training_specs.MetricsDictType
FederatedEvalFnType = training_specs.FederatedEvalFnType
CentralizedEvalFnType = training_specs.CentralizedEvalFnType
FederatedModelFnType = training_specs.FederatedModelFnType
StatFnType = eval_metric_distribution.StatFnType

ClientData = tff.simulation.datasets.ClientData
ServerState = tff.learning.framework.ServerState


def _create_samplers(
    *,  # Caller passes below args by name.
    part_train_eval_cd: tff.simulation.datasets.ClientData,
    part_val_cd: tff.simulation.datasets.ClientData,
    unpart_cd: tff.simulation.datasets.ClientData,
    test_cd: Optional[tff.simulation.datasets.ClientData],
    part_clients_per_eval: Optional[int],
    unpart_clients_per_eval: Optional[int],
    test_clients_for_eval: Optional[int],
    resample_eval_clients: bool,
    eval_clients_random_seed: Optional[int],
) -> Tuple[SamplerType, SamplerType, SamplerType, Optional[SamplerType]]:
  """Create four samplers."""
  part_train_eval_sampler = client_data_utils.FederatedDatasetSampler(
      client_data=part_train_eval_cd,
      num_sample_clients=part_clients_per_eval,
      resample=resample_eval_clients,
      seed=eval_clients_random_seed)

  part_val_sampler = client_data_utils.FederatedDatasetSampler(
      client_data=part_val_cd,
      num_sample_clients=part_clients_per_eval,
      resample=resample_eval_clients,
      seed=eval_clients_random_seed)

  unpart_sampler = client_data_utils.FederatedDatasetSampler(
      client_data=unpart_cd,
      num_sample_clients=unpart_clients_per_eval,
      resample=resample_eval_clients,
      seed=eval_clients_random_seed)

  if test_cd is not None:
    test_sampler = client_data_utils.FederatedDatasetSampler(
        client_data=test_cd,
        num_sample_clients=test_clients_for_eval,
        seed=eval_clients_random_seed)
  else:
    test_sampler = None

  return (part_train_eval_sampler, part_val_sampler, unpart_sampler,
          test_sampler)


def create_federated_eval_fns(
    *,  # Caller passes below args by name.
    tff_model_builder: FederatedModelFnType,
    metrics_builder: Callable[[], List[tf.keras.metrics.Metric]],
    part_train_eval_cd: tff.simulation.datasets.ClientData,
    part_val_cd: tff.simulation.datasets.ClientData,
    unpart_cd: tff.simulation.datasets.ClientData,
    test_cd: Optional[tff.simulation.datasets.ClientData],
    stat_fns: Dict[str, StatFnType],
    rounds_per_eval: int,
    part_clients_per_eval: Optional[int],
    unpart_clients_per_eval: Optional[int],
    test_clients_for_eval: Optional[int],
    resample_eval_clients: bool,
    eval_clients_random_seed: Optional[int],
) -> Tuple[FederatedEvalFnType, FederatedEvalFnType, FederatedEvalFnType,
           Optional[FederatedEvalFnType]]:
  """Create federated evaluation functions: train_eval, validation and test.

  Args:
    tff_model_builder: A callable with no args that returns a
      `tff.learning.Model`.
    metrics_builder: A callable with no args that returns a list of keras
      metrics.
    part_train_eval_cd: Preprocessed training chunk of for training ClientData
      used for evaluation.
    part_val_cd: Preprocessed validation chunk of training ClientData.
    unpart_cd: Preprocessed validation ClientData.
    test_cd: Optional preprocessed ClientData for test after training. If None,
      the test function will not be constructed, and None value will be returned
      at the position of test function.
    stat_fns: A mapping in which each key-value pair represents a custom
      statistic to be evaluated on the client metrics. Each pair consists of a
      string-typed key describing this statistic, and a callable-typed value
      that computes the statistic of metrics. The callable value should accept
      two sequence-typed arguments `all_clients_this_metric` and
      `all_clients_num_examples` and returns the corresponding statistics.
    rounds_per_eval: An integer representing how often to evaluate the global
      model on training and validation dataset.
    part_clients_per_eval: An optional integer representing the number of
      training clients taken from training dataset for evaluation per round. If
      `None`, all training clients will be used.
    unpart_clients_per_eval: An optional integer representing the number of
      clients taken from validation dataset. If `None`, all validation clients
      will be used.
    test_clients_for_eval: An optional integer representing the number of
      clients taken from test dataset, valid only if test_cd is not None. If
      `None`, all test clients will be used.
    resample_eval_clients: A bool used to decide whether or not to resample
      validation clients every evaluation round.
    eval_clients_random_seed: An optional integer used to seed which validation
      and test clients are sampled. If `None`, no seed is used.

  Returns:
    Federated train_train_eval, train_validation, validation and test functions.
  """

  (part_train_eval_sampler, part_val_sampler, unpart_sampler,
   test_sampler) = _create_samplers(
       part_train_eval_cd=part_train_eval_cd,
       part_val_cd=part_val_cd,
       unpart_cd=unpart_cd,
       test_cd=test_cd,
       part_clients_per_eval=part_clients_per_eval,
       unpart_clients_per_eval=unpart_clients_per_eval,
       test_clients_for_eval=test_clients_for_eval,
       resample_eval_clients=resample_eval_clients,
       eval_clients_random_seed=eval_clients_random_seed)

  evaluate_fn = eval_metric_distribution.create_federated_eval_distribution_fn(
      model_fn=tff_model_builder,
      metrics_builder=metrics_builder,
      stat_fns=stat_fns)

  def part_train_eval_fn(state: ServerState, round_num: int) -> MetricsDictType:
    if round_num % rounds_per_eval == 0:
      return evaluate_fn(state.model, next(part_train_eval_sampler))
    else:
      return collections.OrderedDict()

  def part_val_fn(state: ServerState, round_num: int) -> MetricsDictType:
    if round_num % rounds_per_eval == 0:
      return evaluate_fn(state.model, next(part_val_sampler))
    else:
      return collections.OrderedDict()

  def unpart_fn(state: ServerState, round_num: int) -> MetricsDictType:
    if round_num % rounds_per_eval == 0:
      return evaluate_fn(state.model, next(unpart_sampler))
    else:
      return collections.OrderedDict()

  if test_sampler is not None:

    def test_fn(state: ServerState, round_num: int = 0) -> MetricsDictType:
      # test_fn does not need round_num. We define to keep interface consistent.
      del round_num
      return evaluate_fn(state.model, next(test_sampler))
  else:
    test_fn = None

  return part_train_eval_fn, part_val_fn, unpart_fn, test_fn


def create_centralized_eval_fns(
    *,  # Caller passes below args by name.
    tff_model_builder: FederatedModelFnType,
    metrics_builder: Callable[[], List[tf.keras.metrics.Metric]],
    part_train_eval_cd: ClientData,
    part_val_cd: ClientData,
    unpart_cd: ClientData,
    test_cd: Optional[ClientData],
    stat_fns: Dict[str, StatFnType],
    part_clients_per_eval: Optional[int],
    unpart_clients_per_eval: Optional[int],
    test_clients_for_eval: Optional[int],
    resample_eval_clients: bool,
    eval_clients_random_seed: Optional[int],
) -> Tuple[CentralizedEvalFnType, CentralizedEvalFnType, CentralizedEvalFnType,
           CentralizedEvalFnType]:
  """Create centralized evaluation functions: train_eval, validation and test.

  Args:
    tff_model_builder: A callable with no args that returns a
      `tff.learning.Model`.
    metrics_builder: A callable with no args that returns a list of keras
      metrics.
    part_train_eval_cd: Preprocessed training chunk of for training ClientData
      used for evaluation.
    part_val_cd: Preprocessed validation chunk of training ClientData.
    unpart_cd: Preprocessed validation ClientData.
    test_cd: Optional preprocessed ClientData for test after training. If None,
      the test function will not be constructed, and None value will be returned
      at the position of test function.
    stat_fns: A mapping in which each key-value pair represents a custom
      statistic to be evaluated on the client metrics. Each pair consists of a
      string-typed key describing this statistic, and a callable-typed value
      that computes the statistic of metrics. The callable value should accept
      two sequence-typed arguments `all_clients_this_metric` and
      `all_clients_num_examples` and returns the corresponding statistics.
    part_clients_per_eval: An optional integer representing the number of
      training clients taken from training dataset for evaluation per round. If
      `None`, all training clients will be used.
    unpart_clients_per_eval: An optional integer representing the number of
      clients taken from validation dataset. If `None`, all validation clients
      will be used.
    test_clients_for_eval: An optional integer representing the number of
      clients taken from test dataset. If `None`, all test clients will be used.
    resample_eval_clients: A bool used to decide whether or not to resample
      validation clients every evaluation round.
    eval_clients_random_seed: An optional integer used to seed which validation
      and test clients are sampled. If `None`, no seed is used.

  Returns:
    Centralized train_eval fn, validation fn and test fn, all of type
    CentralizedEvalFnType.

  """
  (part_train_eval_sampler, part_val_sampler, unpart_sampler,
   test_sampler) = _create_samplers(
       part_train_eval_cd=part_train_eval_cd,
       part_val_cd=part_val_cd,
       unpart_cd=unpart_cd,
       test_cd=test_cd,
       part_clients_per_eval=part_clients_per_eval,
       unpart_clients_per_eval=unpart_clients_per_eval,
       test_clients_for_eval=test_clients_for_eval,
       resample_eval_clients=resample_eval_clients,
       eval_clients_random_seed=eval_clients_random_seed)

  evaluate_fn = eval_metric_distribution.create_federated_eval_distribution_fn(
      model_fn=tff_model_builder,
      metrics_builder=metrics_builder,
      stat_fns=stat_fns)

  def part_train_eval_fn(keras_model: tf.keras.Model) -> MetricsDictType:
    return evaluate_fn(
        tff.learning.ModelWeights.from_model(keras_model),
        next(part_train_eval_sampler))

  def part_val_fn(keras_model: tf.keras.Model) -> MetricsDictType:
    return evaluate_fn(
        tff.learning.ModelWeights.from_model(keras_model),
        next(part_val_sampler))

  def unpart_fn(keras_model: tf.keras.Model) -> MetricsDictType:
    return evaluate_fn(
        tff.learning.ModelWeights.from_model(keras_model), next(unpart_sampler))

  if test_sampler is not None:

    def test_fn(keras_model: tf.keras.Model) -> MetricsDictType:
      return evaluate_fn(
          tff.learning.ModelWeights.from_model(keras_model), next(test_sampler))
  else:
    test_fn = None

  return part_train_eval_fn, part_val_fn, unpart_fn, test_fn
