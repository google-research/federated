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
"""Configuration classes for creating federated and centralized training tasks."""

from typing import Any, Callable, Dict, List, Optional

import attr
import tensorflow as tf
import tensorflow_federated as tff

MetricsDictType = Dict[str, float]
FederatedModelFnType = Callable[[], tff.learning.Model]
FederatedEvalFnType = Optional[Callable[[Any, int], MetricsDictType]]
CentralizedEvalFnType = Optional[Callable[[Any], MetricsDictType]]


def _check_positive(instance, attribute, value):
  if value <= 0:
    raise ValueError(f'{attribute.name} must be positive. Found {value}.')


@attr.s(eq=False, order=False, frozen=True, kw_only=True)
class TaskSpec(object):
  # pyformat: disable
  """Contains shared information for creating a centralized and federated training task.

  Attributes:

    *The following 6 attributes characterizes the construction & partition of
      federated dataset.*

    sql_database: An optional str indicating the data source. If set to None,
      the TFF original data source will be used. Otherwise the program will load
      SQL-based ClientData from `sql_database`.

    unpart_clients_proportion: An optional floating number in (0.0, 1.0)
      representing the proportion of un-participating clients among the total
      clients.

      - If sql_database is not None, or if sql_database is None but the TFF
        original federated dataset source does *not* provide a vertical split,
        then `unpart_clients_proportion` must not be None. In this case, a
        random set of clients will be drawn from the total sets of clients.

      - If sql_database is None, and the TFF original federated dataset source
        provides a vertical split, then `unpart_clients_proportion` must be
        None, and the TFF original vertical split will be used.

      The remaining clients are defined as the "candidate participating
        clients" (for now).

    train_val_ratio_intra_client: An optional integer representing the ratio of
      ratio of train-validation split for each client.

      - If sql_database is not None, or if sql_database is None but the TFF
        original federated dataset does *not* provide a horizontal split,
        then `train_val_ratio_intra_client` must not be None. In this case, for
        each client, the validation dataset contains 1/(train_val_ratio+1) of
        total samples, round up if fractional. The training dataset contains
        the rest of samples.

      - If sql_database is None, and the TFF original federated dataset
        provides a horizontal split, then then `train_val_ratio_intra_client`
        must be None, and the TFF original horizontal split will be used.

    part_clients_subsampling_rate: A floating number in (0.0, 1.0] representing
      the actual proportion of candidate participating clients. If < 1.0, a
      random subset of clients will be drawn from the "candidate participating
      clients" that become the actual participating clients. This attribute is
      mostly intended for the ablation study on the effect of participation
      rate.

    include_unpart_train_for_val: Whether to include the training dataset of
      unparticipated clients for validation.

      - If include_unpart_train_for_val is True:

        +-----------+----+
        | part_val  | u  |
        +-----------+ n  |
        |           | p  |
        |  particip | a  |
        |   train     r  |
        |           | t  |
        +-----------+----+
        partcip.  unparticip.

        This mode may be preferred for obtaining low-variance metrics.

      - If include_unpart_train_for_val is False (XX means discarded):

        +-----------+----+
        |  part_val | unp|
        +-----------+----+
        |           |XXXX|
        |  particip |XXXX|
        |   train   |XXXX|
        |           |XXXX|
        +-----------+----+
        partcip.  unparticip.

        This mode may be preferred if one wants to compare the percentile /
          variance of part_val and unpart (since they will have the same scale
          of elements.)

    max_elements_per_client: An optional integer controlling the maximum number
      of elements to take per client. If None, keep all elements for each
      client. This is intended primarily to contend with the small set of
      clients with tens of thousands of examples.

    *The following 5 attributes characterizes the evaluation procedure.*

    part_clients_per_eval: An optional integer representing the number of
      training clients taken from training dataset for evaluation per round. If
      `None`, all training clients will be used.

    unpart_clients_per_eval: An optional integer representing the number of
      clients taken from validation dataset. If `None`, all validation clients
      will be used.

    test_clients_for_eval: An optional integer representing the number of
      clients taken from test dataset. If `None`, all test clients will be used.

    resample_eval_clients: A bool used to decide whether or not to resample
      training and validation clients every evaluation round.

    eval_client_batch_size: An integer representing the batch size used on
      validation and test clients.

    *The following seed attribute is intended for reproducibility.*

    shared_random_seed: An optional integer used to seed the pseudo-random
      number generator. The seeds are shared across the following functions: 1)
        Sampling training clients for each training round (for federated
        experiments). 2) Sampling training, validation and test clients for
        evaluation rounds. If `None`, no seed is used. Note that specifying
        `shared_random_seed` does not result in the same clients being sampled
        every round.
  """
  # pyformat: enable
  sql_database: Optional[str] = attr.ib(
      validator=attr.validators.optional(attr.validators.instance_of(str)),
      converter=attr.converters.optional(str))
  unpart_clients_proportion: Optional[float] = attr.ib(
      validator=attr.validators.optional(attr.validators.instance_of(float)),
      converter=attr.converters.optional(float))
  train_val_ratio_intra_client: Optional[int] = attr.ib(
      validator=attr.validators.optional(attr.validators.instance_of(int)),
      converter=attr.converters.optional(int))
  part_clients_subsampling_rate: float = attr.ib(
      validator=attr.validators.instance_of(float), converter=float)
  include_unpart_train_for_val: bool = attr.ib(
      validator=attr.validators.instance_of(bool), converter=bool)
  max_elements_per_client: Optional[int] = attr.ib(
      validator=attr.validators.optional(attr.validators.instance_of(int)),
      converter=attr.converters.optional(int))

  part_clients_per_eval: Optional[int] = attr.ib(
      validator=attr.validators.optional(attr.validators.instance_of(int)),
      converter=attr.converters.optional(int))
  unpart_clients_per_eval: Optional[int] = attr.ib(
      validator=attr.validators.optional(attr.validators.instance_of(int)),
      converter=attr.converters.optional(int))
  test_clients_for_eval: Optional[int] = attr.ib(
      validator=attr.validators.optional(attr.validators.instance_of(int)),
      converter=attr.converters.optional(int))
  resample_eval_clients: bool = attr.ib(
      validator=attr.validators.instance_of(bool), converter=bool)
  eval_client_batch_size: int = attr.ib(
      validator=attr.validators.instance_of(int), converter=int)
  shared_random_seed: Optional[int] = attr.ib(
      validator=attr.validators.optional(attr.validators.instance_of(int)),
      converter=attr.converters.optional(int))


@attr.s(eq=False, order=False, frozen=True, kw_only=True)
class TaskSpecFederated(TaskSpec):
  """A subclass of TaskSpec containing additional information for creating a federated task.

  This class contains a callable `iterative_process_builder` for building a
  `tff.templates.IterativeProcess`, as well as hyperparameters governing
  how to perform federated training using the resulting iterative process.

  We assume that the iterative process has the following functional type
  signatures:

    *   `initialize`: `( -> S@SERVER)` where `S` represents the server state.
    *   `next`: `<S@SERVER, {B*}@CLIENTS> -> <S@SERVER, T@SERVER>` where `S`
        represents the server state, `{B*}` represents the client datasets,
        and `T` represents a python `Mapping` object.

  The iterative process must also have a callable attribute `get_model_weights`
  that takes as input the state of the iterative process, and returns a
  `tff.learning.ModelWeights` object.

  Attributes:
    iterative_process_builder: A function that accepts a no-arg `model_fn`, and
      returns a `tff.templates.IterativeProcess`. The `model_fn` must return a
      `tff.learning.Model`.
    client_epochs_per_round: An integer representing the number of epochs of
      training performed per client in each training round.
    client_batch_size: An integer representing the batch size used on clients.
    train_clients_per_round: An integer representing the number of clients
      participating in each round.
    rounds_per_eval: An integer representing how often to evaluate on training
      and valiadation dataset. See also TaskSpec docstring for the other
      Inherited Attributes.
  """
  iterative_process_builder: Callable[
      [FederatedModelFnType], tff.templates.IterativeProcess] = attr.ib(
          validator=attr.validators.is_callable())
  client_epochs_per_round: int = attr.ib(
      validator=[attr.validators.instance_of(int), _check_positive],
      converter=int)
  client_batch_size: int = attr.ib(
      validator=[attr.validators.instance_of(int), _check_positive],
      converter=int)
  train_clients_per_round: int = attr.ib(
      validator=[attr.validators.instance_of(int), _check_positive],
      converter=int)
  rounds_per_eval: int = attr.ib(
      validator=[attr.validators.instance_of(int), _check_positive],
      converter=int)


@attr.s(eq=False, order=False, frozen=True, kw_only=True)
class TaskSpecCentralized(TaskSpec):
  """A subclass of TaskSpec containing additional information for creating a centralized task.

  Attributes:
    optimizer: A `tf.keras.optimizers.Optimizer` used to perform training.
    batch_size: The batch size, used for training.
    centralized_shuffle_buffer_size: Shuffling buffer size for centralized
      training. See also TaskSpec docstring for the other Inherited Attributes.
  """

  optimizer: tf.keras.optimizers.Optimizer = attr.ib(
      validator=attr.validators.instance_of(tf.keras.optimizers.Optimizer))
  batch_size: int = attr.ib(
      validator=[attr.validators.instance_of(int), _check_positive],
      converter=int)
  centralized_shuffle_buffer_size: int = attr.ib(
      validator=[attr.validators.instance_of(int), _check_positive],
      converter=int)


@attr.s(eq=False, order=False, frozen=True, kw_only=True)
class RunnerSpecFederated(object):
  """Contains information for running a federated training task.

  This class contains a `tff.templates.IterativeProcess`, as well as auxiliary
  utilities for running rounds of the iterative process, and evaluating its
  progress.

  We assume that the iterative process has the following functional type
  signatures:

    *   `initialize`: `( -> S@SERVER)` where `S` represents the server state.
    *   `next`: `<S@SERVER, {B*}@CLIENTS> -> <S@SERVER, T@SERVER>` where `S`
        represents the server state, `{B*}` represents the client datasets,
        and `T` represents a python `Mapping` object.

  The iterative process must also have a callable attribute `get_model_weights`
  that takes as input the state of the iterative process, and returns a
  `tff.learning.ModelWeights` object, which can then be used as input to the
  `unpart_fn` and `test_fn` (if provided).

  Attributes:
    iterative_process: A `tff.templates.IterativeProcess` instance to run.
    client_datasets_fn: Function accepting an integer argument (the round
      number) and returning a list of client datasets to use as federated data
      for that round.
    part_train_eval_fn: An optional callable used to compute evaluation metrics
      of training trunk on training clientdata.
    part_val_fn: An optional callable used to compute evaluation metrics of
      validation chunk on training clientdata.
    unpart_fn: An optional callable used to compute evaluation metrics of
      (validation dataset) on validation clientdata.
    test_fn: An optional callable used to compute test metrics after training.
  """
  iterative_process: tff.templates.IterativeProcess = attr.ib(kw_only=True)
  client_datasets_fn: Callable[[int], List[tf.data.Dataset]] = attr.ib(
      validator=attr.validators.is_callable())
  part_train_eval_fn: FederatedEvalFnType = attr.ib(
      validator=attr.validators.optional(attr.validators.is_callable()))
  part_val_fn: FederatedEvalFnType = attr.ib(
      validator=attr.validators.optional(attr.validators.is_callable()))
  unpart_fn: FederatedEvalFnType = attr.ib(
      validator=attr.validators.optional(attr.validators.is_callable()))
  test_fn: FederatedEvalFnType = attr.ib(
      validator=attr.validators.optional(attr.validators.is_callable()))


@attr.s(eq=False, order=False, frozen=True, kw_only=True)
class RunnerSpecCentralized(object):
  """Contains information for running a centralized training task.

  Attributes:
    keras_model: A compiled `tf.keras.model` instance.
    train_dataset: Training dataset of type `tf.data.Dataset`.
    part_train_eval_fn: An optional callable that accepts a `tf.keras.Model` and
      emits a mapping of evaluation metrics on training chunk of training
      clients.
    part_val_fn: An optional callable that accepts a `tf.keras.Model` and emits
      a mapping of evaluation metrics on validation chunk of training clients.
    unpart_fn: An optional callable used to compute evaluation metrics of
      (validation dataset) on validation clientdata.
    test_fn: An optional callable used to compute test metrics after training.
  """

  keras_model: tf.keras.Model = attr.ib()
  train_dataset: tf.data.Dataset = attr.ib()
  part_train_eval_fn: CentralizedEvalFnType = attr.ib(
      validator=attr.validators.optional(attr.validators.is_callable()))
  part_val_fn: CentralizedEvalFnType = attr.ib(
      validator=attr.validators.optional(attr.validators.is_callable()))
  unpart_fn: CentralizedEvalFnType = attr.ib(
      validator=attr.validators.optional(attr.validators.is_callable()))
  test_fn: CentralizedEvalFnType = attr.ib(
      validator=attr.validators.optional(attr.validators.is_callable()))
