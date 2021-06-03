# Copyright 2019, Google LLC.
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
"""Configuration classes for creating and running federated training tasks."""

from typing import Callable, List, Optional, Union

import attr
import tensorflow as tf
import tensorflow_federated as tff

# Typing information
KerasModelBuilderType = Callable[[], tf.keras.Model]
LossBuilderType = Callable[[], tf.keras.losses.Loss]
MetricsBuilderType = Callable[[], List[tf.keras.metrics.Metric]]
TFFModelBuilderType = Callable[[], tff.learning.Model]
IterProcBuilderType = Callable[[TFFModelBuilderType],
                               tff.templates.IterativeProcess]
PreprocessFnType = Union[Callable[[tf.data.Dataset], tf.data.Dataset],
                         tff.Computation]


def _check_positive(instance, attribute, value):
  if value <= 0:
    raise ValueError(f'{attribute.name} must be positive. Found {value}.')


@attr.s(eq=False, order=False, frozen=True)
class ClientSpec(object):
  """Contains information for configuring clients within a training task.

  Attributes:
    num_epochs: An integer representing the number of passes each client
      performs over its entire local dataset.
    batch_size: An integer representing the batch size used when iterating over
      client datasets.
    max_elements: An optional integer governing the maximum number of
      examples used by each training client.
  """
  num_epochs: int = attr.ib(
      validator=[attr.validators.instance_of(int), _check_positive],
      converter=int)
  batch_size: int = attr.ib(
      validator=[attr.validators.instance_of(int), _check_positive],
      converter=int)
  max_elements: Optional[int] = attr.ib(
      default=-1,
      validator=attr.validators.optional(attr.validators.instance_of(int)))


@attr.s(eq=False, order=False, frozen=True)
class DataSpec(object):
  """Contains data and preprocessing for running a federated training task.

  Attributes:
    train_data: A `tff.simulation.datasets.ClientData` for training.
    validation_data: A `tff.simulation.datasets.ClientData` for computing
      validation metrics.
    test_data: An optional `tff.simulation.datasets.ClientData` for computing
      test metrics.
    train_preprocess_fn: A callable accepting and returning a `tf.data.Dataset`,
      used to perform training preprocessing.
    eval_preprocess_fn: A callable accepting and returning a `tf.data.Dataset`,
      used to perform evaluation (eg. validation, testing) preprocessing.
  """
  train_data: tff.simulation.datasets.ClientData = attr.ib()
  validation_data: Optional[tff.simulation.datasets.ClientData] = attr.ib()
  test_data: tff.simulation.datasets.ClientData = attr.ib()
  train_preprocess_fn: PreprocessFnType = attr.ib()
  eval_preprocess_fn: PreprocessFnType = attr.ib()


@attr.s(eq=False, order=False, frozen=True)
class ModelSpec(object):
  """Contains information about the model being used in a federated task.

  Attributes:
    keras_model_builder: A no-arg callable returning an uncompiled
      `tf.keras.Model`.
    loss_builder: A no-arg callable returning a `tf.keras.losses.Loss`.
    metrics_builder: A no-arg callable returning a list of
      `tf.keras.metrics.Metric`.
  """
  keras_model_builder: KerasModelBuilderType = attr.ib()
  loss_builder: LossBuilderType = attr.ib()
  metrics_builder: MetricsBuilderType = attr.ib()
