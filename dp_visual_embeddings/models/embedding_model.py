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
"""Abstractions for models used in federated learning."""

import abc
from collections.abc import Sequence

import tensorflow as tf
import tensorflow_federated as tff


MODEL_ARG_NAME = 'x'
MODEL_LABEL_NAME = 'y'


class Model(tff.learning.Model, metaclass=abc.ABCMeta):
  """Represents a model for partially local training in TFF simulation.

  See `tff.learning.Model` for more details. This inherited model class further
  distinguish

    * Global variables of `trainable_variables` and `non_trainable_variables`,
    which will be shared between server and client. `trainable_variables` will
    be updated by aggregated results from clients.
    * Client local variables of `client_trainable_variables` and
    `client_non_trainable_variables`, which may be updated and kept on clients.
  """

  @property
  @abc.abstractmethod
  def client_trainable_variables(self) -> Sequence[tf.Variable]:
    """An iterable of trainable `tf.Variable` objects on local clients."""
    pass

  @property
  @abc.abstractmethod
  def client_non_trainable_variables(self) -> Sequence[tf.Variable]:
    """An iterable of non_trainable `tf.Variable` objects on local clients."""
    pass
