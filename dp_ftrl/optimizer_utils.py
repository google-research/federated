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
"""Server Optimizers for Federated Learning."""

import abc
import collections
from typing import Any, Collection, Dict

import tensorflow as tf

from dp_ftrl import tree_aggregation


class ServerOptimizerBase(metaclass=abc.ABCMeta):
  """Base class establishing interface for server optimizer."""

  @abc.abstractmethod
  def model_update(self, state: Dict[str, Any], weight: Collection[tf.Variable],
                   grad: Collection[tf.Tensor],
                   round_idx: int) -> Dict[str, Any]:
    """Returns optimizer states after modifying in-place the provided `weight`.

    Args:
      state: optimizer state, usually defined/initialized in `init_state`.
      weight: model weights to be updated in this function.
      grad: gradients to update the model weights and optimizer states.
      round_idx: round/iteration index.

    Returns:
      Updated optimizer state.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def init_state(self) -> Dict[str, Any]:
    """Returns initialized optimizer state."""
    raise NotImplementedError


class SGDServerOptimizer(ServerOptimizerBase):
  """Simple SGD Optimizer."""

  def __init__(self, learning_rate: float):
    self.lr = learning_rate

  @tf.function
  def model_update(self, state: Dict[str, Any], weight: Collection[tf.Variable],
                   grad: Collection[tf.Tensor],
                   round_idx: int) -> Dict[str, Any]:
    del round_idx, state
    tf.nest.map_structure(lambda w, g: w.assign_sub(self.lr * g), weight, grad)
    return collections.OrderedDict()

  def init_state(self):
    return collections.OrderedDict()


class DPFTRLMServerOptimizer(ServerOptimizerBase):
  """Momentum FTRL Optimizer with Tree aggregation for DP noise."""

  def __init__(self, learning_rate: float, momentum: float, noise_std: float,
               model_weight_shape: Collection[tf.Tensor]):
    """Initialize the momemtum DPFTRL Optimizer."""
    self.lr = learning_rate
    self.momentum = momentum
    self.model_weight_shape = model_weight_shape

    random_generator = tf.random.Generator.from_non_deterministic_state()

    def _noise_fn():
      return tf.nest.map_structure(
          lambda x: random_generator.normal(x, stddev=noise_std),
          model_weight_shape)

    self.noise_generator = tree_aggregation.TFTreeAggregator(
        new_value_fn=_noise_fn)

  @tf.function
  def model_update(self, state: Dict[str, Any], weight: Collection[tf.Variable],
                   grad: Collection[tf.Tensor],
                   round_idx: int) -> Dict[str, Any]:
    """Returns optimizer state after one step update."""
    # TODO(b/172867399): use `attr.s` instead of ditionary for better
    # readability and avoiding pack and unpack
    init_weight, sum_grad, dp_tree_state, momentum_buffer = state[
        'init_weight'], state['sum_grad'], state['dp_tree_state'], state[
            'momentum_buffer']
    round_idx = tf.cast(round_idx, tf.int32)
    if tf.equal(round_idx, tf.constant(0, dtype=tf.int32)):
      init_weight = weight

    sum_grad = tf.nest.map_structure(tf.add, sum_grad, grad)
    cumsum_noise, dp_tree_state = self.noise_generator.get_cumsum_and_update(
        dp_tree_state)

    momentum_buffer = tf.nest.map_structure(
        lambda m, g, n: self.momentum * m + g + n, momentum_buffer, sum_grad,
        cumsum_noise)
    # Different from a conventional SGD step, FTRL use the initial weight w0
    # and (momementum version of) the gradient sum to update the model weight.
    tf.nest.map_structure(lambda w, w0, v: w.assign(w0 - self.lr * v), weight,
                          init_weight, momentum_buffer)

    state = collections.OrderedDict(
        init_weight=init_weight,
        sum_grad=sum_grad,
        dp_tree_state=dp_tree_state,
        momentum_buffer=momentum_buffer)
    return state

  def init_state(self) -> Dict[str, Any]:
    """Returns initialized optimizer and tree aggregation states."""

    def _zero_state():
      return tf.nest.map_structure(tf.zeros, self.model_weight_shape)

    return collections.OrderedDict(
        init_weight=_zero_state(),
        sum_grad=_zero_state(),
        dp_tree_state=self.noise_generator.init_state(),
        momentum_buffer=_zero_state())


class DPSGDMServerOptimizer(ServerOptimizerBase):
  """Momentum DPSGD Optimizer."""

  def __init__(self, learning_rate: float, momentum: float, noise_std: float,
               model_weight_shape: Collection[tf.Tensor]):
    """Initialize the momemtum DPSGD Optimizer."""
    self.lr = learning_rate
    self.momentum = momentum
    self.model_weight_shape = model_weight_shape

    self.noise_std = noise_std
    # TODO(b/177243233): possibly add the state of noise generator to the state
    # of optimzier. TFF now rely on the non-deterministic initialization of the
    # random noise generator and the states/variables have to be created inside
    # the TFF computation.
    self.random_generator = tf.random.Generator.from_non_deterministic_state()

  def _noise_fn(self):
    """Returns random noise to be added for differential privacy."""

    def noise_tensor(model_weight_shape):
      noise = self.random_generator.normal(
          model_weight_shape, stddev=self.noise_std)
      # TODO(b/177259859): reshape because the shape of the noise could have
      # None/? that fails TFF type check.
      noise = tf.reshape(noise, model_weight_shape)
      return noise

    return tf.nest.map_structure(noise_tensor, self.model_weight_shape)

  @tf.function
  def model_update(self, state: Dict[str, Any], weight: Collection[tf.Variable],
                   grad: Collection[tf.Tensor],
                   round_idx: int) -> Dict[str, Any]:
    """Returns optimizer state after one step update."""
    del round_idx
    momentum_buffer = state['momentum_buffer']

    momentum_buffer = tf.nest.map_structure(
        lambda m, g, n: self.momentum * m + g + n, momentum_buffer, grad,
        self._noise_fn())
    tf.nest.map_structure(lambda w, v: w.assign_sub(self.lr * v), weight,
                          momentum_buffer)
    state = collections.OrderedDict(momentum_buffer=momentum_buffer)
    return state

  def init_state(self) -> Dict[str, Any]:
    """Returns initialized momentum buffer."""

    def _zero_state():
      return tf.nest.map_structure(tf.zeros, self.model_weight_shape)

    return collections.OrderedDict(momentum_buffer=_zero_state())
