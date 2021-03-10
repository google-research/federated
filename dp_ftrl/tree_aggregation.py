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
"""Tree aggregation algorithm.

This algorithm computes cumulative sums of noise based on Dwork et al. (2010)
https://dl.acm.org/doi/pdf/10.1145/1806689.1806787. When using appropriate noise
function, it  allows for efficient differentially private algorithms under
continual observation, without prior subsampling or shuffling assumptions.
"""

from typing import Any, Callable, Collection, Union, Tuple

import attr
import tensorflow as tf

TFListTensorType = Union[Collection[tf.Tensor], tf.Tensor]


@tf.function
def get_step_idx(level_state: TFListTensorType) -> tf.Tensor:
  """Returns the current leaf node index based on binary representation."""
  step_idx = tf.constant(0, dtype=tf.int32)
  for i in tf.range(len(level_state)):
    step_idx += tf.cast(level_state[i], tf.int32) * tf.math.pow(2, i)
  return step_idx


@attr.s(eq=False, frozen=True)
class TreeState(object):
  """Class defining state of the tree.

  Attributes:
    level_state: A `tf.Tensor` for the binary representation of the index of
      the most recent leaf node that was processed. The index of leaf node
      starts from 0.
    level_buffer: A `tf.Tensor` saves the last node value entered for the tree
      levels recorded in `level_buffer_idx`.
    level_buffer_idx: A `tf.Tensor` for the tree level index of the
      `level_buffer`.  The tree level index starts from 0, i.e.,
      `level_buffer[0]` when `level_buffer_idx[0]==0` recorded the noise value
      for the most recent leaf node.
  """
  level_state = attr.ib()
  level_buffer = attr.ib()
  level_buffer_idx = attr.ib()


class TFTreeAggregator:
  """Tree aggregator to compute accumulated noise in private algorithms.

  This class implements the tree aggregation algorithm for noise values to
  efficiently privatize streaming algorithms. A buffer at the scale of tree
  depth is maintained and updated when a new conceptual leaf node arrives.

  Attributes:
    get_new_value: Function that returns a noise value for each tree node.
  """

  def __init__(self, new_value_fn: Callable[[], Any]):
    """Initialize the aggregator with a noise generator function."""
    self.get_new_value = new_value_fn

  def init_state(self):
    """Returns initial `TreeState`.

    Initializes `TreeState.level_state` to 0, initial buffer index to 0, and
    initial buffer with noise generator function.

    Returns:
      `TreeState` for a tree of a single leaf node with the respective initial
      `TreeState.level_state`, node value and node index.
    """
    level_state = tf.TensorArray(dtype=tf.int8, size=1, dynamic_size=True)
    level_state = level_state.write(0, tf.constant(0, dtype=tf.int8))
    level_state = level_state.stack()

    new_buffer_idx = tf.TensorArray(dtype=tf.int32, size=1, dynamic_size=True)
    new_buffer_idx = new_buffer_idx.write(0, tf.constant(0, dtype=tf.int32))
    new_buffer_idx = new_buffer_idx.stack()

    new_val = self.get_new_value()
    new_buffer_structure = tf.nest.map_structure(
        lambda x: tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True),
        new_val)
    new_buffer = tf.nest.map_structure(lambda x, y: x.write(0, y),
                                       new_buffer_structure, new_val)
    new_buffer_stacked = tf.nest.map_structure(lambda x: x.stack(), new_buffer)

    return TreeState(
        level_state=level_state,
        level_buffer=new_buffer_stacked,
        level_buffer_idx=new_buffer_idx)

  @tf.function
  def _update_level_state(self,
                          level_state: TFListTensorType) -> TFListTensorType:
    """Returns updated level state.

    level_state[0] is the lowest bit of the binary representation, where
    `index = level_state[0] + 2*level_state[1] + 4*level_state[1] + ...`
    and will be updated by `index <- index + 1`.

    Args:
      level_state: Binary representation of the current leaf node index.
    """
    new_state = tf.TensorArray(
        dtype=tf.int8, size=len(level_state), dynamic_size=True)
    idx = 0
    while tf.less(idx, len(level_state)) and tf.equal(
        level_state[idx], tf.constant(1, dtype=tf.int8)):
      new_state = new_state.write(idx, tf.constant(0, dtype=tf.int8))
      idx += 1
    new_state = new_state.write(idx, tf.constant(1, dtype=tf.int8))
    idx += 1
    while tf.less(idx, len(level_state)):
      new_state = new_state.write(idx, level_state[idx])
      idx += 1
    return new_state.stack()

  @tf.function
  def _get_cumsum(self, level_buffer: Collection[tf.Tensor]) -> tf.Tensor:
    return tf.nest.map_structure(lambda x: tf.reduce_sum(x, axis=0),
                                 level_buffer)

  @tf.function
  def get_cumsum_and_update(self,
                            state: TreeState) -> Tuple[tf.Tensor, TreeState]:
    """Returns tree aggregated value and updated `TreeState` for one step."""

    level_state, level_buffer_idx, level_buffer = state.level_state, state.level_buffer_idx, state.level_buffer
    cumsum = self._get_cumsum(level_buffer)

    new_level_state = self._update_level_state(level_state)
    new_buffer = tf.nest.map_structure(
        lambda x: tf.TensorArray(  # pylint: disable=g-long-lambda
            dtype=tf.float32,
            size=0,
            dynamic_size=True),
        level_buffer)
    new_buffer_idx = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    # Find the lowest level where the state will switch from '0' to '1'
    level_idx = 0
    while tf.less(level_idx, len(new_level_state)) and tf.equal(
        new_level_state[level_idx], tf.constant(1, dtype=tf.int8)):
      level_idx += 1
    # The level lower than the switch at `level_idx` will be reset.
    buffer_idx = 0
    while tf.less(
        buffer_idx,
        len(level_buffer_idx)) and level_buffer_idx[buffer_idx] < level_idx:
      buffer_idx += 1
    # Create a new value at the switch level of `level_idx`
    write_buffer_idx = 0
    new_buffer_idx = new_buffer_idx.write(write_buffer_idx, level_idx)
    new_value = self.get_new_value()
    new_buffer = tf.nest.map_structure(
        lambda x, y: x.write(write_buffer_idx, y), new_buffer, new_value)
    write_buffer_idx += 1
    while tf.less(buffer_idx, len(level_buffer_idx)):
      new_buffer_idx = new_buffer_idx.write(write_buffer_idx,
                                            level_buffer_idx[buffer_idx])
      buffer_val = tf.nest.map_structure(lambda x: x[buffer_idx], level_buffer)
      new_buffer = tf.nest.map_structure(
          lambda x, y: x.write(write_buffer_idx, y), new_buffer, buffer_val)
      buffer_idx += 1
      write_buffer_idx += 1
    new_buffer_idx = new_buffer_idx.stack()
    new_buffer = tf.nest.map_structure(lambda x: x.stack(), new_buffer)
    new_state = TreeState(
        level_state=new_level_state,
        level_buffer=new_buffer,
        level_buffer_idx=new_buffer_idx)
    return cumsum, new_state
