# Copyright 2021, Google LLC. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Factory for per-entry modular clipping before and after aggregation."""

import collections
from typing import Optional

import tensorflow as tf
import tensorflow_federated as tff

CLIP_VALUE_TF_TYPE = tf.int32


class ModularClippingSumFactory(tff.aggregators.UnweightedAggregationFactory):
  """`AggregationProcess` factory for per-element modular clipping of values.

  The created `tff.templates.AggregationProcess` does per-entry modular clipping
  on inputs with an exclusive upper bound: [clip_range_lower, clip_range_upper).
  For example:
      Input  = [20, 5, -15, 10], clip_range_lower=-5, clip_range_upper=10;
      Output = [5,  5,   0, -5].

  The provided `clip_range_lower` and `clip_range_upper` should be integer
  constants within the value range of tf.int32, though we may extend it to
  `tff.templates.EstimationProcess` in the future for adaptive clipping range.

  The clipping logic implemented in this factory may also apply to floating
  inputs, though for now our intention is to use it with tf.int32 records.
  """

  def __init__(
      self,
      clip_range_lower: int,
      clip_range_upper: int,
      inner_agg_factory: Optional[tff.aggregators.UnweightedAggregationFactory] = None):  # pylint: disable=line-too-long

    if inner_agg_factory is None:
      inner_agg_factory = tff.aggregators.SumFactory()
    self._inner_agg_factory = inner_agg_factory

    if not (isinstance(clip_range_lower, int) and
            isinstance(clip_range_upper, int)):
      raise TypeError('`clip_range_lower` and `clip_range_upper` must be '
                      f'Python `int`; got {clip_range_lower} with type '
                      f'{type(clip_range_lower)} and {clip_range_upper} '
                      f'with type {type(clip_range_upper)}, respectively.')

    if clip_range_lower > clip_range_upper:
      raise ValueError('`clip_range_lower` should not be larger than '
                       f'`clip_range_upper`, got {clip_range_lower} and '
                       f'{clip_range_upper}')

    if (clip_range_upper >= 2**31 or clip_range_lower < -2**31 or
        clip_range_upper - clip_range_lower >= 2**31):
      raise ValueError('`clip_range_lower` and `clip_range_upper` should be '
                       'set such that the range of the modulus do not overflow '
                       f'tf.int32. Found clip_range_lower={clip_range_lower} '
                       f'and clip_range_upper={clip_range_upper} respectively.')

    self._get_clip_range = _create_get_clip_range_const(clip_range_lower,
                                                        clip_range_upper)

  def create(self, value_type) -> tff.templates.AggregationProcess:
    inner_agg_process = self._inner_agg_factory.create(value_type)
    init_fn = self._create_init_fn(inner_agg_process.initialize)
    next_fn = self._create_next_fn(inner_agg_process.next,
                                   init_fn.type_signature.result)
    return tff.templates.AggregationProcess(init_fn, next_fn)

  def _create_init_fn(self, inner_agg_initialize):

    @tff.federated_computation
    def init_fn():
      return inner_agg_initialize()

    return init_fn

  def _create_next_fn(self, inner_agg_next, state_type):

    value_type = inner_agg_next.type_signature.parameter[1]
    modular_clip_by_value_tff = tff.tf_computation(modular_clip_by_value)

    @tff.federated_computation(state_type, value_type)
    def next_fn(state, value):
      clip_range_lower, clip_range_upper = self._get_clip_range()

      # Modular clip values before aggregation.
      clipped_value = tff.federated_map(
          modular_clip_by_value_tff,
          (value, tff.federated_broadcast(clip_range_lower),
           tff.federated_broadcast(clip_range_upper)))

      (agg_output_state, agg_output_result,
       agg_output_measurements) = inner_agg_next(state, clipped_value)

      # Clip the aggregate to the same range again (not considering summands).
      clipped_agg_output_result = tff.federated_map(
          modular_clip_by_value_tff,
          (agg_output_result, clip_range_lower, clip_range_upper))

      measurements = collections.OrderedDict(
          agg_process=agg_output_measurements)

      return tff.templates.MeasuredProcessOutput(
          state=agg_output_state,
          result=clipped_agg_output_result,
          measurements=tff.federated_zip(measurements))

    return next_fn


def modular_clip_by_value(value, clip_range_lower, clip_range_upper):

  def mod_clip(v):
    width = clip_range_upper - clip_range_lower
    period = tf.cast(tf.floor(v / width - clip_range_lower / width), v.dtype)
    v_mod_clipped = v - period * width
    return v_mod_clipped

  return tf.nest.map_structure(mod_clip, value)


def _create_get_clip_range_const(clip_range_lower, clip_range_upper):

  def get_clip_range():
    return (tff.federated_value(clip_range_lower, tff.SERVER),
            tff.federated_value(clip_range_upper, tff.SERVER))

  return get_clip_range
