# Copyright 2018, The TensorFlow Federated Authors.
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
"""General utilities specific to the manipulation of tensors and operators."""

import functools

import tensorflow as tf


@tf.function
def zero_all_if_any_non_finite(structure):
  """Zeroes out all entries in input if any are not finite.

  Args:
    structure: A structure supported by tf.nest.

  Returns:
     A tuple (input, 0) if all entries are finite or the structure is empty, or
     a tuple (zeros, 1) if any non-finite entries were found.
  """
  flat = tf.nest.flatten(structure)
  if not flat:
    return (structure, tf.constant(0))
  flat_bools = [tf.reduce_all(tf.math.is_finite(t)) for t in flat]
  all_finite = functools.reduce(tf.logical_and, flat_bools)
  if all_finite:
    return (structure, tf.constant(0))
  else:
    return (tf.nest.map_structure(tf.zeros_like, structure), tf.constant(1))
