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
"""A tff.aggregator for computing cross entropy of client model weight updates with source codes."""

import collections
import tensorflow as tf
import tensorflow_federated as tff


@tff.tf_computation
def get_positive(value):
  value = tf.abs(value)
  mask = tf.greater(value, 0)
  return tf.boolean_mask(value, mask)


@tff.tf_computation
def compute_cross_entropy_gamma(value):
  """Compute the cross entropy of a given sample with the Elias Gamma code."""

  def elias_gamma_code_length(x):
    """Code length of Elias Gamma code."""
    x = tf.cast(x, tf.float32)
    return 1. + 2. * tf.math.floor(tf.math.log(x) / tf.math.log(2.))

  num_total = tf.cast(tf.size(value), tf.float32)
  positive_value = get_positive(value)
  return tf.reduce_sum(elias_gamma_code_length(positive_value)) / num_total


@tff.tf_computation
def compute_cross_entropy_delta(value):
  """Compute the cross entropy of a given sample with the Elias Delta code."""

  def elias_delta_code_length(x):
    """Code length of Elias Delta code."""
    x = tf.cast(x, tf.float32)
    log2 = tf.math.floor(tf.math.log(x) / tf.math.log(2.))
    return 1. + log2 + 2. * tf.math.floor(
        tf.math.log(log2 + 1.) / tf.math.log(2.))

  num_total = tf.cast(tf.size(value), tf.float32)
  positive_value = get_positive(value)
  return tf.reduce_sum(elias_delta_code_length(positive_value)) / num_total


class CrossEntropyFactory(tff.aggregators.UnweightedAggregationFactory):
  """Aggregator that computes entropy of input tensors.

  The created `tff.templates.AggregationProcess` sums values, expected to be
  a structure of integers, placed at CLIENTS, and outputs the sum placed at
  SERVER.

  The process has empty `state`. For computing the summed value `result`,
  implementation delegates to the `tff.federated_sum` operator. The process
  returns a dictionary in `measurements` that reports the averages of the cross
  entropy of client values with source codes across participating clients. The
  key `cross_entropy_gamma` maps to the cross entropy of client values with the
  Elias Gamma code and `cross_entropy_delta` maps to the cross entropy of client
  values with the Elias Delta code. Information about the Elias Gamma and Elias
  Delta codes can be found here: https://ieeexplore.ieee.org/document/1055349.
  """

  def create(self, value_type):
    if not tff.types.is_structure_of_integers(
        value_type) or not value_type.is_tensor():
      raise ValueError("Expect value_type to be an integer tensor, "
                       f"found {value_type}.")

    @tff.federated_computation()
    def init_fn():
      return tff.federated_value((), tff.SERVER)

    @tff.federated_computation(init_fn.type_signature.result,
                               tff.type_at_clients(value_type))
    def next_fn(state, value):
      summed_value = tff.federated_sum(value)

      cross_entropy_gamma = tff.federated_map(compute_cross_entropy_gamma,
                                              value)
      cross_entropy_gamma = tff.federated_mean(cross_entropy_gamma)
      cross_entropy_delta = tff.federated_map(compute_cross_entropy_delta,
                                              value)
      cross_entropy_delta = tff.federated_mean(cross_entropy_delta)

      measurements = collections.OrderedDict(
          cross_entropy_gamma=cross_entropy_gamma,
          cross_entropy_delta=cross_entropy_delta)

      return tff.templates.MeasuredProcessOutput(
          state=state,
          result=summed_value,
          measurements=tff.federated_zip(measurements))

    return tff.templates.AggregationProcess(init_fn, next_fn)
