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
"""A tff.aggregator for computing entropy of client model weight updates."""

import collections
import tensorflow as tf
import tensorflow_federated as tff

from compressed_communication.aggregators import cross_entropy


_BINCOUNT_TYPE = tff.TensorType(tf.int32, shape=(None,))


@tff.tf_computation
def get_bincounts(value):
  return tf.math.bincount(tf.math.abs(value))


def sum_bincounts(value):
  """Federated computation to sum bincounts of unknown length across clients."""

  @tff.tf_computation
  def get_accumulator():
    # Relying upon broadcasting here for summation with the accumulator.
    return tf.zeros((0,), dtype=tf.int32)

  @tff.tf_computation(_BINCOUNT_TYPE, _BINCOUNT_TYPE)
  def add_bincounts(bincounts_1, bincounts_2):
    return tf.reduce_sum(
        tf.ragged.stack([bincounts_1, bincounts_2], axis=0), axis=0)

  @tff.tf_computation(_BINCOUNT_TYPE)
  def report_bincounts(summed_bincounts):
    return summed_bincounts

  return tff.federated_aggregate(
      value,
      zero=get_accumulator(),
      accumulate=add_bincounts,
      merge=add_bincounts,
      report=report_bincounts)


@tff.tf_computation
@tf.function
def compute_entropy(bincounts, include_zeros):
  """Compute the entropy of a distribution of non-zero integers.

  Args:
    bincounts: An array containing the occurrence counts of each integer,
      corresponding to the index of the array.
    include_zeros: A boolean indicating whether to include the zero bin in the
      entropy calculation.

  Returns:
    entropy: A float measurement of the entropy of the given distribution.
  """
  num_total = tf.cast(tf.reduce_sum(bincounts), tf.float64)
  if not include_zeros:
    bincounts = bincounts[1:]
  mask = tf.greater(bincounts, 0)
  nonzero_bincounts = tf.cast(
      tf.boolean_mask(bincounts, mask), tf.float64)
  num_nonzero = tf.cast(tf.reduce_sum(nonzero_bincounts), tf.float64)
  log_nonzero_bincounts = tf.math.log(nonzero_bincounts)
  log_prob = log_nonzero_bincounts - tf.reduce_logsumexp(
      log_nonzero_bincounts)
  entropy = tf.math.reduce_sum(
      log_prob * tf.exp(log_prob)) / -tf.math.log(tf.cast(2, tf.float64))
  return entropy * num_nonzero / num_total


class EntropyFactory(tff.aggregators.UnweightedAggregationFactory):
  """Aggregator that computes entropy of input tensors.

  The created `tff.templates.AggregationProcess` computes the entropy of values,
  expected to be a structure of integers, placed at CLIENTS. By default it uses
  the `SumFactory` as its inner aggregation factory to compute the sum over
  client values and output the sum placed at SERVER. If this aggregation factory
  is initialized with `compute_cross_entropy` set to True, then the inner
  aggregation factory is set to the `CrossEntropyFactory`, which sums over
  client values and computes additional metrics.

  The process returns the inner aggregation process' `state` and `result`, and
  a dictionary mapping `entropy` to the entropy across all client values and
  optionally `cross_entropy` to the inner aggregation process' `measurements`
  in `measurements`.
  """

  def __init__(self, include_zeros=False, compute_cross_entropy=False):
    """Initializer for EntropyFactory.

    Args:
      include_zeros: Boolean indicating whether to include zeros within the
        client values for entropy measurement.
      compute_cross_entropy: Boolean indicating whether to compute cross entropy
        of client values with source codes.
    """
    self.include_zeros = include_zeros
    self.compute_cross_entropy = compute_cross_entropy
    self.inner_agg_factory = tff.aggregators.SumFactory()
    if compute_cross_entropy:
      self.inner_agg_factory = cross_entropy.CrossEntropyFactory()

  def create(self, value_type):
    if not tff.types.is_structure_of_integers(
        value_type) or not value_type.is_tensor():
      raise ValueError("Expect value_type to be an integer tensor, "
                       f"found {value_type}.")

    inner_agg_process = self.inner_agg_factory.create(value_type)

    @tff.federated_computation()
    def init_fn():
      return inner_agg_process.initialize()

    @tff.federated_computation(init_fn.type_signature.result,
                               tff.type_at_clients(value_type))
    def next_fn(state, value):
      client_counts = tff.federated_map(get_bincounts, value)
      summed_counts = sum_bincounts(client_counts)

      entropy = tff.federated_map(
          compute_entropy,
          (summed_counts, tff.federated_value(self.include_zeros, tff.SERVER)))
      measurements = collections.OrderedDict(entropy=entropy)

      inner_agg_output = inner_agg_process.next(state, value)

      if self.compute_cross_entropy:
        measurements["cross_entropy"] = inner_agg_output.measurements

      return tff.templates.MeasuredProcessOutput(
          state=inner_agg_output.state,
          result=inner_agg_output.result,
          measurements=tff.federated_zip(measurements))

    return tff.templates.AggregationProcess(init_fn, next_fn)
