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
"""A tff.aggregator for measuring zero run lengths in client model weight updates."""

import collections

import tensorflow as tf
import tensorflow_federated as tff

from compressed_communication.aggregators import cross_entropy
from compressed_communication.aggregators import entropy


class ZeroRunLengthSumFactory(tff.aggregators.UnweightedAggregationFactory):
  """Aggregator that computes the run length of zeros in the input tensor.

  The created `tff.templates.AggregationProcess` expects value to be a structure
  of integers, placed at CLIENTS. By default it sums up the client values and
  outputs the sum placed at SERVER. This aggregation process computes the run
  lengths of zeros on each client tensor. It uses the `entropy` utility
  functions to get bincounts of zero run lengths across clients and compute the
  entropy of these binned run lengths. The cross entropy of zero run lengths is
  also measured using the utility functions in `cross_entropy`.

  The process returns empty `state`. For computing the summed value `result`,
  implementation delegates to the `tff.federated_sum` operator. The process
  returns a dictionary in `measurements` that maps `zero_run_lengths` to the
  summed bincounts of zero run lengths, `entropy` to the entropy of zero run
  lengths across clients, `cross_entropy_gamma` to the average cross entropy
  of a client's zero run lengths with the Elias Gamma code and
  `cross_entropy_delta` to the average cross entropy of a client's zero run
  lengths with the Elias Delta code.
  """

  def create(self, value_type):
    if not tff.types.is_structure_of_integers(
        value_type) or not value_type.is_tensor():
      raise ValueError("Expect value_type to be an integer tensor, "
                       f"found {value_type}.")

    @tff.tf_computation(value_type)
    def get_zero_run_lengths(value):
      # Append nonzero value at start and end to capture length of any leading
      # or trailing zeros.
      padded_value = tf.concat(
          [tf.constant([1]), value, tf.constant([1])], axis=0)
      nonzero_indices = tf.where(tf.not_equal(padded_value, 0))
      zero_run_lengths = nonzero_indices[1:] - nonzero_indices[:-1]
      # Account for case where there are no trailing zeros.
      zero_run_lengths = tf.cond(
          tf.equal(zero_run_lengths[-1], 1), lambda: zero_run_lengths[:-1],
          lambda: zero_run_lengths)
      return tf.cast(zero_run_lengths, tf.int32)

    @tff.federated_computation()
    def init_fn():
      return tff.federated_value((), tff.SERVER)

    @tff.federated_computation(init_fn.type_signature.result,
                               tff.type_at_clients(value_type))
    def next_fn(state, value):
      summed_value = tff.federated_sum(value)

      client_zero_run_lengths = tff.federated_map(get_zero_run_lengths, value)

      client_zero_run_length_counts = tff.federated_map(
          entropy.get_bincounts, client_zero_run_lengths)
      summed_zero_run_length_counts = entropy.sum_bincounts(
          client_zero_run_length_counts)

      include_zero_bin = tff.federated_value(False, tff.SERVER)
      zero_run_length_entropy = tff.federated_map(
          entropy.compute_entropy,
          (summed_zero_run_length_counts, include_zero_bin))

      cross_entropy_gamma = tff.federated_map(
          cross_entropy.compute_cross_entropy_gamma, client_zero_run_lengths)
      cross_entropy_gamma = tff.federated_mean(cross_entropy_gamma)

      cross_entropy_delta = tff.federated_map(
          cross_entropy.compute_cross_entropy_delta, client_zero_run_lengths)
      cross_entropy_delta = tff.federated_mean(cross_entropy_delta)

      measurements = collections.OrderedDict(
          zero_run_lengths=summed_zero_run_length_counts,
          entropy=zero_run_length_entropy,
          cross_entropy_gamma=cross_entropy_gamma,
          cross_entropy_delta=cross_entropy_delta)

      return tff.templates.MeasuredProcessOutput(
          state=state,
          result=summed_value,
          measurements=tff.federated_zip(measurements))

    return tff.templates.AggregationProcess(init_fn, next_fn)
