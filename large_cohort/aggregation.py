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
"""Utilities for specialized aggregation used in large cohort experiments."""

import collections
import math

import tensorflow as tf
import tensorflow_federated as tff


@tff.tf_computation(tf.float32, tf.float32)
def compute_average_cosine_similarity(square_norm_of_sum, num_vectors):
  """Calculate the average cosine similarity between unit length vectors.

  Args:
    square_norm_of_sum: The squared norm of the sum of the normalized vectors.
    num_vectors: The number of vectors the sum was taken over.

  Returns:
    A float representing the average pairwise cosine similarity among all
      vectors.
  """
  return (square_norm_of_sum - num_vectors) / (
      num_vectors * (num_vectors - 1.0))


class MeasuringMeanFactory(tff.aggregators.MeanFactory):
  """A MeanFactory that computes additional metrics about the model_updates."""

  def create(self, model_update_type,
             weight_type) -> tff.templates.AggregationProcess:
    model_update_sum_process = self._value_sum_factory.create(model_update_type)
    weight_sum_process = self._weight_sum_factory.create(weight_type)

    @tff.federated_computation
    def init_fn():
      """Initialize the state based on the inner processes."""
      state = collections.OrderedDict(
          value_sum_process=model_update_sum_process.initialize(),
          weight_sum_process=weight_sum_process.initialize())
      return tff.federated_zip(state)

    @tff.tf_computation(model_update_type)
    def calculate_global_norm(model_update):
      """Calculate the global norm across all layers of the model update."""
      return tf.linalg.global_norm(tf.nest.flatten(model_update))

    @tff.tf_computation(model_update_type)
    def normalize_vector(model_update):
      update_norm = calculate_global_norm(model_update)
      return tf.nest.map_structure(lambda a: a / update_norm, model_update)

    @tff.tf_computation(model_update_type)
    def calculate_square_global_norm(model_update):
      """Calculate the squared global norm across all layers of a model update."""
      # We compute this directly in order to circumvent precision issues
      # incurred by taking square roots and then re-squaring.
      return calculate_global_norm(model_update)**2

    @tff.federated_computation(
        tff.type_at_clients(model_update_type), tff.type_at_clients(tf.float32))
    def average_norm(model_update, client_weight):
      """Compute average of the global norm of the model updates."""
      client_norms = tff.federated_map(calculate_global_norm, model_update)
      return tff.federated_mean(client_norms, client_weight)

    @tff.tf_computation(model_update_type, weight_type)
    def divide_no_nan(model_update_sum, weight_sum):
      """Compute the mean of the model update and the norm of the mean."""
      mean_model_update = tf.nest.map_structure(
          lambda w: tf.math.divide_no_nan(w, tf.cast(weight_sum, w.dtype)),
          model_update_sum)
      norm_of_mean = tf.linalg.global_norm(tf.nest.flatten(mean_model_update))
      return mean_model_update, norm_of_mean

    @tff.tf_computation(model_update_type, weight_type)
    def multiply_weight(model_update, weight):
      """Multiply each layer of the model weight by the client weight."""
      return tf.nest.map_structure(lambda w: w * tf.cast(weight, w.dtype),
                                   model_update)

    @tff.federated_computation(init_fn.type_signature.result,
                               tff.type_at_clients(model_update_type),
                               tff.type_at_clients(weight_type))
    def next_fn(state, model_update, weight):
      # First weight the model updates the same way they would be weighted
      # using `tff.federated_mean()`.
      weighted_model_updates = tff.federated_map(multiply_weight,
                                                 (model_update, weight))
      model_update_sum_output = model_update_sum_process.next(
          state['value_sum_process'], weighted_model_updates)
      weight_sum_output = weight_sum_process.next(state['weight_sum_process'],
                                                  weight)
      mean_model_update, norm_of_mean_client_update = tff.federated_map(
          divide_no_nan,
          (model_update_sum_output.result, weight_sum_output.result))
      mean_of_norm_of_client_update = average_norm(model_update, weight)
      new_state = tff.federated_zip(
          collections.OrderedDict(
              value_sum_process=model_update_sum_output.state,
              weight_sum_process=weight_sum_output.state))

      normalized_updates = tff.federated_map(normalize_vector, model_update)
      sum_of_normalized_updates = tff.federated_sum(normalized_updates)
      square_norm_of_sum = tff.federated_map(calculate_square_global_norm,
                                             sum_of_normalized_updates)
      num_clients = tff.federated_sum(tff.federated_value(1.0, tff.CLIENTS))
      average_cosine_similarity = tff.federated_map(
          compute_average_cosine_similarity, (square_norm_of_sum, num_clients))

      measurements = tff.federated_zip(
          collections.OrderedDict(
              # Add our custom metrics first.
              mean_of_norm_of_client_update=mean_of_norm_of_client_update,
              norm_of_mean_client_update=norm_of_mean_client_update,
              average_cosine_similarity=average_cosine_similarity,
              # Pass through the next values the same as MeanFactory.
              mean_value=model_update_sum_output.measurements,
              mean_weight=weight_sum_output.measurements))
      return tff.templates.MeasuredProcessOutput(
          state=new_state, result=mean_model_update, measurements=measurements)

    return tff.templates.AggregationProcess(init_fn, next_fn)


def create_aggregator(
    zeroing: bool = True,
    clipping: bool = True) -> tff.aggregators.WeightedAggregationFactory:
  """Creates a custom aggregator for the large cohort experiments.

  This aggregator replicates the behavior of `tff.learning.robust_aggregator`
  with additional metrics that are computed _after_ clipping and zeroing are
  applied.

  Args:
    zeroing: A boolean to toggle whether to add zeroing out extreme client
      updates.
    clipping: A boolean to toggle whether to add clipping to large client
      updates.

  Returns:
    A `tff.aggregators.WeightedAggregationFactory`.
  """
  factory = MeasuringMeanFactory()
  # Same as `tff.aggregators.robust_aggregator` as of 2021-04-29.
  if clipping:
    clipping_norm = tff.aggregators.PrivateQuantileEstimationProcess.no_noise(
        initial_estimate=1.0, target_quantile=0.8, learning_rate=0.2)
    factory = tff.aggregators.clipping_factory(clipping_norm, factory)
  if zeroing:
    zeroing_norm = tff.aggregators.PrivateQuantileEstimationProcess.no_noise(
        initial_estimate=10.0,
        target_quantile=0.98,
        learning_rate=math.log(10.0),
        multiplier=2.0,
        increment=1.0)
    factory = tff.aggregators.zeroing_factory(zeroing_norm, factory)
  return factory  # pytype: disable=bad-return-type  # gen-stub-imports
