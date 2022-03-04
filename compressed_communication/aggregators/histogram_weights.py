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
"""A tff.aggregator for collecting summed histograms of client model weights."""

import tensorflow as tf
import tensorflow_federated as tff


class HistogramWeightsFactory(tff.aggregators.UnweightedAggregationFactory):
  """Aggregator reporting a histogram of client weights as a metric.

  The created tff.templates.AggregationProcess sums values placed at CLIENTS,
  and outputs the sum placed at SERVER.

  The process has empty state and returns summed histograms of client values in
  measurements. For computing both the resulting summed value and summed
  histograms, implementation delegates to the tff.federated_sum operator.

  The value returned in measurements is one histogram if the client value_type
  is a single tensor of weights, or a list of histograms - one for each layer -
  if the client value_type is a struct of weight tensors.
  """

  def __init__(self, mn=-1.0, mx=1.0, nbins=50):
    """Initializer for HistogramWeightsFactory.

    Defines the tf.histogram_fixed_width bins and bounds.

    Args:
      mn: A float that specifies the lower bound of the histogram.
      mx: A float that specifies the  upper bound of the histogram.
      nbins: An integer that specifies the number of bins in the histogram.
    """
    self._min = mn
    self._max = mx
    self._nbins = nbins

  def create(self, value_type):
    if not (tff.types.is_structure_of_floats(value_type) or
            (value_type.is_tensor() and value_type.dtype == tf.float32)):
      raise ValueError("Expect value_type to be float tensor or structure of "
                       f"float tensors, found {value_type}.")

    @tff.federated_computation()
    def init_fn():
      return tff.federated_value((), tff.SERVER)

    @tff.tf_computation(value_type)
    def compute_client_histogram(value):
      bounds = [self._min, self._max]
      histogram_fn = lambda x: tf.histogram_fixed_width(x, bounds, self._nbins)
      return tf.nest.map_structure(histogram_fn, value)

    @tff.federated_computation(init_fn.type_signature.result,
                               tff.type_at_clients(value_type))
    def next_fn(state, value):
      summed_value = tff.federated_sum(value)

      client_histograms = tff.federated_map(compute_client_histogram, value)
      server_histograms = tff.federated_sum(client_histograms)

      return tff.templates.MeasuredProcessOutput(
          state=state, result=summed_value, measurements=server_histograms)

    return tff.templates.AggregationProcess(init_fn, next_fn)

