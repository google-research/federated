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
"""A tff.templates.MeasuredProcess for collecting histograms of global model weights."""

import tensorflow as tf
import tensorflow_federated as tff


class HistogramModelBroadcastProcess(tff.templates.MeasuredProcess):
  """Aggregator reporting a histogram of the global model weights broadcasted.

  The created tff.templates.MeasuredProcess broadcasts the model at SERVER to
  CLIENTS.

  The process has empty state and returns a histogram of model weight values in
  measurements. The value returned in measurements is one histogram of all model
  weights.
  """

  def __init__(self, weights_type, mn=-1.0, mx=1.0, nbins=50):
    """Initializer for HistogramModelBroadcastProcess.

    Defines the tf.histogram_fixed_width bins and bounds.

    Args:
      weights_type: Type of model weights tensor.
      mn: A float that specifies the lower bound of the histogram.
      mx: A float that specifies the  upper bound of the histogram.
      nbins: An integer that specifies the number of bins in the histogram.
    """
    self._weights_type = weights_type
    self._min = mn
    self._max = mx
    self._nbins = nbins

    @tff.federated_computation()
    def init_fn():
      return tff.federated_value((), tff.SERVER)

    @tff.federated_computation(init_fn.type_signature.result,
                               tff.type_at_server(self._weights_type))
    def next_fn(state, weights):
      broadcast_weights = tff.federated_broadcast(weights)

      @tff.tf_computation
      def histogram_fn(weights):
        flattened = tf.concat(
            [tf.reshape(w, [1, -1]) for w in tf.nest.flatten(weights)], axis=1)
        return tf.histogram_fixed_width(flattened, [self._min, self._max],
                                        self._nbins)

      histogram = tff.federated_map(histogram_fn, weights)
      return tff.templates.MeasuredProcessOutput(state,
                                                 broadcast_weights,
                                                 histogram)

    super().__init__(init_fn, next_fn)
