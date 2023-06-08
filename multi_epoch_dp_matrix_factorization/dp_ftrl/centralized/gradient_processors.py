# Copyright 2023, Google LLC.
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
"""Accumulates gradient batches implement DP for centralized training."""
import abc
import collections
from typing import Any

import jax
from jax import numpy as jnp
import tensorflow_federated as tff

GradProcessingStateType = Any
GradientType = Any
PyTree = Any


# TODO(b/244756081): Remove DP from the classnames here when we implement
# nonprivate versions.
class DPGradientBatchProcessor(metaclass=abc.ABCMeta):
  """Interface for applying DP to a batch of gradients.

  These gradients may be literally per-example or averaged gradients
  per-microbatch; we will generally assume per-example gradients in the
  discussion below.

  This interface is responsible for clipping the incoming gradients to
  appropriate values, and adding appropriate per-step noise, to ensure
  a resulting gradient estimate is differentially private. For now, we assume
  that constructors of this interface know enough to account for the privacy
  budget of the entire training procedure.

  Implementations are not necessarily guaranteed to be portable; in particular,
  its methods may be implemented in non-serializable Python.

  This interface is responsible *only* for returning a differentially private
  value to be fed to an optimizer 'later', though in some cases (e.g., when
  factorizing a matrix which expressed SGDM as a linear operator acting on the
  gradient stream) certain facets of the optimization procedure are naturally
  implemented behind this interface. Since this interface produces results
  which will eventually be fed into an optimizer, implementations of this
  interface will usually inject so-called 'residual', or per-step, noise, a
  concept which is explained further in the docstring of
  `_create_residual_linear_query_dp_factory` in ../../tff_aggregator.py.
  """

  @abc.abstractmethod
  def init(self) -> GradProcessingStateType:
    """Initializes the gradient processor."""

  @abc.abstractmethod
  def apply(
      self, state: GradProcessingStateType, gradients: PyTree
  ) -> tuple[GradProcessingStateType, GradientType]:
    """Processes a PyTree of per-example gradients.

    The gradients are assumed to be represented as a PyTree whose leaves
    contain an array of per-example gradients corresponding to the appropriate
    model tensors.

    Args:
      state: Any state necessary for `apply` to compute the appropriate
        estimate. Assumed to be a value of the same Python type returned from
        `init`.
      gradients: A PyTree of gradient estimates. Assumed to represent
        per-example gradient values.

    Returns:
      A two tuple, whose first element is an 'updated' state which can be
      passed to the next invocation of `apply`, and whose second element is
      the differentially private estimate described above.
    """


def nested_odict_to_dict(nested_odict: ...) -> ...:
  """Converts a potentially nested odict to dict Python type."""
  if isinstance(nested_odict, collections.OrderedDict):
    return {k: nested_odict_to_dict(v) for k, v in nested_odict.items()}
  return nested_odict


class DPAggregatorBackedGradientProcessor(DPGradientBatchProcessor):
  """Implementation of gradient processor backed by a TFF aggregation process.

  The aggregator factory parameter will be passed a *single* value to aggregate;
  we will normalize the sensitivity *outside* of TFF, and therefore the
  aggregator must not be attempting to normalize the sensitivity based on the
  number of microbatches (or clients). If an aggregator constructor accepts,
  say, a num_clients argument, setting this value to 1 will be sufficient.
  """

  def __init__(
      self, dp_aggregator: tff.templates.AggregationProcess, l2_norm_clip: float
  ):
    """Initializes the processor.

    Args:
      dp_aggregator: A `tff.templates.AggregationProcess` which will inject
        noise to its arguments but *will not normalize* the result.
      l2_norm_clip: The global l2 norm clip used to clip the per-example
        gradients.
    """
    self._dp_aggregator = dp_aggregator
    self._clip_norm = l2_norm_clip

  def init(self) -> GradProcessingStateType:
    return self._dp_aggregator.initialize()

  def apply(
      self,
      state: GradProcessingStateType,
      gradients: PyTree,
  ) -> tuple[GradProcessingStateType, GradientType]:
    # TODO(b/244756081): This implementation more naturally integrates with
    # the DPSumQueries directly--we could, for example, remove the business
    # about not letting the aggregator normalize in the docstring. To follow up
    # and reevaluate the cost of swapping.
    global_grad_norms = jnp.sqrt(
        sum(
            [
                jnp.sum(jnp.square(x), axis=list(range(1, len(x.shape), 1)))
                for x in jax.tree_util.tree_leaves(gradients)
            ]
        )
    )
    num_elements_in_batch = jnp.size(global_grad_norms)

    def _broadcast_norms_and_clip_gradients(grad_element):
      shape_for_broadcasting = list(global_grad_norms.shape) + [1] * (
          len(grad_element.shape) - 1
      )
      reshaped_grad_norms = jnp.reshape(
          global_grad_norms, shape_for_broadcasting
      )
      raw_divide_result = jnp.divide(
          self._clip_norm * grad_element, reshaped_grad_norms
      )
      division_nans_replaced = jnp.nan_to_num(raw_divide_result, nan=0.0)
      return jnp.where(
          reshaped_grad_norms > self._clip_norm,
          division_nans_replaced,
          grad_element,
      )

    clipped_gradients = jax.tree_map(
        _broadcast_norms_and_clip_gradients, gradients
    )
    averaged_clipped_grad = jax.tree_map(
        lambda x: jnp.mean(x, axis=0), clipped_gradients
    )

    # We just use this as a noise generator now. We pass a single
    # client.
    process_output = self._dp_aggregator.next(
        state, [jax.tree_map(jnp.zeros_like, averaged_clipped_grad)]
    )
    noise = process_output.result
    # We have to perform this sensitivity normalization outside of TFF. Thus the
    # comments in the class docstring on ensuring *no* normalization is present
    # in the aggregator.
    normalized_noise = jax.tree_map(lambda x: x / num_elements_in_batch, noise)
    # Haiku returns a dict for model parameters, and Jax's tree-traversals (used
    # internally in optax) don't immediately interop between ODict and dict. So
    # we convert back to dict here, though it is a bit unfortunate.
    noised_clipped_gradients = jax.tree_map(
        jnp.add, nested_odict_to_dict(normalized_noise), averaged_clipped_grad
    )
    return process_output.state, noised_clipped_gradients


class NoPrivacyGradientProcessor(DPGradientBatchProcessor):
  """An implementation of jax mean in this interface."""

  def init(self) -> int:
    # The return value here carries no  meaning.
    return 0

  def apply(
      self, state: GradProcessingStateType, gradients: PyTree
  ) -> tuple[GradProcessingStateType, GradientType]:
    averaged_grad = jax.tree_map(lambda x: jnp.mean(x, axis=0), gradients)
    return state, averaged_grad
