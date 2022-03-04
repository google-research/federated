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
"""Utilities for building aggregators for compression experiments."""

import collections
import math

import numpy as np
import tensorflow_federated as tff

from compressed_communication import builder_configs
from compressed_communication.aggregators import entropy
from compressed_communication.aggregators import group
from compressed_communication.aggregators import histogram_weights
from compressed_communication.aggregators import quantize
from compressed_communication.aggregators import quantize_encode
from compressed_communication.aggregators import quantize_encode_client_lambda
from compressed_communication.aggregators.comparison_methods import drive
from compressed_communication.aggregators.comparison_methods import one_bit_sgd
from compressed_communication.aggregators.comparison_methods import qsgd
from compressed_communication.aggregators.comparison_methods import terngrad
from compressed_communication.aggregators.comparison_methods import three_lc
from compressed_communication.aggregators.comparison_methods import top_k


def configure_aggregator(factory: tff.aggregators.AggregationFactory,
                         rotation: str = "identity",
                         concatenate: bool = True,
                         zeroing: bool = True,
                         clipping: bool = True,
                         weighted: bool = True,
                         group_layers: bool = False,
                         task: str = ""):
  """Optionally wraps an aggregation factory with additional aggregation methods.

  Args:
    factory: The inner aggregation factory.
    rotation: A string to specify what rotation (if any) to apply, one of
      ["dft", "hadamard" or "identity"].
    concatenate: A boolean indicating whether to concatenate all tensors in
      client update to a single tensor (`True`) or not (`False`) within the
      aggregation process.
    zeroing: A boolean indicating whether to add zeroing out extreme client
      updates (`True`) or not (`False`).
    clipping: A boolean indicating whether to add clipping to large client
      updates (`True`) or not (`False`).
    weighted: A boolean indicating whether client model weights should be
      averaged in a weighted manner (`True`) or unweighted manner (`False`).
    group_layers: A booolean to toggle whether to group client updates by layer
      type and apply aggregation factory separately to each layer type.
    task: A string indicating to which task this aggregation factory is applied,
      one of ["emnist_character" or "stackoverflow_word"].

  Returns:
    A `tff.aggregators.AggregationFactory`.
  """
  if rotation == "hadamard":
    factory = tff.aggregators.HadamardTransformFactory(factory)
  elif rotation == "dft":
    factory = tff.aggregators.DiscreteFourierTransformFactory(factory)
  else:
    if rotation != "identity":
      raise ValueError(
          "Provided `rotation` must be one of 'dft', 'hadamard' or 'identity'.")

  if concatenate:
    factory = tff.aggregators.concat_factory(factory)

  if group_layers:
    if task == "stackoverflow_word":
      factory = group.GroupFactory(
          grouped_indices=collections.OrderedDict(
              embedding=[0], kernel=[1, 4, 6], recurrent=[2], bias=[3, 5, 7]),
          inner_agg_factories=collections.OrderedDict(
              embedding=factory,
              kernel=factory,
              recurrent=factory,
              bias=factory))
    elif task == "emnist_character":
      factory = group.GroupFactory(
          grouped_indices=collections.OrderedDict(
              kernel=[0, 2, 4, 6], bias=[1, 3, 5, 7]),
          inner_agg_factories=collections.OrderedDict(
              kernel=factory, bias=factory))
    else:
      raise ValueError(f"Layer type grouping has not been defined for: {task}. "
                       "Supported tasks: emnist_character, stackoverflow_word")

  if weighted:
    factory = tff.aggregators.MeanFactory(factory)
  else:
    factory = tff.aggregators.UnweightedMeanFactory(factory)

  # Same as `tff.aggregators.robust_aggregator` as of 2021-07-21.
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

  return factory


def build_histogram_aggregator(
    rotation: str = "identity",
    concatenate: bool = True,
    zeroing: bool = True,
    clipping: bool = True,
    weighted: bool = True) -> tff.aggregators.AggregationFactory:
  """Creates an aggregation factory to track histogram of client updates.

  Args:
    rotation: A string to specify what rotation (if any) to apply, one of
      ["dft", "hadamard" or "identity"].
    concatenate: A boolean to toggle whether to concatenate all tensors in
      client update to a single tensor within the aggregation process.
    zeroing: A boolean to toggle whether to add zeroing out extreme client
      updates.
    clipping: A boolean to toggle whether to add clipping to large client
      updates.
    weighted: A boolean to toggle whether client model weights should be
      averaged in a weighted manner (`True`) or unweighted manner (`False`).

  Returns:
    A `tff.aggregators.AggregationFactory`.
  """
  factory = histogram_weights.HistogramWeightsFactory(
      mn=-1.0, mx=1.0, nbins=2001)

  return configure_aggregator(factory, rotation, concatenate, zeroing, clipping,
                              weighted)


def build_entropy_cross_entropy_aggregator(
    step_size: float = 0.5,
    rounding_type: str = "stochastic",
    rotation: str = "identity",
    concatenate: bool = True,
    zeroing: bool = True,
    clipping: bool = True,
    weighted: bool = True,
    group_layers: bool = True,
    task: str = "emnist_character") -> tff.aggregators.AggregationFactory:
  """Creates an aggregation factory to quantize and compute entropy and cross-entropy.

  Args:
    step_size: A float that determines the step size between adjacent
      quantization levels to be used as the initial scale factor.
    rounding_type: A string that determines what type of rounding to apply,
      one of ["uniform", "stochastic", "dithered"].
    rotation: A string to specify what rotation (if any) to apply, one of
      ["dft", "hadamard" or "identity"].
    concatenate: A boolean indicating whether to concatenate all tensors in
      client update to a single tensor (`True`) or not (`False`) within the
      aggregation process.
    zeroing: A boolean indicating whether to add zeroing out extreme client
      updates (`True`) or not (`False`).
    clipping: A boolean indicating whether to add clipping to large client
      updates (`True`) or not (`False`).
    weighted: A boolean indicating whether client model weights should be
      averaged in a weighted manner (`True`) or unweighted manner (`False`).
    group_layers: A booolean to toggle whether to group client updates by layer
      type and apply aggregation factory separately to each layer type.
    task: A string indicating to which task this aggregation factory is applied,
      one of ["emnist_character" or "stackoverflow_word"].

  Returns:
    A `tff.aggregators.AggregationFactory`.
  """
  if rounding_type not in ["uniform", "stochastic", "dithered"]:
    raise ValueError("Expected `rounding_type` to be one one of [\"uniform\", "
                     f"\"stochastic\", \"dithered\"], found {rounding_type}.")

  factory = quantize.QuantizeFactory(
      step_size,
      entropy.EntropyFactory(include_zeros=False, compute_cross_entropy=True),
      rounding_type)

  return configure_aggregator(factory, rotation, concatenate, zeroing, clipping,
                              weighted, group_layers, task)


def build_rotation_ablation_aggregator(
    step_size: float = 0.5,
    rounding_type: str = "uniform",
    rotation: str = "identity",
    concatenate: bool = True,
    zeroing: bool = True,
    clipping: bool = True,
    weighted: bool = True) -> tff.aggregators.AggregationFactory:
  """Creates an aggregation factory to quantize and compute entropy and cross-entropy.

  Args:
    step_size: A float that determines the step size between adjacent
      quantization levels to be used as the initial scale factor.
    rounding_type: A string that determines what type of rounding to apply,
      one of ["uniform", "stochastic", "dithered"].
    rotation: A string to specify what rotation (if any) to apply, one of
      ["dft", "hadamard" or "identity"].
    concatenate: A boolean to toggle whether to concatenate all tensors in
      client update to a single tensor within the aggregation process.
    zeroing: A boolean to toggle whether to add zeroing out extreme client
      updates.
    clipping: A boolean to toggle whether to add clipping to large client
      updates.
    weighted: A boolean to toggle whether client model weights should be
      averaged in a weighted manner (`True`) or unweighted manner (`False`).

  Returns:
    A `tff.aggregators.AggregationFactory`.
  """
  if rounding_type not in ["uniform", "stochastic", "dithered"]:
    raise ValueError("Expected `rounding_type` to be one one of [\"uniform\", "
                     f"\"stochastic\", \"dithered\"], found {rounding_type}.")

  factory = quantize.QuantizeFactory(step_size,
                                     entropy.EntropyFactory(include_zeros=True),
                                     rounding_type)

  return configure_aggregator(factory, rotation, concatenate, zeroing, clipping,
                              weighted)


def build_no_compression_aggregator(
    rotation: str = "identity",
    concatenate: bool = True,
    zeroing: bool = True,
    clipping: bool = True,
    weighted: bool = True) -> tff.aggregators.AggregationFactory:
  """Creates an aggregation factory for no compression.

  Args:
    rotation: A string to specify what rotation (if any) to apply, one of
      ["dft", "hadamard" or "identity"].
    concatenate: A boolean indicating whether to concatenate all tensors in
      client update to a single tensor (`True`) or not (`False`) within the
      aggregation process.
    zeroing: A boolean indicating whether to add zeroing out extreme client
      updates (`True`) or not (`False`).
    clipping: A boolean indicating whether to add clipping to large client
      updates (`True`) or not (`False`).
    weighted: A boolean indicating whether client model weights should be
      averaged in a weighted manner (`True`) or unweighted manner (`False`).

  Returns:
    A `tff.aggregators.AggregationFactory`.
  """
  factory = tff.aggregators.SumFactory()

  return configure_aggregator(factory, rotation, concatenate, zeroing, clipping,
                              weighted)


def build_drive_aggregator(
    rotation: str = "hadamard",
    concatenate: bool = True,
    zeroing: bool = True,
    clipping: bool = True,
    weighted: bool = True) -> tff.aggregators.AggregationFactory:
  """Creates an aggregation factory for comparing to DRIVE.

  Args:
    rotation: A string to specify what rotation (if any) to apply, one of
      ["dft", "hadamard" or "identity"].
    concatenate: A boolean indicating whether to concatenate all tensors in
      client update to a single tensor (`True`) or not (`False`) within the
      aggregation process.
    zeroing: A boolean indicating whether to add zeroing out extreme client
      updates (`True`) or not (`False`).
    clipping: A boolean indicating whether to add clipping to large client
      updates (`True`) or not (`False`).
    weighted: A boolean indicating whether client model weights should be
      averaged in a weighted manner (`True`) or unweighted manner (`False`).

  Returns:
    A `tff.aggregators.AggregationFactory`.
  """
  factory = drive.DRIVEFactory()
  return configure_aggregator(factory, rotation, concatenate, zeroing, clipping,
                              weighted)


def build_one_bit_sgd_aggregator(
    rotation: str = "identity",
    concatenate: bool = True,
    zeroing: bool = True,
    clipping: bool = True,
    weighted: bool = True) -> tff.aggregators.AggregationFactory:
  """Creates an aggregation factory for comparing to 1bit-SGD.

  Args:
    rotation: A string to specify what rotation (if any) to apply, one of
      ["dft", "hadamard" or "identity"].
    concatenate: A boolean indicating whether to concatenate all tensors in
      client update to a single tensor (`True`) or not (`False`) within the
      aggregation process.
    zeroing: A boolean indicating whether to add zeroing out extreme client
      updates (`True`) or not (`False`).
    clipping: A boolean indicating whether to add clipping to large client
      updates (`True`) or not (`False`).
    weighted: A boolean indicating whether client model weights should be
      averaged in a weighted manner (`True`) or unweighted manner (`False`).

  Returns:
    A `tff.aggregators.AggregationFactory`.
  """
  factory = one_bit_sgd.OneBitSGDFactory()
  return configure_aggregator(factory, rotation, concatenate, zeroing, clipping,
                              weighted)


def build_qsgd_aggregator(
    num_steps: float,
    rotation: str = "identity",
    concatenate: bool = True,
    zeroing: bool = True,
    clipping: bool = True,
    weighted: bool = True) -> tff.aggregators.AggregationFactory:
  """Creates an aggregation factory for comparing to QSGD.

  Args:
    num_steps: A float specifying the number of steps to quantize to.
    rotation: A string to specify what rotation (if any) to apply, one of
      ["dft", "hadamard" or "identity"].
    concatenate: A boolean indicating whether to concatenate all tensors in
      client update to a single tensor (`True`) or not (`False`) within the
      aggregation process.
    zeroing: A boolean indicating whether to add zeroing out extreme client
      updates (`True`) or not (`False`).
    clipping: A boolean indicating whether to add clipping to large client
      updates (`True`) or not (`False`).
    weighted: A boolean indicating whether client model weights should be
      averaged in a weighted manner (`True`) or unweighted manner (`False`).

  Returns:
    A `tff.aggregators.AggregationFactory`.
  """
  factory = qsgd.QSGDFactory(num_steps=num_steps)
  return configure_aggregator(factory, rotation, concatenate, zeroing, clipping,
                              weighted)


def build_terngrad_aggregator(
    rotation: str = "identity",
    concatenate: bool = True,
    zeroing: bool = True,
    clipping: bool = True,
    weighted: bool = True) -> tff.aggregators.AggregationFactory:
  """Creates an aggregation factory for comparing to TernGrad.

  Args:
    rotation: A string to specify what rotation (if any) to apply, one of
      ["dft", "hadamard" or "identity"].
    concatenate: A boolean indicating whether to concatenate all tensors in
      client update to a single tensor (`True`) or not (`False`) within the
      aggregation process.
    zeroing: A boolean indicating whether to add zeroing out extreme client
      updates (`True`) or not (`False`).
    clipping: A boolean indicating whether to add clipping to large client
      updates (`True`) or not (`False`).
    weighted: A boolean indicating whether client model weights should be
      averaged in a weighted manner (`True`) or unweighted manner (`False`).

  Returns:
    A `tff.aggregators.AggregationFactory`.
  """
  factory = terngrad.TernGradFactory()
  return configure_aggregator(factory, rotation, concatenate, zeroing, clipping,
                              weighted)


def build_three_lc_aggregator(
    sparsity_factor: float,
    rotation: str = "identity",
    concatenate: bool = True,
    zeroing: bool = True,
    clipping: bool = True,
    weighted: bool = True) -> tff.aggregators.AggregationFactory:
  """Creates an aggregation factory for comparing to 3LC.

  Args:
    sparsity_factor: A float specifying what sparsity level to quantize to.
    rotation: A string to specify what rotation (if any) to apply, one of
      ["dft", "hadamard" or "identity"].
    concatenate: A boolean indicating whether to concatenate all tensors in
      client update to a single tensor (`True`) or not (`False`) within the
      aggregation process.
    zeroing: A boolean indicating whether to add zeroing out extreme client
      updates (`True`) or not (`False`).
    clipping: A boolean indicating whether to add clipping to large client
      updates (`True`) or not (`False`).
    weighted: A boolean indicating whether client model weights should be
      averaged in a weighted manner (`True`) or unweighted manner (`False`).

  Returns:
    A `tff.aggregators.AggregationFactory`.
  """
  factory = three_lc.ThreeLCFactory(sparsity_factor=sparsity_factor)
  return configure_aggregator(factory, rotation, concatenate, zeroing, clipping,
                              weighted)


def build_top_k_aggregator(
    fraction_to_select: float,
    rotation: str = "identity",
    concatenate: bool = True,
    zeroing: bool = True,
    clipping: bool = True,
    weighted: bool = True) -> tff.aggregators.AggregationFactory:
  """Creates an aggregation factory for comparing to Top-K.

  Args:
    fraction_to_select: A float specifying what fraction of elements to select
      on each client.
    rotation: A string to specify what rotation (if any) to apply, one of
      ["dft", "hadamard" or "identity"].
    concatenate: A boolean indicating whether to concatenate all tensors in
      client update to a single tensor (`True`) or not (`False`) within the
      aggregation process.
    zeroing: A boolean indicating whether to add zeroing out extreme client
      updates (`True`) or not (`False`).
    clipping: A boolean indicating whether to add clipping to large client
      updates (`True`) or not (`False`).
    weighted: A boolean indicating whether client model weights should be
      averaged in a weighted manner (`True`) or unweighted manner (`False`).

  Returns:
    A `tff.aggregators.AggregationFactory`.
  """
  factory = top_k.TopKFactory(fraction_to_select=fraction_to_select)
  return configure_aggregator(factory, rotation, concatenate, zeroing, clipping,
                              weighted)


def build_quantization_encode_aggregator(
    step_size: float = 0.5,
    rounding_type: str = "uniform",
    normalization_type: str = "constant",
    step_size_sched: str = "fixed",
    step_size_sched_hparam: float = 0.,
    min_step_size: float = 0.01,
    rotation: str = "identity",
    concatenate: bool = True,
    zeroing: bool = True,
    clipping: bool = True,
    weighted: bool = True) -> tff.aggregators.AggregationFactory:
  """Creates an aggregation factory for quantization and entropy coding.

  Args:
    step_size: A float that determines the step size between adjacent
      quantization levels to be used as the initial scale factor.
    rounding_type: A string that determines what type of rounding to apply,
      one of ["uniform", "stochastic", "dithered"].
    normalization_type: A string that determines what normalization to apply to
      the client updates, one of ["constant", "mean_magnitude", "max_magnitude",
      "dimensionless_norm"].
    step_size_sched: A string the determines what function should be used to
      adjust the `step_size`, one of ["fixed", "linear_decay",
      "exponential_decay", "step_decay"].
    step_size_sched_hparam: A float hyperparameter for the selected
      `step_size_sched` function (0. for "fixed", total_rounds for
      "linear_decay", exp for "exponential_decay", freq for "step_decay").
    min_step_size: A float specifying the minimum value of `step_size` to allow
      when decaying it according to `step_size_sched`.
    rotation: A string to specify what rotation (if any) to apply, one of
      ["dft", "hadamard" or "identity"].
    concatenate: A boolean indicating whether to concatenate all tensors in
      client update to a single tensor (`True`) or not (`False`) within the
      aggregation process.
    zeroing: A boolean indicating whether to add zeroing out extreme client
      updates (`True`) or not (`False`).
    clipping: A boolean indicating whether to add clipping to large client
      updates (`True`) or not (`False`).
    weighted: A boolean indicating whether client model weights should be
      averaged in a weighted manner (`True`) or unweighted manner (`False`).

  Returns:
    A `tff.aggregators.AggregationFactory`.
  """
  if rounding_type not in ["uniform", "stochastic", "dithered"]:
    raise ValueError("Expected `rounding_type` to be one one of [\"uniform\", "
                     f"\"stochastic\", \"dithered\"], found {rounding_type}.")

  if normalization_type not in [
      "constant", "mean_magnitude", "max_magnitude", "dimensionless_norm"
  ]:
    raise ValueError(
        "Expected `normalization_type` to be one one of [\"constant\", "
        "\"mean_magnitude\", \"max_magnitude\", \"dimensionless_norm\"], found "
        f"{normalization_type}.")

  if step_size_sched not in [
      "fixed", "linear_decay", "exponential_decay", "step_decay"
  ]:
    raise ValueError(
        "Expected `step_size_sched` to be one one of [\"fixed\", "
        "\"linear_decay\", \"exponential_decay\", \"step_decay\"], found "
        f"{step_size_sched}.")

  factory = quantize_encode.QuantizeEncodeFactory(step_size, rounding_type,
                                                  normalization_type,
                                                  step_size_sched,
                                                  step_size_sched_hparam,
                                                  min_step_size)

  return configure_aggregator(factory, rotation, concatenate, zeroing, clipping,
                              weighted)


def build_vote_step_size_aggregator(
    step_size: float,
    rounding_type: str = "uniform",
    sampling_width: float = 1.15,
    rotation: str = "identity",
    concatenate: bool = True,
    zeroing: bool = True,
    clipping: bool = True,
    weighted: bool = True) -> tff.aggregators.AggregationFactory:
  r"""Creates an aggregation factory for client voting experiments.

  Args:
    step_size: A float that determines the step size between adjacent
      quantization levels to be used as the initial scale factor, one of
      [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 2.5, 3.75, 5.0, 7.5, 10.0].
    rounding_type: A string that determines what type of rounding to apply, one
      of ["uniform", "stochastic", "dithered"].
    sampling_width: A float that determines the width of the distribution of
      step_size_options to sweep over, by default 1.15.
    rotation: A string to specify what rotation (if any) to apply, one of
      ["dft", "hadamard" or "identity"].
    concatenate: A boolean indicating whether to concatenate all tensors in
      client update to a single tensor (`True`) or not (`False`) within the
      aggregation process.
    zeroing: A boolean indicating whether to add zeroing out extreme client
      updates (`True`) or not (`False`).
    clipping: A boolean indicating whether to add clipping to large client
      updates (`True`) or not (`False`).
    weighted: A boolean indicating whether client model weights should be
      averaged in a weighted manner (`True`) or unweighted manner (`False`).

  Returns:
    A `tff.aggregators.AggregationFactory`.
  """
  if step_size not in builder_configs.LAGRANGE_MULTIPLIER_VALUES.keys():
    raise ValueError(
        "Expected `step_size` to be one one of [0.05, 0.1, 0.25, 0.5, 1.0, "
        f"2.0, 2.5, 3.75, 5.0, 7.5, 10.0], found {step_size}.")
  if rounding_type not in ["uniform", "stochastic", "dithered"]:
    raise ValueError("Expected `rounding_type` to be one one of [\"uniform\", "
                     f"\"stochastic\", \"dithered\"], found {rounding_type}.")

  lagrange_multiplier = builder_configs.LAGRANGE_MULTIPLIER_VALUES[
      step_size]
  step_size_options = [
      step_size * scale for scale in sampling_width**np.linspace(-3, 3, 7)
  ]
  factory = quantize_encode_client_lambda.QuantizeEncodeClientLambdaFactory(
      lagrange_multiplier, step_size, step_size_options, rounding_type)

  return configure_aggregator(factory, rotation, concatenate, zeroing, clipping,
                              weighted)
