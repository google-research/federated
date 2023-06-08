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
"""Wires together matrix factorization query and TFF DPAggregator."""

from collections.abc import Callable
from typing import Any, Optional

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import tensorflow_privacy as tfp

from multi_epoch_dp_matrix_factorization import matrix_constructors
from multi_epoch_dp_matrix_factorization import matrix_factorization_query


def _make_residual_matrix(w_matrix: tf.Tensor) -> tf.Tensor:
  """Creates a one-index residual matrix.

  That is, for any vector v, the constructed matrix X satisfies:

  Xv[i] = Wv[i] - Wv[i-1]

  for an input W, where Wv[-1] is interpreted as 0.

  Args:
    w_matrix: Matrix for which to compute residual matrix, as described above.

  Returns:
    A residual matrix.
  """
  num_rows_w = w_matrix.shape[0]
  np_eye = np.eye(num_rows_w - 1)
  # Add a row and column of zeros, on the top and right of the lower-dimensional
  # identity.
  offset_matrix = np.pad(np_eye, pad_width=[(1, 0), (0, 1)])
  tf_offset = tf.constant(offset_matrix, dtype=w_matrix.dtype)
  offset_w = tf_offset @ w_matrix
  return w_matrix - offset_w


def _create_residual_linear_query_dp_factory(
    *,
    tensor_specs: matrix_factorization_query.NestedTensorSpec,
    l2_norm_clip: float,
    noise_multiplier: float,
    w_matrix: tf.Tensor,
    h_matrix: tf.Tensor,
    residual_clear_query_fn: Callable[
        [matrix_factorization_query.NestedTensorSpec],
        matrix_factorization_query.OnlineQuery,
    ],
    clients_per_round: int,
    seed: Optional[int],
    emit_zeros_after_last_round: bool = False,
) -> tff.aggregators.UnweightedAggregationFactory:
  """Implements on-the-fly noise generation for a factorized linear query.

  This function represents the integration of the mechanisms presented in
  "Private Online Prefix Sums via Optimal Matrix Factorizations",
  https://arxiv.org/abs/2202.08312, with TFF Aggregators.

  It produces a query that computes the _residuals_ of an underlying linear
  query S.  That is, we have an original linear query `S` which is factorized
  as `S = W H`. Then, this query produces a `UnweightedAggregationFactory` that
  on round `t` provides a DP estimate of  `Sx_t - Sx_{t-1}`.

  The `residual_clear_query_fn` must be a function that computes
  `Sx_t - Sx_{t-1}`, likely in a more efficient manner than matrix
  multiplication. For example, if the query `S` is the prefix-sum query (lower
  triangular matrix of 1s), this `residual_clear_query_fn` should return a
  `matrix_factorization_query.OnlineQuery` computing the identity (which is the
  term-by-term residual of prefix-sum).

  The function to which we apply isotropic Gaussian noise in this factorization
  is *guaranteed to have sensitivity exactly `l2_norm_clip` for single-pass
  algorithms*; internally, this is accomplished by normalizing the provided
  factorization.

  Notably, this guarantee is not similarly made by tree-based aggregation
  mechanisms, which have sensitivity depending (logithmically) on the number
  of elements aggregated.

  Args:
    tensor_specs: Nested tensor specs specifying the structure to which the
      constructed mechanism will be applied.
    l2_norm_clip: Global l2 norm to which to clip client contributions in the
      constructed aggregator factory.
    noise_multiplier: Configures Gaussian noise with `stddev = l2_norm_clip *
      noise_multiplier`; see comments on sensitivity above.
    w_matrix: The W term of a matrix factorization S = WH to use.
    h_matrix: The H term of a matrix factorization S = WH to use.
    residual_clear_query_fn: Callable which accepts nested tensor specs and
      returns an instance of `matrix_factorization_query.OnlineQuery`. As noted
      above, this online query should represent the 'residuals' of the matrix S
      (IE, its t^th element should be Sx_t - Sx_{t-1}), for integration with TFF
      aggregators.
    clients_per_round: The number of clients per round to be used with this
      mechanism. Used to normalize the resulting values.
    seed: Optional seed which will guarantee deterministic noise generation.
    emit_zeros_after_last_round: If true, if training continues to run after the
      number of rounds supported by the mechanism, the mechanism will simply
      release zeros. If false, an error is raised.

  Returns:
    An instance of `tff.aggregators.UnweightedAggregationFactory` which
    implements residual of prefix-sum computations with the streaming matrix
    factorization mechanism.
  """
  if h_matrix.shape[0] != w_matrix.shape[1]:
    raise ValueError(
        f'Unexpected shapes W {w_matrix.shape}, H {h_matrix.shape}'
    )
  del h_matrix

  # To integrate with tff.learning, we must compute the residuals of our linear
  # query, which requires computing the residuals of w.
  normalized_residual_w = _make_residual_matrix(w_matrix)

  def make_noise_mech(tensor_specs, stddev):
    return matrix_factorization_query.OnTheFlyFactorizedNoiseMechanism(
        tensor_specs=tensor_specs,
        stddev=stddev,
        w_matrix=normalized_residual_w,
        seed=seed,
    )

  if emit_zeros_after_last_round:
    num_rounds_before_zeros = w_matrix.shape[1]
  else:
    num_rounds_before_zeros = matrix_factorization_query.ERROR_ON_EXTRA_ROUNDS
  sum_query = matrix_factorization_query.FactorizedGaussianSumQuery(
      l2_norm_clip=l2_norm_clip,
      stddev=noise_multiplier * l2_norm_clip,
      tensor_specs=tensor_specs,
      clear_query_fn=residual_clear_query_fn,
      factorized_noise_fn=make_noise_mech,
      num_rounds_before_zeros=num_rounds_before_zeros,
  )

  mean_query = tfp.NormalizedQuery(sum_query, denominator=clients_per_round)
  dp_query = tff.aggregators.DifferentiallyPrivateFactory(mean_query)
  measured_query = tff.learning.add_debug_measurements(dp_query)
  return measured_query


def create_residual_prefix_sum_dp_factory(
    *,
    tensor_specs: matrix_factorization_query.NestedTensorSpec,
    l2_norm_clip: float,
    noise_multiplier: float,
    w_matrix: tf.Tensor,
    h_matrix: tf.Tensor,
    clients_per_round: int,
    seed: Optional[int],
) -> tff.aggregators.UnweightedAggregationFactory:
  """Implements on-the-fly noise generation for the prefix-sum query.

  W and H are assumed to represent a so-called 'streaming factorization' of
  the prefix-sum matrix S, as discussed in https://arxiv.org/abs/2202.08312.

  Args:
    tensor_specs: Nested tensor specs specifying the structure to which the
      constructed mechanism will be applied.
    l2_norm_clip: Global l2 norm to which to clip client contributions in the
      constructed aggregator factory.
    noise_multiplier: Configures Gaussian noise with `stddev = l2_norm_clip *
      noise_multiplier`; see comments on sensitivity above.
    w_matrix: The W term of a matrix factorization S = WH to use.
    h_matrix: The H term of a matrix factorization S = WH to use.
    clients_per_round: The number of clients per round to be used with this
      mechanism. Used to normalize the resulting values.
    seed: Optional seed which will guarantee deterministic noise generation.

  Returns:
    An instance of `tff.aggregators.UnweightedAggregationFactory` which
    implements residual of prefix-sum computations with the streaming matrix
    factorization mechanism.
  """
  factorized_matrix = (w_matrix @ h_matrix).numpy()
  expected_matrix = np.tril(
      np.ones(shape=[w_matrix.shape[0]] * 2, dtype=factorized_matrix.dtype)
  )
  np.testing.assert_allclose(factorized_matrix, expected_matrix, atol=1e-8)
  return _create_residual_linear_query_dp_factory(
      tensor_specs=tensor_specs,
      l2_norm_clip=l2_norm_clip,
      noise_multiplier=noise_multiplier,
      w_matrix=w_matrix,
      h_matrix=h_matrix,
      clients_per_round=clients_per_round,
      seed=seed,
      residual_clear_query_fn=matrix_factorization_query.IdentityOnlineQuery,
  )


def create_residual_momentum_dp_factory(
    *,
    tensor_specs: matrix_factorization_query.NestedTensorSpec,
    l2_norm_clip: float,
    noise_multiplier: float,
    w_matrix: tf.Tensor,
    h_matrix: tf.Tensor,
    clients_per_round: int,
    seed: Optional[int],
    momentum_value: float,
    learning_rates: Any = None,
    emit_zeros_after_last_round: bool = False,
) -> tff.aggregators.UnweightedAggregationFactory:
  """Implements on-the-fly noise generation for the momentum partial sum query.

  W and H are assumed to represent a so-called 'streaming factorization' of
  the momentum matrix S.

  Args:
    tensor_specs: Nested tensor specs specifying the structure to which the
      constructed mechanism will be applied.
    l2_norm_clip: Global l2 norm to which to clip client contributions in the
      constructed aggregator factory.
    noise_multiplier: Multiplier to compute the standard deviation of noise to
      apply to clipped tensors, after transforming with (a normalized version
      of) the matrix `h_matrix`.
    w_matrix: The W term of a matrix factorization S = WH to use.
    h_matrix: The H term of a matrix factorization S = WH to use.
    clients_per_round: The number of clients per round to be used with this
      mechanism. Used to normalize the resulting values.
    seed: Optional seed which will guarantee deterministic noise generation.
    momentum_value: Value of the momentum parameter.
    learning_rates: A vector of learning rates, one per iteration/round.
    emit_zeros_after_last_round: If true, if training continues to run after the
      number of rounds supported by the mechanism, the mechanism will simply
      release zeros. If false, an error is raised.

  Returns:
    An instance of `tff.aggregators.UnweightedAggregationFactory` which
    implements residual of prefix-sum computations with the streaming matrix
    factorization mechanism.

  Raises: An assertion error if the provided factorization does not factorize
     the specified momentum matrix within an absolute tolerance of 1e-8.
  """

  factorized_matrix = (w_matrix @ h_matrix).numpy()
  momentum_matrix = matrix_constructors.momentum_sgd_matrix(
      num_iters=w_matrix.shape[0],
      momentum=momentum_value,
      learning_rates=learning_rates,
  )
  np.testing.assert_allclose(factorized_matrix, momentum_matrix, atol=1e-8)

  def _clear_query_fn(
      tensor_specs: matrix_factorization_query.NestedTensorSpec,
  ) -> matrix_factorization_query.OnlineQuery:
    return matrix_constructors.MomentumWithLearningRatesResidual(
        tensor_specs, momentum_value, learning_rates=learning_rates
    )

  return _create_residual_linear_query_dp_factory(
      tensor_specs=tensor_specs,
      l2_norm_clip=l2_norm_clip,
      noise_multiplier=noise_multiplier,
      w_matrix=w_matrix,
      h_matrix=h_matrix,
      clients_per_round=clients_per_round,
      seed=seed,
      residual_clear_query_fn=_clear_query_fn,
      emit_zeros_after_last_round=emit_zeros_after_last_round,
  )
