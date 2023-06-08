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
"""Implements matrix-mechanism based DPQuery for streaming linear queries.

The implemenation in this file is inspired by the paper:
  "Private Online Prefix Sums via Optimal Matrix Factorizations"
  Mcmahan, Rush, Thakurta
  https://arxiv.org/abs/2202.08312

This implementation defines two components, one for computing nonprivate linear
queries, and the other for adding the appropriate noise corresponding to a
precomputed factorization. The matrix mechanism we implement here can be
understood as implementing the map:

  x -> W(Hx + b)

where WH = S (S being a matrix representation of the linear query we wish to
privatize), and b is a sample from an isotropic Gaussian with some specified
variance.

For this implementation, we leverage the equation:

  W(Hx + b) = Sx + Wb

Similarly to the first equation noted on page 12 of the paper mentioned above,
we see that to compute the t^th element of our output vector, we need compute
the t^th element of the 'in-the-clear' query S, along with the t^th element of
the noise vector Wb (which will be some linear combination of samples from an
isotropic Gaussian). For the (adaptive) streaming setting, W will generally be
assumed to be lower-triangular, as existing proofs of privacy will carry through
under this assumption. The two components defined here (`OnlineQuery` and
`FactorizedGaussianNoiseMechanism`) correspond to methods for computing these
two terms (Sx and Wb, respectively). These interfaces are then stitched
together to implement a `tfp.DPQuery` (in particular, a
`tfp.SumAggregationQuery`) by `FactorizedGaussianSumQuery`, which represents
the computation above. It is the constructor of the
`FactorizedGaussianSumQuery` who is responsible for ensuring that the two
components match up.

In this implementation, we stitch these components together in a sum-aggregated
manner. Meaning, e.g. in federated learning, client updates would first be
clipped, then aggregated via summation, before the aggregated result is finally
passed as the `observation` argument to an `OnlineQuery`.
"""

import abc
from collections.abc import Callable, Iterable
import random
from typing import Any, Optional, Union

import attr
import dp_accounting
import tensorflow as tf
import tensorflow_privacy as tfp

NestedTensorStructure = Iterable[Union[tf.Tensor, 'NestedTensorStructure']]
NestedTensorSpec = Iterable[Union[tf.TensorSpec, 'NestedTensorSpec']]

# Sentinel value for num_rounds_before_zeros
ERROR_ON_EXTRA_ROUNDS = -1


class OnlineQuery(metaclass=abc.ABCMeta):
  """Interface for in-the-clear computation of linear queries."""

  @abc.abstractmethod
  def initial_state(self) -> NestedTensorStructure:
    """Returns initial state for computing query in-the-clear."""
    pass

  @abc.abstractmethod
  def compute_query(
      self, state: NestedTensorStructure, observation: Any
  ) -> tuple[NestedTensorStructure, NestedTensorStructure]:
    """Computes query in-the-clear.

    Args:
      state: The current state.
      observation: The observation for this round / iteration, e.g., the average
        gradient for machine learning.

    Returns:
      A tuple (result, state) containing the result of the query for
      the current round and the updated state.
    """
    pass


def _zeros_like_tensorspecs(
    tensor_specs: NestedTensorSpec) -> NestedTensorStructure:
  return tf.nest.map_structure(lambda x: tf.zeros(shape=x.shape, dtype=x.dtype),
                               tensor_specs)


class CumulativeSumQuery(OnlineQuery):
  """An OnlineQuery that computes cumulative sums of a stream of vectors."""

  def __init__(self, tensor_specs):
    self._tensor_specs = tensor_specs

  def initial_state(self) -> NestedTensorStructure:
    """Initializes cumulative sum state with zeros."""
    return _zeros_like_tensorspecs(self._tensor_specs)

  def compute_query(
      self, state: NestedTensorStructure, observation: NestedTensorStructure
  ) -> tuple[NestedTensorStructure, NestedTensorStructure]:
    """Computes and returns sum of `state` and `observation`.

    Notice that in this query the result and the updated state are identical;
    this will not be the case in general, though it happens to be in this
    implementation.

    Args:
      state: Partial sum of values observed so far.
      observation: New value to add to this partial sum. Assumed to be the same
        structure as `state`.

    Returns:
      A tuple with two elements, each representing the sum of `state` and
      `observation`.
    """
    partial_sum = tf.nest.map_structure(tf.add, state, observation)
    return partial_sum, partial_sum


class IdentityOnlineQuery(OnlineQuery):
  """Trivial implementation of OnlineQuery interface, computing identity."""

  def __init__(self, tensor_specs):
    del tensor_specs  # Unused
    return

  def initial_state(self):
    return []

  def compute_query(self, state, observation):
    return observation, state


class FactorizedGaussianNoiseMechanism(metaclass=abc.ABCMeta):
  """Interface for generating noise matching Wb for matrix factorization.

  Implementations of this class must respect functional semantics in their
  `compute_noise` methods: when called with the same arguments, the exact same
  noise value should be returned. It is therefore crucial that for fresh noise
  to be added, the returned updated state from `compute_noise` must be used.
  """

  @abc.abstractmethod
  def initialize(self) -> Any:
    """Returns state variable used to parameterize noise computation."""
    pass

  @abc.abstractmethod
  def compute_noise(self, state: Any) -> tuple[NestedTensorStructure, Any]:
    """Returns a noise sample and updated `state`."""
    pass


def _compute_highdim_gaussian_sample(
    tensor_specs, stddev, samples_needed
) -> list[NestedTensorStructure]:
  """Returns a sequence of Gaussian samples according to `tensor_specs`."""

  random_normal = tf.random_normal_initializer(stddev=stddev)

  def get_noise(tensor_spec: tf.TensorSpec):
    return tf.cast(random_normal(tensor_spec.shape), dtype=tensor_spec.dtype)

  samples = []
  for _ in range(samples_needed):
    noise_sample = tf.nest.map_structure(get_noise, tensor_specs)
    samples.append(noise_sample)
  return samples


def _assert_not_too_many_rounds(crnt_round: int, num_rounds_before_zeros: int):
  return tf.debugging.assert_less(
      crnt_round,
      num_rounds_before_zeros,
      message='Attempted to compute a result that this factorized '
      'mechanism does not support. This mechanism was '
      'initialized with a factorization with '
      f'{num_rounds_before_zeros} rows, and can '
      f'therefore only support {num_rounds_before_zeros} rounds ('
      'calls to get_noised_result()). '
      f'Attempted to access {crnt_round}th result.')


@tf.function
def _compute_min_and_max_nonzero_indices(
    row: tf.Tensor,
) -> tuple[tf.Tensor, tf.Tensor]:
  """Computes indices of first and last nonzero entries in `row`."""
  where_row_nonzero = tf.where(
      tf.not_equal(row, tf.constant(0, dtype=row.dtype)))
  if tf.equal(tf.size(where_row_nonzero), 0):
    # If we dont early-return in the case that all elements are zero, the
    # reductions below return garbage
    return tf.constant(0, dtype=tf.int64), tf.constant(0, dtype=tf.int64)
  min_nonzero_index = tf.math.reduce_min(where_row_nonzero)
  max_nonzero_index = tf.math.reduce_max(where_row_nonzero)
  return min_nonzero_index, max_nonzero_index


class OnTheFlyFactorizedNoiseMechanism(FactorizedGaussianNoiseMechanism):
  """A factorized noise mechanism which computes its noise on the fly.

  Every time this mechanism is asked for noise corresponding to a row of its
  `w_matrix`, it will regenerate all the necessary isotropic Gaussian noise
  and add this noise into an accumulator, with coefficients specified by the
  queried row of `w_matrix`, returning this accumulator. This implementation
  may be slower during the online portion of noise generation than
  `PrecomputeFactorizedNoiseMechanism`, since it needs to regenerate and add
  up all the associated noise. However, this implementation has constant memory
  requirements, effectively trading computation for memory.

  The `state` returned by this function corresponds to the row of the
  factorization being computed.
  """

  def __init__(self,
               tensor_specs: NestedTensorSpec,
               stddev: float,
               w_matrix: tf.Tensor,
               seed: Optional[int] = None):
    """Constructs a mechanism to compute the noise associated to `w_matrix`.

    Args:
      tensor_specs: A nested structure of `tf.TensorSpecs` specifying the
        structure of tensors for which to generate noise.
      stddev: The standard deviation of the isotropic Gaussian noise to
        generate.
      w_matrix: The matrix specifying the linear combinations of isotropic
        Gaussian noise to return on each request.
      seed: Optional seed to use for noise generation. If not `None`, the
        sequence of noise this mechanism generates will be determinstic.
    """
    if w_matrix.shape.ndims != 2:
      raise ValueError('Invalid w_matrix argument: w_matrix should be a rank-2 '
                       'tensor (in other words, a matrix).')
    self._tensor_specs = tensor_specs
    self._stddev = stddev
    if seed is None:
      self._initial_seed = random.getrandbits(32)
    else:
      if not isinstance(seed, int):
        raise TypeError('Seed argument must be either None, or an integer '
                        f'value. Found a value of type {type(int)}.')
      self._initial_seed = seed
    self._flat_tensor_specs = tf.nest.flatten(self._tensor_specs)
    self._w_matrix = w_matrix
    self._w_matrix_num_rows = self._w_matrix.shape[0]

  def initialize(self) -> int:
    """Returns 0, indicating index of next row of factorization to compute."""
    return 0

  @tf.function
  def _compute_noise_for_idx(self, index: int) -> NestedTensorStructure:
    """Generates noise for index `index`, according to class configuration."""

    # We must explicitly bake in self._w_matrix as a constant, since we are
    # attempting to effectively capture an eager tensor inside a graph context
    # if this function is traced with TFF serialization mechanisms. This call
    # forces TensorFlow to read the appropriate tensor into the graph from the
    # external environment.
    w_matrix_as_constant = tf.constant(self._w_matrix)

    def noise_for_tensorspec(tensor_spec: tf.TensorSpec, seed):
      # This pattern, and the coupled callsites below, ensure deterministic
      # noise generation for a single instance of this noise mechanism. This
      # fact, the functional relationship between the inputs and outputs of this
      # function, is crucial to the effectiveness of this mechanism.
      return tf.random.stateless_normal(
          shape=tensor_spec.shape,
          mean=0.0,
          stddev=self._stddev,
          seed=(self._initial_seed, seed),
          dtype=tensor_spec.dtype)

    accumulator = _zeros_like_tensorspecs(self._tensor_specs)
    num_flat_tensor_specs = len(self._flat_tensor_specs)

    min_nonzero_index, max_nonzero_index = _compute_min_and_max_nonzero_indices(
        w_matrix_as_constant[index])
    # Iterate over the nonzero columns of w_matrix at row index.
    for i in range(min_nonzero_index, max_nonzero_index + 1):
      flat_noise_sample = [
          noise_for_tensorspec(spec, num_flat_tensor_specs * i + index)
          for index, spec in enumerate(self._flat_tensor_specs)
      ]
      noise_sample = tf.nest.pack_sequence_as(self._tensor_specs,
                                              flat_noise_sample)
      # pylint: disable=cell-var-from-loop
      coefficient = w_matrix_as_constant[index][i]
      to_add = tf.nest.map_structure(
          lambda x: x * tf.cast(coefficient, x.dtype), noise_sample)
      # pylint: enable=cell-var-from-loop
      accumulator = tf.nest.map_structure(tf.add, accumulator, to_add)
    return accumulator

  @tf.function
  def compute_noise(self, state: int) -> tuple[NestedTensorStructure, int]:
    """Returns noise corresponding to the index specified by `state`.

    Args:
      state: An integer specifying the index of the factorization to use. That
        is, a `state` argument of `i` will return the noise corresponding to
        `Wz[i]`.

    Returns:
      A linear combination of samples from an isotropic Gaussian distribution,
      weighted according to the entries of the row specified by `state` in the
      matrix `w` passed in tothe initializer of this class.

    Raises:
      ValueError: If the `state` argument either specifies a row which is
      out-of-bounds for the matrix `w`.
    """
    with tf.control_dependencies(
        [_assert_not_too_many_rounds(state, self._w_matrix_num_rows)]):
      noise = self._compute_noise_for_idx(index=state)
    return noise, state + 1


def _compute_num_diagonal_elements(d_matrix: tf.Tensor) -> int:
  """Python function computing the number of diagonal elements in a k-diagonal matrix d.
  """
  where_nonzero = tf.where(
      tf.not_equal(d_matrix, tf.constant(0, dtype=d_matrix.dtype)))
  diagonal_differences = tf.map_fn(lambda x: tf.abs(x[0] - x[1]), where_nonzero)
  # We add 1 since the diff will be zero if we only have elements on the
  # diagonal.
  num_diag_elements = tf.math.reduce_max(diagonal_differences) + 1
  return num_diag_elements.numpy()


class _SpecAndIndex:
  """Opaque data class to hold index and TensorSpec.

  A custom class ensures that tf.nest will not attempt to traverse.
  """

  def __init__(self, index: int, spec: NestedTensorSpec):
    self.index = index
    self.spec = spec


class StructuredFactorizedNoiseMechanism(FactorizedGaussianNoiseMechanism):
  r"""A structured factorized noise mechanism which can be efficiently computed.

  This mechanism is designed to accept a structured matrix W, which can be
  represented by:

  W = (AB^T) \odot M + D

  where:
    * \odot represents Hadamard (pointwise) multiplication
    * A and B are matrices of shape [n, r], where r << n.
    * D contains nonzero entries on the diagonal, and below the diagonal to
      some depth d.
    * M is 1 wherever D is zero in the lower triangle.

  This factorization enables an implementation of noise generation with time and
  space complexity sublinear as follows: for a mechanism applied to tensors with
  m parameters, this mechanism requires O(rm + n**2) memory overall, and
  O((d+r)m) compute per noise-generation step.

  The algorithm in this mechanism can be found described in Section 4.2 of
  https://arxiv.org/pdf/2202.08312.pdf, and pseudocode for this algorithm can be
  found in Appendix D of the same paper.
  """

  def __init__(self,
               tensor_specs: NestedTensorSpec,
               stddev: float,
               *,
               a_matrix: tf.Tensor,
               b_matrix: tf.Tensor,
               d_matrix: tf.Tensor,
               seed: Optional[int] = None):
    """Constructs a mechanism to compute the noise associated to `w_matrix`.

    Args:
      tensor_specs: A nested structure of `tf.TensorSpecs` specifying the
        structure of tensors for which to generate noise.
      stddev: The standard deviation of the isotropic Gaussian noise to
        generate.
      a_matrix: An [n, r]-shaped matrix representing the A term of the
        factorization described above.
      b_matrix: An [n, r]-shaped matrix representing the B term of the
        factorization described above.
      d_matrix: A matrix representing the diagonal component D of the structured
        representation for W described above.
      seed: Optional integer specifying the seed used for noise generation. If
        None, seed will be set via `random.getrandbits`. If set, this class
        guarantees that the noise it generates will be deterministic.
    """
    if len(a_matrix.shape) != 2:
      raise ValueError('Invalid a_matrix argument: a_matrix should be a rank-2 '
                       'tensor (in other words, a matrix).')
    if len(b_matrix.shape) != 2:
      raise ValueError('Invalid b_matrix argument: b_matrix should be a rank-2 '
                       'tensor (in other words, a matrix).')
    if len(d_matrix.shape) != 2:
      raise ValueError('Invalid d_matrix argument: d_matrix should be a rank-2 '
                       'tensor (in other words, a matrix).')
    n = a_matrix.shape[0]
    r = a_matrix.shape[1]
    self._matrix_num_rows = n
    if b_matrix.shape != a_matrix.shape:
      raise ValueError('a_matrix and b_matrix must be the same shape (n by r '
                       'matrices); encountered a_matrix of shape '
                       '{a_matrix.shape}, b_matrix of shape {b_matrix.shape}')
    if d_matrix.shape.as_list() != [n, n]:
      raise ValueError('d_matrix must be square, with number of rows and '
                       'columns identical to the number of rows of a_matrix '
                       f'and b_matrix. Found d_matrix of shape {d_matrix.shape}'
                       ', while the number of rows of a_matrix and b_matrix '
                       f'is {n}.')
    self._tensor_specs = tensor_specs
    self._stddev = stddev
    if seed is None:
      self._initial_seed = random.getrandbits(32)
    else:
      self._initial_seed = seed
    flat_tensor_specs = tf.nest.flatten(self._tensor_specs)
    specs_and_indices = []
    for idx, spec in enumerate(flat_tensor_specs):
      specs_and_indices.append(_SpecAndIndex(index=idx, spec=spec))
    self._flat_specs_and_indices = tuple(specs_and_indices)
    self._a_matrix = a_matrix
    self._b_matrix = b_matrix
    self._d_matrix = d_matrix
    self._factorization_rank = r
    self._num_diagonal_elements = _compute_num_diagonal_elements(self._d_matrix)

  def initialize(self) -> tuple[int, list[NestedTensorStructure]]:
    """Returns initial index and buffer state."""
    return 0, [
        _zeros_like_tensorspecs(self._tensor_specs)
        for _ in range(self._factorization_rank)
    ]

  @tf.function
  def _isotropic_noise_for_index(self, *,
                                 noise_index: int) -> NestedTensorStructure:
    """Computes deterministic isotropic noise sample for given index."""

    n_tensors = len(self._flat_specs_and_indices)

    def noise_for_tensorspec(tensor_spec: tf.TensorSpec, seed):
      # This pattern, and the coupled callsites below, ensure deterministic
      # noise generation for a single instance of this noise mechanism. This
      # fact, the functional relationship between the inputs and outputs of this
      # function, is crucial to the effectiveness of this mechanism.
      return tf.random.stateless_normal(
          shape=tensor_spec.shape,
          mean=0.0,
          stddev=self._stddev,
          seed=(self._initial_seed, seed),
          dtype=tensor_spec.dtype)

    def _noise_for_spec_and_index(x):
      return noise_for_tensorspec(x.spec, n_tensors * noise_index + x.index)

    flat_noise_sample = tf.nest.map_structure(_noise_for_spec_and_index,
                                              self._flat_specs_and_indices)

    return tf.nest.pack_sequence_as(self._tensor_specs, flat_noise_sample)

  @tf.function
  def _compute_diagonal_noise_for_index(self,
                                        index: int) -> NestedTensorStructure:
    """Computes diagonal portion of structured-factorization noise mechanism."""
    # Corresponds to line 12 in Algorithm 1.
    accumulator = _zeros_like_tensorspecs(self._tensor_specs)

    min_nonzero_index, max_nonzero_index = _compute_min_and_max_nonzero_indices(
        self._d_matrix[index])
    # Iterate over the nonzero columns of d_matrix at row index. Notice that
    # since d is assumed to be k-diagonal, the iteration over i takes at most k
    # steps.
    for i in range(min_nonzero_index, max_nonzero_index + 1):
      noise_sample = self._isotropic_noise_for_index(noise_index=i)
      coefficient = self._d_matrix[index][i]
      # pylint: disable=cell-var-from-loop
      to_add = tf.nest.map_structure(
          lambda x: x * tf.cast(coefficient, x.dtype), noise_sample)
      # pylint: enable=cell-var-from-loop
      accumulator = tf.nest.map_structure(tf.add, accumulator, to_add)
    return accumulator

  @tf.function
  def _compute_low_rank_noise(
      self, index: int, beta: list[NestedTensorStructure]
  ) -> tuple[NestedTensorStructure, list[NestedTensorStructure]]:
    """Computes low-rank portion of the noise.

    This function corresponds to lines 14-17 of Algorithm 1.

    Args:
      index: The index of W for which we wish to compute the low-rank portion of
        the noise. Corresponds to the variable i in Algorithm 1.
      beta: The accumulator used to store the state necessary to compute the
        noise for `index`. Corresponds to beta in Algorithm 1.

    Returns:
      A two-tuple containing:
        * A `NestedTensorStructure` representing the noise generated for
          `index`.
        * A list of `NestedTensorStructures` representing the updated beta
          state variable.
    """
    # The conditional on line 14. This inequality is strict here (as opposed to
    # the paper) because of the difference in indexing (the code is 0-indexed,
    # the paper is 1-indexed).
    if index < self._num_diagonal_elements:
      return _zeros_like_tensorspecs(self._tensor_specs), beta
    # Line 15.
    b_index = index - self._num_diagonal_elements
    # Initialize the rest of the loop variables.
    b_row = self._b_matrix[b_index, :]
    noise_sample = self._isotropic_noise_for_index(noise_index=b_index)
    a_row = self._a_matrix[index, :]
    low_rank_noise_accumulator = _zeros_like_tensorspecs(self._tensor_specs)

    col_idx = 0
    # This while loop couples lines 16 and 17 in the pseudocode for algorithm 1.
    new_beta = []
    while col_idx < self._factorization_rank:
      b_coefficient = b_row[col_idx]
      beta_entry = beta[col_idx]

      new_beta_entry = tf.nest.map_structure(
          tf.add, beta_entry,
          tf.nest.map_structure(lambda x: x * tf.cast(b_coefficient, x.dtype),
                                noise_sample))
      # Update beta; corresponds to line 16, inside-the-loop.
      new_beta.append(new_beta_entry)
      # Update the accumulator; corresponds to line 17 inside-the-loop.
      a_coefficient = a_row[col_idx]
      low_rank_noise_accumulator = tf.nest.map_structure(
          tf.add, low_rank_noise_accumulator,
          tf.nest.map_structure(lambda x: x * tf.cast(a_coefficient, x.dtype),
                                new_beta_entry))
      col_idx += 1
    return low_rank_noise_accumulator, new_beta

  @tf.function
  def compute_noise(
      self, state: tuple[int, list[NestedTensorStructure]]
  ) -> tuple[NestedTensorStructure, tuple[int, list[NestedTensorStructure]]]:
    """Returns noise corresponding to the index specified by `state`.

    Args:
      state: A tuple with two elements: * an integer specifying the index of the
        factorization to use. That is, a `state` argument of `i` will return the
        noise corresponding to `Wz[i]`. * A list of nested tensor structures,
        representing the buffer beta in algorithm 1 of
        https://arxiv.org/pdf/2202.08312.pdf.

    Returns:
      A tuple with two elements:
        * A linear combination of samples from an isotropic Gaussian
          distribution, weighted according to the factorization described above
          and Algorithm 1 in https://arxiv.org/pdf/2202.08312.pdf.
        * An updated state, containing the next row to query and an updated
          buffer beta.

    Raises:
      ValueError: If the `state` argument either specifies a row which is
      out-of-bounds for the matrix `W`.
    """
    with tf.control_dependencies(
        [_assert_not_too_many_rounds(state[0], self._matrix_num_rows)]):
      index, beta = state
      diagonal_noise = self._compute_diagonal_noise_for_index(index)
      low_rank_noise, updated_beta = self._compute_low_rank_noise(index, beta)
      total_noise = tf.nest.map_structure(tf.add, diagonal_noise,
                                          low_rank_noise)
      return total_noise, (index + 1, updated_beta)


class PrecomputeFactorizedNoiseMechanism(FactorizedGaussianNoiseMechanism):
  """Implements the simplest possible factorized noise mechanism.

  In particular, this mechanism precomputes the necessary samples from a
  high-dimensional Gaussian, and stores them as an instance variable. The
  appropriate linear combinations are then computed on each invocation of the
  noise generation function. This mechanism requires sufficient memory to store
  one Gaussian sample structured according to `tensor_specs` for each row of
  `w_matrix`.

  The `state` returned by this function corresponds to the row of the
  factorization being computed.
  """

  def __init__(self, tensor_specs: NestedTensorSpec, stddev: float,
               w_matrix: tf.Tensor):
    """Constructs a mechanism to compute the noise associated to `w_matrix`.

    This mechanism represents a maximally simple, semantically correct
    implementation of anisotropic Gaussian noise generated according to
    `w_matrix` (IE, yielding in turn the elements of the tensor
    `w_matrix @ z`, where `z` is a vector of tensors sampled from a Gaussian
    with structure specified by `tensor_specs`).

    Args:
      tensor_specs: A nested structure of `tf.TensorSpecs` specifying the
        structure of tensors for which to generate noise.
      stddev: The standard deviation of the isotropic Gaussian noise to
        generate.
      w_matrix: The matrix specifying the linear combinations of isotropic
        Gaussian noise to return on each request.
    """
    if len(w_matrix.shape) != 2:
      raise ValueError('Invalid w_matrix argument: w_matrix should be a rank-2 '
                       'tensor (in other words, a matrix).')
    self._tensor_specs = tensor_specs
    self._stddev = stddev
    # We need as many samples as columns of W.
    num_samples_necessary = w_matrix.shape[1]
    self._gaussian_samples = _compute_highdim_gaussian_sample(
        self._tensor_specs, self._stddev, num_samples_necessary)
    self._w_matrix = w_matrix
    self._w_matrix_num_rows = self._w_matrix.shape[0]

  def initialize(self) -> int:
    """Returns 0, indicating index of next row of factorization to compute."""
    return 0

  @tf.function
  def _get_precomputed_noise_for_row(self,
                                     row_idx: int) -> NestedTensorStructure:
    """Reads precomputed noise, accumulates and returns."""
    # Grab the row corresponding to the desired observation.
    w_row = self._w_matrix[row_idx, :]
    accumulator = _zeros_like_tensorspecs(self._tensor_specs)
    idx = 0
    # Iterate over the columns of w_matrix at row index.
    while idx < w_row.shape[0]:
      coefficient = w_row[idx]
      to_add = tf.nest.map_structure(
          lambda x: x * tf.cast(coefficient, x.dtype),  # pylint: disable=cell-var-from-loop
          self._gaussian_samples[idx])
      accumulator = tf.nest.map_structure(tf.add, accumulator, to_add)
      idx += 1
    return accumulator

  @tf.function
  def compute_noise(self, state: int) -> tuple[NestedTensorStructure, int]:
    """Returns noise corresponding to the index specified by `state`.

    Args:
      state: An integer specifying the index of the factorization to use. That
        is, a `state` argument of `i` will return the noise corresponding to
        `Wz[i]`.

    Returns:
      A linear combination of samples from an isotropic Gaussian distribution,
      weighted according to the entries of the row specified by `state` in the
      matrix `w` passed in tothe initializer of this class.

    Raises:
      ValueError: If the `state` argument either specifies a row which is
      out-of-bounds for the matrix `w`.
    """
    with tf.control_dependencies(
        [_assert_not_too_many_rounds(state, self._w_matrix_num_rows)]):
      noise = self._get_precomputed_noise_for_row(row_idx=state)
    return noise, state + 1


@attr.s(frozen=True)
class FactorizedSumQueryState:
  """Data structure holding state for `FactorizedSumQuery`.

  Contains three elements:
    * noise_mech_state: The state corresponding to the subclass of
      `FactorizedGaussianNoiseMechanism` by which the `FactorizedSumQuery`
      is parameterized.
    * clear_query_state: The state corresponding to the `OnlineQuery` by which
      the `FactorizedSumQuery` is parameterized.
    * l2_norm_clip: The global l2 norm to which vectors coming into the
      `FactorizedSumQueryState` should be clipped.
  """

  noise_mech_state = attr.ib()
  clear_query_state = attr.ib()
  l2_norm_clip = attr.ib()
  current_round = attr.ib()


@tf.function
def _flatten_and_clip(record: NestedTensorStructure, clip_norm: float):
  flat_record = tf.nest.flatten(record)
  clipped, _ = tf.clip_by_global_norm(flat_record, clip_norm)
  return tf.nest.pack_sequence_as(record, clipped)


class FactorizedGaussianSumQuery(tfp.SumAggregationDPQuery):
  r"""A Gaussian-noise sum-aggregated DPQuery with matrix factorization.

  In particular, an instantiation of this query will compute a differentially
  private estimate of the linear mapping x -> Sx, where S represents the
  `OnlineQuery` this class takes as a parameter.

  The method of computing this privatized mapping depends on a preexisting
  matrix factorization. That is, we assume existence of a factorization S = W H.
  We assume this factorization to be square and lower-triangular, as discussed
  in the paper "Private Online Prefix Sums via Optimal Matrix Factorizations",
  https://arxiv.org/pdf/2202.08312.pdf.

  We compute the DP estimate by essentially transforming the incoming
  vector x to H-space (IE, computing the mapping x -> Hx), adding Gaussian noise
  in this space, then reconstructing our estimate of S via W. That is, running
  this mechanism will effectively compute (assuming appropriate boundedness for
  x):

  \hat{Sx} = W(Hx + b)

  where b represents a sample from a high-dimensional (isotropic) Gaussian.

  In the implementation itself, we leverage the capacity to distribute the W to
  compute instead:

  \hat{Sx} = Sx + Wb

  WARNING: To ensure that the appropriate noise is added for each index, this
  mechanism will have functional semantics for `get_noised_result`: meaning,
  when called twice with the same arguments, exactly the same result will
  appear. This property is also preserved under partial-evaluations--meaning,
  if we 'fix' the global state argument via partial evaluation, the resulting
  partial function will be functional with respect to its remaining arguments.
  It is therefore crucial for privacy guarantees that the updated global_state
  argument is passed when fresh noise is desired.
  """

  def __init__(self,
               l2_norm_clip: float,
               stddev: float,
               tensor_specs: NestedTensorSpec,
               clear_query_fn: Callable[[NestedTensorSpec], OnlineQuery],
               factorized_noise_fn: Callable[[NestedTensorSpec, float],
                                             FactorizedGaussianNoiseMechanism],
               num_rounds_before_zeros: int = ERROR_ON_EXTRA_ROUNDS):
    """Constructs a FactorizedGaussianSumQuery.

    Args:
      l2_norm_clip: Clip norm for per-client updates.
      stddev: Standard deviation of Gaussian noise to add to per-round result.
        Not scaled by l2_norm clip.
      tensor_specs: Specification for values to be aggregated.
      clear_query_fn: Function to compute the unnoised query.
      factorized_noise_fn: Noise generation function.
      num_rounds_before_zeros: If set to ERROR_ON_EXTRA_ROUNDS, a TF error is
        raised if results for more rounds than the mechanism supports are
        requested. If set to a non-negative integer, the mechanism simply
        returns zeros after this many rounds.
    """

    self._l2_norm_clip = l2_norm_clip
    self._tensor_specs = tensor_specs
    self._clear_query = clear_query_fn(self._tensor_specs)
    self._noise_mech = factorized_noise_fn(self._tensor_specs, stddev)

    if num_rounds_before_zeros < ERROR_ON_EXTRA_ROUNDS:
      raise ValueError(
          f'Unexpected `num_rounds_before_zeros`={num_rounds_before_zeros}.'
          'Specify a value >=0 or ERROR_ON_EXTRA_ROUNDS'
      )

    self._num_rounds_before_zeros = num_rounds_before_zeros

  def initial_global_state(self) -> FactorizedSumQueryState:
    return FactorizedSumQueryState(
        self._noise_mech.initialize(),
        self._clear_query.initial_state(),
        self._l2_norm_clip,
        current_round=0)

  def derive_sample_params(self, query_state: FactorizedSumQueryState) -> float:
    return query_state.l2_norm_clip

  def preprocess_record(self, params: float,
                        record: NestedTensorStructure) -> NestedTensorStructure:
    return _flatten_and_clip(record=record, clip_norm=params)

  @tf.function
  def get_noised_result(
      self,
      sample_state: NestedTensorStructure,
      global_state: FactorizedSumQueryState,
  ) -> tuple[
      NestedTensorStructure, FactorizedSumQueryState, dp_accounting.DpEvent
  ]:
    if (self._num_rounds_before_zeros >= 0 and
        global_state.current_round >= self._num_rounds_before_zeros):
      return (_zeros_like_tensorspecs(self._tensor_specs), global_state,
              dp_accounting.UnsupportedDpEvent())
    else:
      clear_record, clear_query_state = self._clear_query.compute_query(
          global_state.clear_query_state, sample_state)
      noise, noise_mech_state = self._noise_mech.compute_noise(
          global_state.noise_mech_state)
      noised_result = tf.nest.map_structure(tf.add, clear_record, noise)
      new_global_state = FactorizedSumQueryState(
          noise_mech_state=noise_mech_state,
          clear_query_state=clear_query_state,
          l2_norm_clip=global_state.l2_norm_clip,
          current_round=global_state.current_round + 1)
      # We follow the pattern of tree aggregation here, and return an
      # unsupported DP event. The responsibility for ensuring an apropriate
      # participation pattern is owned by the clients of this class.
      # TODO(b/230000870): Consider implementing an appropriate DpEvent,
      # and returning it once from this method.
      event = dp_accounting.UnsupportedDpEvent()
      return noised_result, new_global_state, event
