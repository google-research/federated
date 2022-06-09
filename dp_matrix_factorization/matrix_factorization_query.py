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
"""Implements matrix-mechanism based DPQuery for streaming linear queries.

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
import random
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

import attr
import tensorflow as tf
import tensorflow_privacy as tfp

from differential_privacy.python.accounting import dp_event

NestedTensorStructure = Iterable[Union[tf.Tensor, 'NestedTensorStructure']]
NestedTensorSpec = Iterable[Union[tf.TensorSpec, 'NestedTensorSpec']]


class OnlineQuery(metaclass=abc.ABCMeta):
  """Interface for in-the-clear computation of linear queries."""

  @abc.abstractmethod
  def initial_state(self) -> NestedTensorStructure:
    """Returns initial state for computing query in-the-clear."""
    pass

  @abc.abstractmethod
  def compute_query(
      self, state: NestedTensorStructure,
      observation: Any) -> Tuple[NestedTensorStructure, NestedTensorStructure]:
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
  ) -> Tuple[NestedTensorStructure, NestedTensorStructure]:
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
  def compute_noise(self, state: Any) -> Tuple[NestedTensorStructure, Any]:
    """Returns a noise sample and updated `state`."""
    pass


def _compute_highdim_gaussian_sample(
    tensor_specs, stddev, samples_needed) -> List[NestedTensorStructure]:
  """Returns a sequence of Gaussian samples according to `tensor_specs`."""

  random_normal = tf.random_normal_initializer(stddev=stddev)

  def get_noise(tensor_spec: tf.TensorSpec):
    return tf.cast(random_normal(tensor_spec.shape), dtype=tensor_spec.dtype)

  samples = []
  for _ in range(samples_needed):
    noise_sample = tf.nest.map_structure(get_noise, tensor_specs)
    samples.append(noise_sample)
  return samples


@tf.function
def _compute_min_and_max_nonzero_indices(
    row: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
  """Computes indices of first and last nonzero entries in `row`."""
  where_row_nonzero = tf.where(
      tf.not_equal(row, tf.constant(0, dtype=row.dtype)))
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
  def compute_noise(self, state: int) -> Tuple[NestedTensorStructure, int]:
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
    tf.debugging.assert_less(
        state,
        self._w_matrix_num_rows,
        message='Attempted to compute a sample that this factorized '
        'mechanism does not support. This mechanism was '
        'initialized with a factorization with '
        f'{self._w_matrix_num_rows} rows, and can '
        f'therefore only support {self._w_matrix_num_rows} samples.'
        f'Attempted to access {state}th sample.')
    noise = self._compute_noise_for_idx(index=state)
    return noise, state + 1


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
  def compute_noise(self, state: int) -> Tuple[NestedTensorStructure, int]:
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
    tf.debugging.assert_less(
        state,
        self._w_matrix_num_rows,
        message='Attempted to compute a sample that this factorized '
        'mechanism does not support. This mechanism was '
        'initialized with a factorization with '
        f'{self._w_matrix_num_rows} rows, and can '
        f'therefore only support {self._w_matrix_num_rows} samples.'
        f'Attempted to access {state}th sample.')
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
  matrix factorization. That is, we assume existence of a factorization S = WH.
  We assume this factorization to be square and lower-triangular.

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

  def __init__(self, l2_norm_clip: float, stddev: float,
               tensor_specs: NestedTensorSpec,
               clear_query_fn: Callable[[NestedTensorSpec], OnlineQuery],
               factorized_noise_fn: Callable[[NestedTensorSpec, float],
                                             FactorizedGaussianNoiseMechanism]):

    self._l2_norm_clip = l2_norm_clip
    self._tensor_specs = tensor_specs
    self._clear_query = clear_query_fn(self._tensor_specs)
    self._noise_mech = factorized_noise_fn(self._tensor_specs, stddev)

  def initial_global_state(self) -> FactorizedSumQueryState:
    return FactorizedSumQueryState(self._noise_mech.initialize(),
                                   self._clear_query.initial_state(),
                                   self._l2_norm_clip)

  def derive_sample_params(self, query_state: FactorizedSumQueryState) -> float:
    return query_state.l2_norm_clip

  def preprocess_record(self, params: float,
                        record: NestedTensorStructure) -> NestedTensorStructure:
    return _flatten_and_clip(record=record, clip_norm=params)

  def get_noised_result(
      self, sample_state: NestedTensorStructure,
      global_state: FactorizedSumQueryState
  ) -> Tuple[NestedTensorStructure, FactorizedSumQueryState, dp_event.DpEvent]:
    clear_record, clear_query_state = self._clear_query.compute_query(
        global_state.clear_query_state, sample_state)
    noise, noise_mech_state = self._noise_mech.compute_noise(
        global_state.noise_mech_state)
    noised_result = tf.nest.map_structure(tf.add, clear_record, noise)
    new_global_state = FactorizedSumQueryState(
        noise_mech_state=noise_mech_state,
        clear_query_state=clear_query_state,
        l2_norm_clip=global_state.l2_norm_clip)
    # We follow the pattern of tree aggregation here, and return an unsupported
    # DP event. The responsibility for ensuring a single-epoch call pattern is
    # owned by the clients of this class.
    # TODO(b/230000870): Consider implementing a SingleEpochFactorized event,
    # and returning it once from this method.
    event = dp_event.UnsupportedDpEvent()
    return noised_result, new_global_state, event
