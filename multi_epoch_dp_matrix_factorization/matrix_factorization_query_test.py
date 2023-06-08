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
import random

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow_privacy.privacy.dp_query import gaussian_query
from tensorflow_privacy.privacy.dp_query import test_utils
from tensorflow_privacy.privacy.dp_query import tree_aggregation_query

from multi_epoch_dp_matrix_factorization import matrix_factorization_query

# Large enough for any of the tests. Could possibly be smaller.
MAX_MATRIX_SIZE = 1000


class IsotropicGaussian(
    matrix_factorization_query.FactorizedGaussianNoiseMechanism
):
  """Wraps usual Gaussian DP SumQuery in factorized mechanism interface."""

  def __init__(self, tensor_specs, stddev):
    self._tensor_specs = tensor_specs
    # Clipping is handled by the higher-level mechanism
    self._gaussian_query = gaussian_query.GaussianSumQuery(
        l2_norm_clip=float('inf'), stddev=stddev
    )

  def initialize(self):
    return self._gaussian_query.initial_global_state()

  def compute_noise(self, state):
    zeros = tf.nest.map_structure(
        lambda x: tf.zeros(shape=x.shape, dtype=x.dtype), self._tensor_specs
    )
    noise, state, _ = self._gaussian_query.get_noised_result(zeros, state)
    return noise, state


def _gaussian_as_factorized_query(
    tensor_specs, l2_norm_clip, stddev, factorized_noise_fn, **kwargs
):
  clear_query_fn = matrix_factorization_query.IdentityOnlineQuery
  return matrix_factorization_query.FactorizedGaussianSumQuery(
      l2_norm_clip,
      stddev,
      tensor_specs,
      clear_query_fn,
      factorized_noise_fn,
      **kwargs,
  )


def _tensor_spec_from_tensors(tensors):
  if isinstance(tensors, float):
    return tf.TensorSpec(dtype=tf.float32, shape=[])
  return tf.nest.map_structure(
      lambda x: tf.TensorSpec(shape=x.shape, dtype=x.dtype), tensors
  )


def inefficient_gaussian_precompute(x, y):
  return matrix_factorization_query.PrecomputeFactorizedNoiseMechanism(
      x, y, tf.eye(MAX_MATRIX_SIZE)
  )


def inefficient_gaussian_on_the_fly_compute(x, y):
  return matrix_factorization_query.OnTheFlyFactorizedNoiseMechanism(
      x, y, tf.eye(MAX_MATRIX_SIZE)
  )


def structured_factorized_gaussian_mechanism(x, y):
  # We initialize this to simply a factorization of the identity which does not
  # exercise the low-rank noise generation codepath.
  a_matrix = tf.zeros(shape=[MAX_MATRIX_SIZE, 1])
  b_matrix = tf.zeros(shape=[MAX_MATRIX_SIZE, 1])
  d_matrix = tf.eye(MAX_MATRIX_SIZE)
  return matrix_factorization_query.StructuredFactorizedNoiseMechanism(
      tensor_specs=x,
      stddev=y,
      a_matrix=a_matrix,
      b_matrix=b_matrix,
      d_matrix=d_matrix,
  )


@parameterized.named_parameters(
    ('isotropic_gaussian', IsotropicGaussian),
    (
        'factorized_isotropic_gaussian_precompute',
        inefficient_gaussian_precompute,
    ),
    (
        'structured_factorized_gaussian',
        structured_factorized_gaussian_mechanism,
    ),
    (
        'factorized_isotropic_gaussian_on_the_fly_compute',
        inefficient_gaussian_on_the_fly_compute,
    ),
)
class GaussianFactorizedInstantiationTest(
    tf.test.TestCase, parameterized.TestCase
):

  def test_gaussian_sum_no_clip_no_noise(self, factorized_noise_fn):
    with self.cached_session() as sess:
      record1 = tf.constant([2.0, 0.0])
      record2 = tf.constant([-1.0, 1.0])

      query = _gaussian_as_factorized_query(
          _tensor_spec_from_tensors(record1),
          l2_norm_clip=10.0,
          stddev=0.0,
          factorized_noise_fn=factorized_noise_fn,
      )
      query_result, _ = test_utils.run_query(query, [record1, record2])
      result = sess.run(query_result)
      expected = [1.0, 1.0]
      self.assertAllClose(result, expected)

  def test_gaussian_sum_with_clip_no_noise(self, factorized_noise_fn):
    with self.cached_session() as sess:
      record1 = tf.constant([-6.0, 8.0])  # Clipped to [-3.0, 4.0].
      record2 = tf.constant([4.0, -3.0])  # Not clipped.

      query = _gaussian_as_factorized_query(
          _tensor_spec_from_tensors(record1),
          l2_norm_clip=5.0,
          stddev=0.0,
          factorized_noise_fn=factorized_noise_fn,
      )
      query_result, _ = test_utils.run_query(query, [record1, record2])
      result = sess.run(query_result)
      expected = [1.0, 1.0]
      self.assertAllClose(result, expected)

  def test_gaussian_sum_with_noise(self, factorized_noise_fn):
    with self.cached_session() as sess:
      record1, record2 = tf.constant(2.71828), tf.constant(3.14159)
      stddev = 1.0

      query = _gaussian_as_factorized_query(
          _tensor_spec_from_tensors(record1),
          l2_norm_clip=5.0,
          stddev=stddev,
          factorized_noise_fn=factorized_noise_fn,
      )

      global_state = query.initial_global_state()

      noised_sums = []
      for _ in range(1000):
        # We move the run_query function inside the loop in TF2; if we pinned
        # this test to TF1 behavior (as is the case in the file from which this
        # test was forked), we could rely on the implementation of the Gaussian
        # noise mechanism to inject fresh randomness for the sess.run calls.
        query_result, global_state = test_utils.run_query(
            query, [record1, record2], global_state=global_state
        )
        noised_sums.append(sess.run(query_result))

      result_stddev = np.std(noised_sums)
      self.assertNear(result_stddev, stddev, 0.1)

  def test_gaussian_sum_merge(self, factorized_noise_fn):
    records1 = [tf.constant([2.0, 0.0]), tf.constant([-1.0, 1.0])]
    records2 = [tf.constant([3.0, 5.0]), tf.constant([-1.0, 4.0])]

    def get_sample_state(records):
      query = _gaussian_as_factorized_query(
          _tensor_spec_from_tensors(records1),
          l2_norm_clip=10.0,
          stddev=1.0,
          factorized_noise_fn=factorized_noise_fn,
      )
      global_state = query.initial_global_state()
      params = query.derive_sample_params(global_state)
      sample_state = query.initial_sample_state(records[0])
      for record in records:
        sample_state = query.accumulate_record(params, sample_state, record)
      return sample_state

    sample_state_1 = get_sample_state(records1)
    sample_state_2 = get_sample_state(records2)

    merged = gaussian_query.GaussianSumQuery(10.0, 1.0).merge_sample_states(
        sample_state_1, sample_state_2
    )

    with self.cached_session() as sess:
      result = sess.run(merged)

    expected = [3.0, 10.0]
    self.assertAllClose(result, expected)

  def test_emits_zeros_after_max_num_rounds(self, factorized_noise_fn):
    record = tf.constant(2.0)
    query = _gaussian_as_factorized_query(
        _tensor_spec_from_tensors(record),
        l2_norm_clip=1.0,
        stddev=1.0,
        factorized_noise_fn=factorized_noise_fn,
        num_rounds_before_zeros=1,
    )
    global_state = query.initial_global_state()
    params = query.derive_sample_params(global_state)

    # Round 1 - Should not return zeros
    sample_state = query.initial_sample_state(record)
    sample_state = query.accumulate_record(params, sample_state, record)
    result, global_state, _ = query.get_noised_result(
        sample_state, global_state
    )
    self.assertGreater(tf.abs(result), 0.0)

    # Round 2 and later should return zeros
    for _ in range(2):
      sample_state = query.initial_sample_state(record)
      sample_state = query.accumulate_record(params, sample_state, record)
      result, global_state, _ = query.get_noised_result(
          sample_state, global_state
      )
      self.assertEqual(result, 0.0)


@parameterized.named_parameters(
    (
        'precompute',
        matrix_factorization_query.PrecomputeFactorizedNoiseMechanism,
    ),
    (
        'on_the_fly_compute',
        matrix_factorization_query.OnTheFlyFactorizedNoiseMechanism,
    ),
)
class FullGeneralityNoiseMechanismTest(
    tf.test.TestCase, parameterized.TestCase
):

  def test_adds_zero_noise_with_zero_stddev(self, noise_mech_fn):
    small_w = np.eye(2)
    mechanism = noise_mech_fn(
        tensor_specs=tf.TensorSpec(dtype=tf.float32, shape=[]),
        stddev=0,
        w_matrix=tf.constant(small_w),
    )
    state = mechanism.initialize()
    sample, _ = mechanism.compute_noise(state)
    self.assertEqual(sample, 0)

  def test_raises_with_out_of_bounds_state(self, noise_mech_fn):
    dim = 2
    small_w = np.eye(dim)
    mechanism = noise_mech_fn(
        tensor_specs=[tf.TensorSpec(dtype=tf.float32, shape=[])],
        stddev=0,
        w_matrix=tf.constant(small_w),
    )
    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError, 'mechanism does not support.'
    ):
      # noise for index dim is undefined with the w above
      mechanism.compute_noise(dim)

  def test_functional_compute_noise_semantics(self, noise_mech_fn):
    small_w = np.eye(2)
    mechanism = noise_mech_fn(
        tensor_specs=[tf.TensorSpec(dtype=tf.float32, shape=[])],
        stddev=1,
        w_matrix=tf.constant(small_w),
    )
    first_noise = mechanism.compute_noise(0)
    first_noise_again = mechanism.compute_noise(0)
    self.assertEqual(first_noise, first_noise_again)

  def test_gaussian_mechanism_generates_two_identical_samples(
      self, noise_mech_fn
  ):
    tensor_specs = [tf.TensorSpec(dtype=tf.float32, shape=[])] * 2
    stddev = 1.0
    # This w_matrix specifies that the first and the second samples should be
    # identical.
    w_matrix = tf.constant([[1.0, 0.0], [1.0, 0.0]])
    noise_mech = noise_mech_fn(tensor_specs, stddev, w_matrix)
    state = noise_mech.initialize()
    first_sample, state = noise_mech.compute_noise(state)
    second_sample, state = noise_mech.compute_noise(state)
    self.assertEqual(first_sample, second_sample)

  def test_nondiagonal_matrix_generates_expected_variance(self, noise_mech_fn):
    num_samples = 1000
    tensor_specs = tf.TensorSpec(dtype=tf.float32, shape=[])
    stddev = 1.0

    # Constructs a nondiagonal matrix each of whose rows has l_2 norm 1.
    matrix = np.zeros(shape=[num_samples, num_samples])
    matrix[0][0] = 1.0
    for i in range(1, num_samples):
      matrix[i][i] = 1 / 2**0.5
      matrix[i][i - 1] = 1 / 2**0.5

    w_matrix = tf.constant(matrix)

    noise_mech = noise_mech_fn(tensor_specs, stddev, w_matrix)
    state = noise_mech.initialize()
    samples = []
    for _ in range(num_samples):
      sample, state = noise_mech.compute_noise(state)
      samples.append(sample)

    result_stddev = np.std(samples)
    self.assertNear(result_stddev, stddev, 0.1)

  def test_gaussian_mechanism_with_nested_structure_generates_different_noise(
      self, noise_mech_fn
  ):
    tensor_specs = [
        [tf.TensorSpec(dtype=tf.float32, shape=[])],
        tf.TensorSpec(dtype=tf.float32, shape=[]),
    ]
    stddev = 1.0
    # Should return a single sample from a structured Gaussian
    w_matrix = tf.constant([[1.0]])
    noise_mech = noise_mech_fn(tensor_specs, stddev, w_matrix)
    state = noise_mech.initialize()
    sample, state = noise_mech.compute_noise(state)
    # Walk the structure and assert the sample looks as expected.
    self.assertEqual(state, 1)
    self.assertLen(sample, 2)
    self.assertLen(sample[0], 1)
    self.assertIsInstance(sample[0][0], tf.Tensor)
    self.assertEqual(sample[0][0].dtype, tf.float32)
    self.assertEqual(sample[0][0].shape.as_list(), [])
    self.assertIsInstance(sample[1], tf.Tensor)
    self.assertEqual(sample[1].dtype, tf.float32)
    self.assertEqual(sample[1].shape.as_list(), [])
    # But that the random noise generator gave us fresh noise at each tensor.
    self.assertNotEqual(sample[0][0], sample[1])


def _generate_dim_and_rank_named_parameters():
  strings_and_params = []
  for n in range(2, 10):
    for r in range(1, n):
      strings_and_params.append((f'n_{n}_r_{r}', n, r))
  return tuple(strings_and_params)


class StructuredFactorizedNoiseMechanismTest(
    tf.test.TestCase, parameterized.TestCase
):

  def test_raises_mismatched_a_and_b(self):
    n = 2
    r = 1
    a_matrix = tf.ones(shape=[n, r])
    b_matrix = tf.ones(shape=[n, r + 1])
    d_matrix = tf.eye(n)
    with self.assertRaisesRegex(ValueError, 'must be the same shape'):
      matrix_factorization_query.StructuredFactorizedNoiseMechanism(
          tensor_specs=tf.TensorSpec(dtype=tf.float32, shape=[]),
          stddev=0,
          a_matrix=a_matrix,
          b_matrix=b_matrix,
          d_matrix=d_matrix,
      )

  def test_raises_non_matrix_a(self):
    n = 2
    r = 1
    a_matrix = tf.ones(shape=[n])
    b_matrix = tf.ones(shape=[n, r])
    d_matrix = tf.eye(3)
    with self.assertRaisesRegex(ValueError, 'a_matrix.*rank-2 tensor'):
      matrix_factorization_query.StructuredFactorizedNoiseMechanism(
          tensor_specs=tf.TensorSpec(dtype=tf.float32, shape=[]),
          stddev=0,
          a_matrix=a_matrix,
          b_matrix=b_matrix,
          d_matrix=d_matrix,
      )

  def test_raises_non_matrix_b(self):
    n = 2
    r = 1
    a_matrix = tf.ones(shape=[n, r])
    b_matrix = tf.ones(shape=[n])
    d_matrix = tf.eye(3)
    with self.assertRaisesRegex(ValueError, 'b_matrix.*rank-2 tensor'):
      matrix_factorization_query.StructuredFactorizedNoiseMechanism(
          tensor_specs=tf.TensorSpec(dtype=tf.float32, shape=[]),
          stddev=0,
          a_matrix=a_matrix,
          b_matrix=b_matrix,
          d_matrix=d_matrix,
      )

  def test_raises_non_matrix_d(self):
    n = 2
    r = 1
    a_matrix = tf.ones(shape=[n, r])
    b_matrix = tf.ones(shape=[n, r])
    d_matrix = tf.ones(shape=[n])
    with self.assertRaisesRegex(ValueError, 'd_matrix.*rank-2 tensor'):
      matrix_factorization_query.StructuredFactorizedNoiseMechanism(
          tensor_specs=tf.TensorSpec(dtype=tf.float32, shape=[]),
          stddev=0,
          a_matrix=a_matrix,
          b_matrix=b_matrix,
          d_matrix=d_matrix,
      )

  def test_raises_d_mismatched_to_a_and_b(self):
    n = 2
    r = 1
    a_matrix = tf.ones(shape=[n, r])
    b_matrix = tf.ones(shape=[n, r])
    d_matrix = tf.eye(3)
    with self.assertRaisesRegex(ValueError, 'Found d_matrix'):
      matrix_factorization_query.StructuredFactorizedNoiseMechanism(
          tensor_specs=tf.TensorSpec(dtype=tf.float32, shape=[]),
          stddev=0,
          a_matrix=a_matrix,
          b_matrix=b_matrix,
          d_matrix=d_matrix,
      )

  def test_raises_with_out_of_bounds_state(self):
    n = 2
    r = 1
    a_matrix = tf.ones(shape=[n, r])
    b_matrix = tf.ones(shape=[n, r])
    d_matrix = tf.eye(n)
    mechanism = matrix_factorization_query.StructuredFactorizedNoiseMechanism(
        tensor_specs=tf.TensorSpec(dtype=tf.float32, shape=[]),
        stddev=0,
        a_matrix=a_matrix,
        b_matrix=b_matrix,
        d_matrix=d_matrix,
    )
    _, beta = mechanism.initialize()
    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError, 'mechanism does not support.'
    ):
      mechanism.compute_noise((n, beta))

  def test_adds_zero_noise_with_zero_stddev(self):
    n = 2
    r = 1
    a_matrix = tf.ones(shape=[n, r])
    b_matrix = tf.ones(shape=[n, r])
    d_matrix = tf.eye(n)
    mechanism = matrix_factorization_query.StructuredFactorizedNoiseMechanism(
        tensor_specs=tf.TensorSpec(dtype=tf.float32, shape=[]),
        stddev=0,
        a_matrix=a_matrix,
        b_matrix=b_matrix,
        d_matrix=d_matrix,
    )
    state = mechanism.initialize()
    sample, _ = mechanism.compute_noise(state)
    self.assertEqual(sample, 0)

  def test_functional_compute_noise_semantics(self):
    n = 2
    r = 1
    a_matrix = tf.ones(shape=[n, r])
    b_matrix = tf.ones(shape=[n, r])
    d_matrix = tf.eye(n)
    mechanism = matrix_factorization_query.StructuredFactorizedNoiseMechanism(
        tensor_specs=tf.TensorSpec(dtype=tf.float32, shape=[]),
        stddev=1,
        a_matrix=a_matrix,
        b_matrix=b_matrix,
        d_matrix=d_matrix,
    )
    _, beta = mechanism.initialize()
    first_noise = mechanism.compute_noise((0, beta))
    first_noise_again = mechanism.compute_noise((0, beta))
    self.assertEqual(first_noise, first_noise_again)

  def test_mechanism_generates_two_identical_samples(self):
    n = 2
    r = 1
    # This factorization corresponds to [[1, 1], [1, 1]]
    a_matrix = tf.ones(shape=[n, r])
    b_matrix = tf.ones(shape=[n, r])
    np_matrix = np.eye(n)
    np_matrix[1][1] = 0
    # This d matrix is [[1, 0], [0, 0]]. Therefore the factorization we have
    # here corresponds to the matrix [[1, 0], [1, 0]], and should generate two
    # identical samples of noise.
    d_matrix = tf.constant(np_matrix)
    tensor_specs = [tf.TensorSpec(dtype=tf.float32, shape=[])] * 2
    stddev = 1.0
    noise_mech = matrix_factorization_query.StructuredFactorizedNoiseMechanism(
        tensor_specs=tensor_specs,
        stddev=stddev,
        a_matrix=a_matrix,
        b_matrix=b_matrix,
        d_matrix=d_matrix,
    )
    state = noise_mech.initialize()
    first_sample, state = noise_mech.compute_noise(state)
    second_sample, state = noise_mech.compute_noise(state)
    self.assertEqual(first_sample, second_sample)

  def test_gaussian_mechanism_with_nested_structure_generates_different_noise(
      self,
  ):
    tensor_specs = [
        [tf.TensorSpec(dtype=tf.float32, shape=[])],
        tf.TensorSpec(dtype=tf.float32, shape=[]),
    ]
    n = 1
    r = 1
    a_matrix = tf.zeros(shape=[n, n])
    b_matrix = tf.zeros(shape=[n, r])
    d_matrix = tf.eye(n)
    noise_mech = matrix_factorization_query.StructuredFactorizedNoiseMechanism(
        tensor_specs=tensor_specs,
        stddev=1,
        a_matrix=a_matrix,
        b_matrix=b_matrix,
        d_matrix=d_matrix,
    )

    state = noise_mech.initialize()
    sample, state = noise_mech.compute_noise(state)
    # Walk the structure and assert the sample looks as expected.
    self.assertLen(state, 2)
    self.assertEqual(state[0], 1)
    self.assertLen(sample, 2)
    self.assertLen(sample[0], 1)
    self.assertIsInstance(sample[0][0], tf.Tensor)
    self.assertEqual(sample[0][0].dtype, tf.float32)
    self.assertEqual(sample[0][0].shape.as_list(), [])
    self.assertIsInstance(sample[1], tf.Tensor)
    self.assertEqual(sample[1].dtype, tf.float32)
    self.assertEqual(sample[1].shape.as_list(), [])
    # But that the random noise generator gave us fresh noise at each tensor.
    self.assertNotEqual(sample[0][0], sample[1])

  @parameterized.named_parameters(*_generate_dim_and_rank_named_parameters())
  def test_structured_factorization_equivalent_to_direct_factorization(
      self, n, r
  ):
    # This factorization corresponds to the prefix-sum S matrix. We need to
    # normalize one term by r to ensure we get 1s in every entry.
    a_matrix = 1 / r * tf.ones(shape=[n, r])
    b_matrix = tf.ones(shape=[n, r])
    d_matrix = tf.eye(n)

    # Construct the prefix-sum matrix S directly.
    w_matrix = tf.constant(np.tril(np.ones(shape=[n, n])))
    tensor_specs = [tf.TensorSpec(dtype=tf.float32, shape=[])] * 2
    stddev = 1.0
    # Setting the seed equal in these two mechanisms ensures they generate the
    # same noise sequence. Technically this is not guaranteed by the interfaces,
    # but is an aspect of the implementation that helps testing.
    seed = random.getrandbits(32)
    structured_noise_mech = (
        matrix_factorization_query.StructuredFactorizedNoiseMechanism(
            tensor_specs=tensor_specs,
            stddev=stddev,
            a_matrix=a_matrix,
            b_matrix=b_matrix,
            d_matrix=d_matrix,
            seed=seed,
        )
    )

    direct_noise_mech = (
        matrix_factorization_query.OnTheFlyFactorizedNoiseMechanism(
            tensor_specs=tensor_specs,
            stddev=stddev,
            w_matrix=w_matrix,
            seed=seed,
        )
    )

    structured_state = structured_noise_mech.initialize()
    direct_noise_state = direct_noise_mech.initialize()

    row_idx = 0
    while row_idx < n:
      structured_sample, structured_state = structured_noise_mech.compute_noise(
          structured_state
      )
      direct_noise_sample, direct_noise_state = direct_noise_mech.compute_noise(
          direct_noise_state
      )
      # Were working with floats, and these are computed differently, so we
      # expect these to be a little distinct.
      self.assertAllClose(structured_sample, direct_noise_sample)
      row_idx += 1


# The matrix construction functions used for testing below are inlined from the
# streaming matrix-factorization repository.


def _double_leaves(tree_matrix: np.ndarray) -> np.ndarray:
  rows, cols = tree_matrix.shape
  return np.block([
      [tree_matrix, np.zeros(shape=(rows, cols))],
      [np.zeros(shape=(rows, cols)), tree_matrix],
      [np.ones(shape=(2 * cols))],
  ])


def _binary_tree_matrix(*, log_2_leaves: int) -> np.ndarray:
  m = np.array([[1.0]])
  for _ in range(log_2_leaves):
    m = _double_leaves(m)
  return m


def _make_full_honaker_factorized_query(
    log_2_leaves: int,
    l2_norm_clip: float,
    stddev: float,
    tensor_specs: matrix_factorization_query.NestedTensorSpec,
    precompute: bool,
) -> matrix_factorization_query.FactorizedGaussianSumQuery:
  clear_query_fn = matrix_factorization_query.CumulativeSumQuery

  s_dimensionality = 2**log_2_leaves
  s_matrix = np.tril(np.ones(shape=(s_dimensionality, s_dimensionality)))
  binary_tree_h = _binary_tree_matrix(log_2_leaves=log_2_leaves)
  full_honaker_w = s_matrix @ np.linalg.pinv(binary_tree_h)

  if precompute:

    def factorized_noise_fn(x, y):
      return matrix_factorization_query.PrecomputeFactorizedNoiseMechanism(
          x, y, w_matrix=tf.constant(full_honaker_w)
      )

  else:

    def factorized_noise_fn(x, y):
      return matrix_factorization_query.OnTheFlyFactorizedNoiseMechanism(
          x, y, w_matrix=tf.constant(full_honaker_w)
      )

  return matrix_factorization_query.FactorizedGaussianSumQuery(
      l2_norm_clip, stddev, tensor_specs, clear_query_fn, factorized_noise_fn
  )


# Inlined functions for testing compatibility with previous tree aggregation
# implementations follow.


def _get_noise_fn(specs, stddev, seed=1):
  random_generator = tf.random.Generator.from_seed(seed)

  def noise_fn():
    shape = tf.nest.map_structure(lambda spec: spec.shape, specs)
    return tf.nest.map_structure(
        lambda x: random_generator.normal(x, stddev=stddev), shape
    )

  return noise_fn


def _get_l2_clip_fn():
  def l2_clip_fn(record_as_list, clip_value):
    clipped_record, _ = tf.clip_by_global_norm(record_as_list, clip_value)
    return clipped_record

  return l2_clip_fn


class CumulativeSumOnlineQueryTest(tf.test.TestCase, parameterized.TestCase):

  def test_computes_sum_of_integers(self):
    tensor_specs = tf.TensorSpec(dtype=tf.int32, shape=[])
    prefix_sum_query = matrix_factorization_query.CumulativeSumQuery(
        tensor_specs=tensor_specs
    )
    state = prefix_sum_query.initial_state()
    for idx, observation in enumerate(range(100)):
      result, state = prefix_sum_query.compute_query(state, observation)
      self.assertEqual(result, int((idx * (idx + 1)) / 2))

  def test_raises_mismatched_structures(self):
    tensor_specs = tf.TensorSpec(dtype=tf.int32, shape=[])
    prefix_sum_query = matrix_factorization_query.CumulativeSumQuery(
        tensor_specs=tensor_specs
    )
    state = prefix_sum_query.initial_state()
    mismatched_tensor_specs = [tensor_specs, tensor_specs]

    with self.assertRaisesRegex(ValueError, 'same nested structure'):
      prefix_sum_query.compute_query(
          state,
          matrix_factorization_query._zeros_like_tensorspecs(
              mismatched_tensor_specs
          ),
      )

  def test_sums_structure_with_multiple_dtypes(self):
    tensor_specs = [
        tf.TensorSpec(dtype=tf.int32, shape=[]),
        tf.TensorSpec(dtype=tf.float32, shape=[]),
    ]
    prefix_sum_query = matrix_factorization_query.CumulativeSumQuery(
        tensor_specs=tensor_specs
    )
    state = prefix_sum_query.initial_state()
    for idx, raw_observation in enumerate(range(100)):
      tensor_observation = [
          tf.constant(raw_observation, tf.int32),
          tf.constant(raw_observation, tf.float32),
      ]
      result, state = prefix_sum_query.compute_query(state, tensor_observation)
      result_value = idx * (idx + 1) / 2
      expected_result = [
          tf.constant(int(result_value), dtype=tf.int32),
          tf.constant(float(result_value), dtype=tf.float32),
      ]
      self.assertEqual(result, expected_result)

  @parameterized.named_parameters(
      ('two_records_precompute', [2.71828, 3.14159], _get_noise_fn, True),
      (
          'two_records_on_the_fly_compute',
          [2.71828, 3.14159],
          _get_noise_fn,
          False,
      ),
      (
          'five_records_precompute',
          np.random.uniform(low=0.1, size=5).tolist(),
          _get_noise_fn,
          True,
      ),
      (
          'five_records_on_the_fly_compute',
          np.random.uniform(low=0.1, size=5).tolist(),
          _get_noise_fn,
          False,
      ),
  )
  def test_full_honaker_lower_variance_than_tree_agg(
      self, records, value_generator, precompute
  ):
    log_2_leaves = 7
    clip_value = 10.0
    stddev = 1.0
    num_trials, vector_size = 10, 2**log_2_leaves

    record_specs = tf.TensorSpec([vector_size])
    records = [tf.constant(r, shape=[vector_size]) for r in records]
    tree_noised_sums = []
    factorized_noised_sums = []
    for i in range(num_trials):
      tree_query = tree_aggregation_query.TreeCumulativeSumQuery(
          clip_fn=_get_l2_clip_fn(),
          clip_value=clip_value,
          noise_generator=value_generator(record_specs, seed=i, stddev=stddev),
          record_specs=record_specs,
          use_efficient=True,
      )

      factorized_query = _make_full_honaker_factorized_query(
          log_2_leaves=log_2_leaves,
          l2_norm_clip=clip_value,
          stddev=stddev,
          tensor_specs=record_specs,
          precompute=precompute,
      )
      tree_query_result, _ = test_utils.run_query(tree_query, records)
      factorized_query_result, _ = test_utils.run_query(
          factorized_query, records
      )

      tree_noised_sums.append(tree_query_result.numpy())
      factorized_noised_sums.append(factorized_query_result.numpy())
    tree_stddev = np.std(tree_noised_sums)
    factorized_stddev = np.std(factorized_noised_sums)

    self.assertNear(tree_stddev, stddev, 0.7)  # value for chi-squared test
    # factorized stddev is about 80% that of tree_stddev.
    self.assertLess(factorized_stddev, tree_stddev)


if __name__ == '__main__':
  tf.test.main()
