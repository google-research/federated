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
"""Tests for aggregator_builder."""
from unittest import mock

from absl.testing import flagsaver
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from dp_matrix_factorization import matrix_constructors
from dp_matrix_factorization.dp_ftrl import aggregator_builder


def _simple_keras_model():
  keras_model = tf.keras.models.Sequential([
      tf.keras.layers.Input(shape=[10]),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(10)
  ])
  return keras_model


def _model_fn():
  return tff.learning.from_keras_model(
      _simple_keras_model(),
      loss=tf.keras.losses.MeanSquaredError(),
      input_spec=[
          tf.TensorSpec(shape=[10]),
          tf.TensorSpec(shape=[], dtype=tf.int64)
      ])


_DIMENSION = 5


def _build_aggregator_by_method(aggregator_method, momentum=0.0, **kwargs):
  return aggregator_builder.build_aggregator(
      aggregator_method=aggregator_method,
      model_fn=_model_fn,
      clip_norm=1.,
      noise_multiplier=1.,
      clients_per_round=1,
      num_rounds=_DIMENSION,
      noise_seed=0,
      momentum=momentum,
      **kwargs)


def _prefix_sum_matrix(dim: int, dtype: tf.DType) -> tf.Tensor:
  return tf.constant(np.tril(np.ones(shape=[dim, dim])), dtype=dtype)


def _momentum_matrix(dim: int, dtype: tf.DType, momentum=0.9) -> tf.Tensor:
  return tf.constant(
      matrix_constructors.momentum_sgd_matrix(dim, momentum), dtype)


# (W, H, optional_learning_rate_vector) tuples
# returned by _load_w_h_and_maybe_lr
_PREFIX_RVAL = (_prefix_sum_matrix(_DIMENSION,
                                   tf.float32), tf.eye(_DIMENSION), None)

_LR_MOMENTUM_0p90_RVAL = (_momentum_matrix(_DIMENSION, tf.float32, 0.9),
                          tf.eye(_DIMENSION), tf.ones(_DIMENSION))

_LR_MOMENTUM_0p00_RVAL = (_momentum_matrix(_DIMENSION, tf.float64, 0.0),
                          tf.eye(_DIMENSION, dtype=tf.float64), None)


class AggregatorBuilderTest(tf.test.TestCase):

  @mock.patch.object(
      aggregator_builder,
      '_load_w_h_and_maybe_lr',
      return_value=(tf.eye(_DIMENSION), tf.eye(_DIMENSION), None))
  @flagsaver.flagsaver(matrix_root_path='/root')
  def test_raises_with_non_factorization_of_prefix_sum(self, mock_fn):
    with self.assertRaises(AssertionError):
      _build_aggregator_by_method(aggregator_method='opt_prefix_sum_matrix')

    # The error is raised after this function has been called, since we need to
    # see the factorization to know it is incorrect.
    mock_fn.assert_called_once_with(f'/root/prefix_opt/size={_DIMENSION}')

  def test_raises_unknown_factorization(self):
    with self.assertRaisesRegex(NotImplementedError, 'not known.'):
      _build_aggregator_by_method(aggregator_method='unknown_method')

  def test_tree_aggregator_constructs(self):
    aggregator = _build_aggregator_by_method(
        aggregator_method='tree_aggregation')
    self.assertIsInstance(aggregator,
                          tff.aggregators.DifferentiallyPrivateFactory)

  @mock.patch.object(
      aggregator_builder, '_load_w_h_and_maybe_lr', return_value=_PREFIX_RVAL)
  @flagsaver.flagsaver(matrix_root_path='/root')
  def test_opt_factorization_constructs(self, mock_fn):
    aggregator = _build_aggregator_by_method(
        aggregator_method='opt_prefix_sum_matrix')
    self.assertIsInstance(aggregator,
                          tff.aggregators.DifferentiallyPrivateFactory)
    mock_fn.assert_called_once_with(f'/root/prefix_opt/size={_DIMENSION}')

  @mock.patch.object(
      aggregator_builder,
      '_load_w_h_and_maybe_lr',
      return_value=(_momentum_matrix(_DIMENSION, tf.float32,
                                     0.9), tf.eye(_DIMENSION), None))
  @flagsaver.flagsaver(matrix_root_path='/root')
  def test_opt_momentum_factorization_constructs(self, mock_fn):
    aggregator = _build_aggregator_by_method(
        aggregator_method='opt_momentum_matrix', momentum=0.9)
    self.assertIsInstance(aggregator,
                          tff.aggregators.DifferentiallyPrivateFactory)
    mock_fn.assert_called_once_with(f'/root/momentum_0p90/size={_DIMENSION}')

  @mock.patch.object(
      aggregator_builder, '_load_w_h_and_maybe_lr', return_value=_PREFIX_RVAL)
  @flagsaver.flagsaver(matrix_root_path='/root')
  def test_streaming_honaker_factorization_constructs(self, mock_fn):
    aggregator = _build_aggregator_by_method(
        aggregator_method='streaming_honaker_matrix')
    self.assertIsInstance(aggregator,
                          tff.aggregators.DifferentiallyPrivateFactory)
    mock_fn.assert_called_once_with(
        f'/root/prefix_online_honaker/size={_DIMENSION}')

  @mock.patch.object(
      aggregator_builder, '_load_w_h_and_maybe_lr', return_value=_PREFIX_RVAL)
  @flagsaver.flagsaver(matrix_root_path='/root')
  def test_full_honaker_factorization_constructs(self, mock_fn):
    aggregator = _build_aggregator_by_method(
        aggregator_method='full_honaker_matrix')
    self.assertIsInstance(aggregator,
                          tff.aggregators.DifferentiallyPrivateFactory)
    mock_fn.assert_called_once_with(
        f'/root/prefix_full_honaker/size={_DIMENSION}')

  @mock.patch.object(
      aggregator_builder,
      '_load_w_h_and_maybe_lr',
      return_value=_LR_MOMENTUM_0p00_RVAL)
  @flagsaver.flagsaver(matrix_root_path='/root')
  def test_momentum_matrix_constructs_0p00(self, mock_fn):
    aggregator = _build_aggregator_by_method(
        aggregator_method='opt_momentum_matrix', momentum=0.0)
    self.assertIsInstance(aggregator,
                          tff.aggregators.DifferentiallyPrivateFactory)
    mock_fn.assert_called_once_with(f'/root/momentum_0p00/size={_DIMENSION}')

  @mock.patch.object(
      aggregator_builder,
      '_load_w_h_and_maybe_lr',
      return_value=_LR_MOMENTUM_0p90_RVAL)
  @flagsaver.flagsaver(matrix_root_path='/root')
  def test_lr_momentum_matrix_constructs(self, mock_fn):
    aggregator = _build_aggregator_by_method(
        aggregator_method='lr_momentum_matrix',
        momentum=0.9,
        lr_momentum_matrix_name='foo_momentum_0p90')
    self.assertIsInstance(aggregator,
                          tff.aggregators.DifferentiallyPrivateFactory)
    mock_fn.assert_called_once_with(
        f'/root/foo_momentum_0p90/size={_DIMENSION}')

  @mock.patch.object(
      aggregator_builder,
      '_load_w_h_and_maybe_lr',
      return_value=_LR_MOMENTUM_0p90_RVAL)
  @flagsaver.flagsaver(matrix_root_path='/root')
  def test_lr_momentum_matrix_constructs_no_inferred_momentum(self, mock_fn):
    aggregator = _build_aggregator_by_method(
        aggregator_method='lr_momentum_matrix',
        momentum=0.9,
        lr_momentum_matrix_name='foo_const_lr_matrix')
    self.assertIsInstance(aggregator,
                          tff.aggregators.DifferentiallyPrivateFactory)
    mock_fn.assert_called_once_with(
        f'/root/foo_const_lr_matrix/size={_DIMENSION}')

  @flagsaver.flagsaver(matrix_root_path='/root')
  def test_raises_momentum_mismatch(self):
    with self.assertRaisesRegex(ValueError,
                                r'Mismatch between inferred momentum'):
      _build_aggregator_by_method(
          aggregator_method='lr_momentum_matrix',
          momentum=0.9,
          lr_momentum_matrix_name='foo_momentum_0p50')


if __name__ == '__main__':
  tf.test.main()
