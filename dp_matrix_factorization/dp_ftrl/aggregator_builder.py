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
"""Library for utilities to construct single-epoch aggregators for DPFTRL.

This library is written for Google-internal use, and may read from network
(e.g., owning CNS paths which contain matrix factorizations), and therefore
tests for this code must be written carefully to avoid such network access.
This file will, however, eventually be branched / migrated to OSS, and the paths
will be updated to point to publicly-available storage, once we write and
publish the paper to which this code is associated.
"""
import re
from typing import Callable, Optional, Tuple

from absl import flags
import tensorflow as tf
import tensorflow_federated as tff

from dp_matrix_factorization import tff_aggregator

_FACTORIZATION_ROOT = '/tmp/matrix_factorizations'
MATRIX_ROOT_PATH = flags.DEFINE_string('matrix_root_path', _FACTORIZATION_ROOT,
                                       'Root path for loading matrices.')

_W_MATRIX_STRING = 'w_matrix_tensor_pb'
_H_MATRIX_STRING = 'h_matrix_tensor_pb'
_LR_VECTOR_STRING = 'lr_vector_tensor_pb'

AGGREGATION_METHODS = frozenset({
    'tree_aggregation', 'opt_prefix_sum_matrix', 'streaming_honaker_matrix',
    'full_honaker_matrix', 'opt_momentum_matrix', 'lr_momentum_matrix'
})

# Matrix names
_PREFIX_ONLINE_HONAKER = 'prefix_online_honaker'
_PREFIX_FULL_HONAKER = 'prefix_full_honaker'
_PREFIX_OPT = 'prefix_opt'


def _join_path(*args):
  # Always use "/" for Google storage.
  return '/'.join(args)


def _get_matrix_path(n: int, mechanism_name: str) -> str:
  size_str = f'size={n:d}'
  return _join_path(MATRIX_ROOT_PATH.value, mechanism_name, size_str)


def _get_momentum_path(n: int, momentum: float) -> str:
  if not 0.0 <= momentum <= 1.0:
    raise ValueError(f'momentum {momentum} outside of range [0, 1]')
  if round(momentum, 2) != momentum:
    raise ValueError(f'Specify momentum in hundreths. Found {momentum}')
  return _get_matrix_path(
      n=n, mechanism_name=f'momentum_0p{100*momentum:02.0f}')


def _load_w_h_and_maybe_lr(
    path: str) -> Tuple[tf.Tensor, tf.Tensor, Optional[tf.Tensor]]:
  """Returns a tuple (w_matrix, h_matrix, learning_rate_vector)."""
  if not (tf.io.gfile.exists(path) and tf.io.gfile.isdir(path)):
    raise ValueError(f'Matrix factorization directory {path} does not exist. '
                     'Check flag values or ask for the files to be '
                     'generated.')
  w_matrix = tf.io.parse_tensor(
      tf.io.read_file(_join_path(path, _W_MATRIX_STRING)), tf.float64)
  h_matrix = tf.io.parse_tensor(
      tf.io.read_file(_join_path(path, _H_MATRIX_STRING)), tf.float64)
  lr_file = _join_path(path, _LR_VECTOR_STRING)
  lr_tensor = None
  if tf.io.gfile.exists(lr_file):
    lr_tensor = tf.io.parse_tensor(
        tf.io.read_file(_join_path(path, _LR_VECTOR_STRING)), tf.float64)
  return w_matrix, h_matrix, lr_tensor


def _get_prefix_sum_w_h(num_rounds: int,
                        aggregator_method: str) -> Tuple[tf.Tensor, tf.Tensor]:
  """Returns (W, H) for prefix sum methods."""
  if aggregator_method == 'opt_prefix_sum_matrix':
    path = _get_matrix_path(num_rounds, _PREFIX_OPT)
  elif aggregator_method == 'streaming_honaker_matrix':
    path = _get_matrix_path(num_rounds, _PREFIX_ONLINE_HONAKER)
  elif aggregator_method == 'full_honaker_matrix':
    path = _get_matrix_path(num_rounds, _PREFIX_FULL_HONAKER)
  else:
    raise NotImplementedError(
        f'Unexpected aggregator_method {aggregator_method}')
  w_matrix, h_matrix, lr_vector = _load_w_h_and_maybe_lr(path)
  assert lr_vector is None
  return w_matrix, h_matrix


def _infer_momentum_from_path(path: str) -> Optional[float]:
  match = re.search(r'momentum_0p(\d\d)', path)
  if match:
    return float(match.group(1)) / 100
  return None


def build_aggregator(
    *,
    aggregator_method: str,
    model_fn: Callable[[], tff.learning.Model],
    clip_norm: float,
    noise_multiplier: float,
    clients_per_round: int,
    num_rounds: int,
    noise_seed: Optional[int],
    momentum: float = 0.0,
    lr_momentum_matrix_name: Optional[str] = None
) -> tff.aggregators.DifferentiallyPrivateFactory:
  """Builds DP aggregators for integration with DPFTRLM tff.learning process."""

  if clip_norm <= 0:
    raise ValueError('`clip_norm` must be positive; '
                     f'got clip norm {clip_norm}.')
  if clients_per_round <= 0:
    raise ValueError('`clients_per_round` must be positive; '
                     f'got report goal {clients_per_round}.')
  if noise_multiplier < 0:
    raise ValueError('`noise_multiplier` must be nonnegative; '
                     f'got noise multiplier {noise_multiplier}.')
  if num_rounds <= 0:
    raise ValueError('`num_rounds` must be positive; '
                     f'got num rounds {num_rounds}.')
  if momentum < 0:
    raise ValueError('`momentum` must be nonnegative; '
                     f'got momentum {momentum}.')

  if (lr_momentum_matrix_name is not None and
      aggregator_method != 'lr_momentum_matrix'):
    raise ValueError('`lr_momentum_matrix_name` is only supported when'
                     'aggregator_method="lr_momentum_matrix"')

  model_weight_specs = tff.framework.type_to_tf_tensor_specs(
      tff.learning.framework.weights_type_from_model(model_fn).trainable)

  if aggregator_method not in AGGREGATION_METHODS:
    raise NotImplementedError(
        f'Aggregator method {aggregator_method} not known. Supported '
        'aggregation methods: \n' +
        ''.join([f'{x} \n' for x in AGGREGATION_METHODS]))

  if aggregator_method == 'tree_aggregation':
    return tff.aggregators.DifferentiallyPrivateFactory.tree_aggregation(
        noise_multiplier=noise_multiplier,
        clients_per_round=clients_per_round,
        l2_norm_clip=clip_norm,
        record_specs=model_weight_specs,
        noise_seed=noise_seed,
        use_efficient=True)
  elif aggregator_method in [  # Prefix sum methods
      'opt_prefix_sum_matrix',
      'streaming_honaker_matrix',
      'full_honaker_matrix',
  ]:
    w_matrix, h_matrix = _get_prefix_sum_w_h(num_rounds, aggregator_method)
    return tff_aggregator.create_residual_prefix_sum_dp_factory(
        tensor_specs=model_weight_specs,
        l2_norm_clip=clip_norm,
        noise_multiplier=noise_multiplier,
        w_matrix=w_matrix,
        h_matrix=h_matrix,
        clients_per_round=clients_per_round,
        seed=noise_seed)
  elif aggregator_method == 'opt_momentum_matrix':
    path = _get_momentum_path(num_rounds, momentum)
    w_matrix, h_matrix, lr_vector = _load_w_h_and_maybe_lr(path)
    assert lr_vector is None
    return tff_aggregator.create_residual_momentum_dp_factory(
        tensor_specs=model_weight_specs,
        l2_norm_clip=clip_norm,
        noise_multiplier=noise_multiplier,
        w_matrix=w_matrix,
        h_matrix=h_matrix,
        clients_per_round=clients_per_round,
        seed=noise_seed,
        momentum_value=momentum)
  elif aggregator_method == 'lr_momentum_matrix':
    if lr_momentum_matrix_name is None:
      raise ValueError('Must supply `lr_momentum_matrix_name` for the '
                       'lr_momentum_matrix method.')
    inferred_momentum = _infer_momentum_from_path(lr_momentum_matrix_name)

    if (inferred_momentum is None) or (momentum == inferred_momentum):
      # No inferred momentum, or they agree, so use the argument value
      pass
    elif inferred_momentum != momentum and momentum == 0.0:
      # If the argument is the default value of 0.0, we trust inferred
      momentum = inferred_momentum
    else:
      raise ValueError(
          f'Mismatch between inferred momentum {inferred_momentum} implied '
          f'by name {lr_momentum_matrix_name} and supplied argument '
          f'momentum={momentum}')

    path = _get_matrix_path(
        n=num_rounds, mechanism_name=lr_momentum_matrix_name)
    w_matrix, h_matrix, lr_vector = _load_w_h_and_maybe_lr(path)
    assert lr_vector is not None
    return tff_aggregator.create_residual_momentum_dp_factory(
        tensor_specs=model_weight_specs,
        l2_norm_clip=clip_norm,
        noise_multiplier=noise_multiplier,
        w_matrix=w_matrix,
        h_matrix=h_matrix,
        clients_per_round=clients_per_round,
        seed=noise_seed,
        momentum_value=momentum,
        learning_rates=lr_vector)
  else:
    raise NotImplementedError(
        'Mismatch encountered between aggregation method and pattern-matching '
        'in build_aggregator. This indicates an error in the implementation of '
        'build_aggregator, a missed implementation of an allowed aggregation '
        'method.')
