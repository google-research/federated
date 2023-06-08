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
"""Library for utilities to construct single-epoch aggregators for DPFTRL.

This library is written for Google-internal use, and may read from network
(e.g., owning CNS paths which contain matrix factorizations), and therefore
tests for this code must be written carefully to avoid such network access.
This file will, however, eventually be branched / migrated to OSS, and the paths
will be updated to point to publicly-available storage, once we write and
publish the paper to which this code is associated.
"""

from collections.abc import Callable
from typing import Any, Optional, Union

import tensorflow_federated as tff

from multi_epoch_dp_matrix_factorization import matrix_io
from multi_epoch_dp_matrix_factorization import tff_aggregator


AGGREGATION_METHODS = frozenset({
    'tree_aggregation',
    'opt_prefix_sum_matrix',
    'streaming_honaker_matrix',
    'full_honaker_matrix',
    'opt_momentum_matrix',
    'lr_momentum_matrix',
    'dp_sgd',
})


def build_aggregator(
    *,
    aggregator_method: str,
    model_fn: Callable[[], tff.learning.models.VariableModel],
    clip_norm: float,
    noise_multiplier: float,
    clients_per_round: int,
    num_rounds: int,
    noise_seed: Optional[int],
    momentum: float = 0.0,
    lr_momentum_matrix_name: Optional[str] = None,
    verify_sensitivity_fn: Optional[Callable[[Any], None]] = None,
) -> Union[
    tff.aggregators.DifferentiallyPrivateFactory,
    tff.aggregators.UnweightedAggregationFactory,
]:
  """Builds DP aggregators for integration with DPFTRLM tff.learning process."""
  if verify_sensitivity_fn is None:
    verify_sensitivity_fn = lambda _: None

  if clip_norm <= 0:
    raise ValueError(
        f'`clip_norm` must be positive; got clip norm {clip_norm}.'
    )
  if clients_per_round <= 0:
    raise ValueError(
        '`clients_per_round` must be positive; '
        f'got report goal {clients_per_round}.'
    )
  if noise_multiplier < 0:
    raise ValueError(
        '`noise_multiplier` must be nonnegative; '
        f'got noise multiplier {noise_multiplier}.'
    )
  if num_rounds <= 0:
    raise ValueError(
        f'`num_rounds` must be positive; got num rounds {num_rounds}.'
    )
  if momentum < 0:
    raise ValueError(
        f'`momentum` must be nonnegative; got momentum {momentum}.'
    )

  if lr_momentum_matrix_name and aggregator_method != 'lr_momentum_matrix':
    raise ValueError(
        '`lr_momentum_matrix_name` is only supported when'
        'aggregator_method="lr_momentum_matrix"'
    )

  model_weight_specs = tff.types.type_to_tf_tensor_specs(
      tff.learning.models.weights_type_from_model(model_fn).trainable
  )

  if aggregator_method not in AGGREGATION_METHODS:
    raise NotImplementedError(
        f'Aggregator method {aggregator_method} not known. Supported '
        'aggregation methods: \n'
        + ''.join([f'{x} \n' for x in AGGREGATION_METHODS])
    )

  if aggregator_method == 'dp_sgd':
    # A special case, useful for baselines or unnoised training.
    dp_query = tff.aggregators.DifferentiallyPrivateFactory.gaussian_fixed(
        noise_multiplier=noise_multiplier,
        clients_per_round=clients_per_round,
        clip=clip_norm,
    )
    measured_query = tff.learning.add_debug_measurements(dp_query)
    return measured_query
  elif aggregator_method == 'tree_aggregation':
    return tff.aggregators.DifferentiallyPrivateFactory.tree_aggregation(
        noise_multiplier=noise_multiplier,
        clients_per_round=clients_per_round,
        l2_norm_clip=clip_norm,
        record_specs=model_weight_specs,
        noise_seed=noise_seed,
        use_efficient=True,
    )
  elif aggregator_method in [  # Prefix sum methods
      'opt_prefix_sum_matrix',
      'streaming_honaker_matrix',
      'full_honaker_matrix',
  ]:
    w_matrix, h_matrix = matrix_io.get_prefix_sum_w_h(
        num_rounds, aggregator_method
    )
    verify_sensitivity_fn(h_matrix)
    return tff_aggregator.create_residual_prefix_sum_dp_factory(
        tensor_specs=model_weight_specs,
        l2_norm_clip=clip_norm,
        noise_multiplier=noise_multiplier,
        w_matrix=w_matrix,
        h_matrix=h_matrix,
        clients_per_round=clients_per_round,
        seed=noise_seed,
    )
  elif aggregator_method == 'opt_momentum_matrix':
    path = matrix_io.get_momentum_path(num_rounds, momentum)
    w_matrix, h_matrix, lr_vector = matrix_io.load_w_h_and_maybe_lr(path)
    verify_sensitivity_fn(h_matrix)
    assert lr_vector is None
    return tff_aggregator.create_residual_momentum_dp_factory(
        tensor_specs=model_weight_specs,
        l2_norm_clip=clip_norm,
        noise_multiplier=noise_multiplier,
        w_matrix=w_matrix,
        h_matrix=h_matrix,
        clients_per_round=clients_per_round,
        seed=noise_seed,
        momentum_value=momentum,
    )
  elif aggregator_method == 'lr_momentum_matrix':
    if lr_momentum_matrix_name is None:
      raise ValueError(
          'Must supply `lr_momentum_matrix_name` for the '
          'lr_momentum_matrix method.'
      )
    inferred_momentum = matrix_io.infer_momentum_from_path(
        lr_momentum_matrix_name
    )

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
          f'momentum={momentum}'
      )

    path = matrix_io.get_matrix_path(
        n=num_rounds, mechanism_name=lr_momentum_matrix_name
    )
    w_matrix, h_matrix, lr_vector = matrix_io.load_w_h_and_maybe_lr(path)
    verify_sensitivity_fn(h_matrix)
    return tff_aggregator.create_residual_momentum_dp_factory(
        tensor_specs=model_weight_specs,
        l2_norm_clip=clip_norm,
        noise_multiplier=noise_multiplier,
        w_matrix=w_matrix,
        h_matrix=h_matrix,
        clients_per_round=clients_per_round,
        seed=noise_seed,
        momentum_value=momentum,
        learning_rates=lr_vector,
    )
  else:
    raise NotImplementedError(
        'Mismatch encountered between aggregation method and pattern-matching '
        'in build_aggregator. This indicates an error in the implementation of '
        'build_aggregator, a missed implementation of an allowed aggregation '
        'method.'
    )
