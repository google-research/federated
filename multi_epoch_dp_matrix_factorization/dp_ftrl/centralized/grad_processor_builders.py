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
"""Constructors for the DPGradientProcessor interface."""
import enum

import attr
import haiku as hk
import numpy as np
import tensorflow_federated as tff

from multi_epoch_dp_matrix_factorization import matrix_io
from multi_epoch_dp_matrix_factorization import tff_aggregator
from multi_epoch_dp_matrix_factorization.dp_ftrl.centralized import gradient_processors
from multi_epoch_dp_matrix_factorization.fft import generate_noise
from multi_epoch_dp_matrix_factorization.multiple_participations import contrib_matrix_builders

_NUMERICAL_STABILITY_CONSTANT = 1e-9


@attr.s(frozen=True, auto_attribs=True)
class MatrixMechanismInfo:
  """Data class for holding all necessary information about matrix mechanisms.

  This class is intended to provide a full specification of matrix mechanisms,
  and should be fairly slow to change--it is up to the author of any changes
  to ensure that the values embedded in instances of this class are consistent
  with the data written at the appropriate locations, though consumers of
  instances of this class may use some of the data embedded in the instances
  to perform cursory checks on this consistency. In particular, some of this
  data could conceivably be alternatively derived from data.
  """

  path: str
  num_epochs: int
  steps_per_epoch: int
  sensitivity: float
  momentum: float


@enum.unique
class GradProcessorSpec(enum.Enum):
  """Enum class listing the mechanisms we intend to explore."""

  NO_PRIVACY = 'no_privacy'
  DP_SGD = 'dp_sgd'
  FULL_HONAKER = 'full_honaker'
  FULL_HONAKER_TREE_COMPLETION_1S = 'full_honaker_tree_completion_1s'
  FULL_HONAKER_TREE_COMPLETION_2S = 'full_honaker_tree_completion_2s'
  FULL_HONAKER_TREE_COMPLETION_5S = 'full_honaker_tree_completion_5s'
  FULL_HONAKER_TREE_COMPLETION_10S = 'full_honaker_tree_completion_10s'
  FULL_HONAKER_TREE_COMPLETION_20S = 'full_honaker_tree_completion_20s'
  ONLINE_HONAKER = 'online_honaker'
  ONLINE_HONAKER_TREE_COMPLETION_1S = 'online_honaker_tree_completion_1s'
  ONLINE_HONAKER_TREE_COMPLETION_2S = 'online_honaker_tree_completion_2s'
  ONLINE_HONAKER_TREE_COMPLETION_5S = 'online_honaker_tree_completion_5s'
  ONLINE_HONAKER_TREE_COMPLETION_10S = 'online_honaker_tree_completion_10s'
  ONLINE_HONAKER_TREE_COMPLETION_20S = 'online_honaker_tree_completion_20s'
  PREFIX_SUM_OPT_SINGLE_EPOCH_2000_STEPS = (
      'prefix_sum_opt_single_epoch_2000_steps'
  )
  PREFIX_SUM_OPT_SINGLE_EPOCH_100_STEPS = (
      'prefix_sum_opt_single_epoch_100_steps'
  )
  PREFIX_SUM_OPT_SINGLE_EPOCH_100_STEPS_DENISOV = (
      'prefix_sum_opt_single_epoch_100_steps_denisov'
  )
  PREFIX_SUM_OPT_FIVE_EPOCHS_100_STEPS = 'prefix_sum_opt_five_epochs_100_steps'
  PREFIX_SUM_OPT_TEN_EPOCHS_100_STEPS = 'prefix_sum_opt_ten_epochs_100_steps'
  PREFIX_SUM_OPT_TWENTY_EPOCHS_100_STEPS = (
      'prefix_sum_opt_twenty_epochs_100_steps'
  )
  KOSKOLOVA_OPT_TWENTY_EPOCHS_100_STEPS = (
      'koskolova_opt_twenty_epochs_100_steps'
  )
  # FFT Optimal Decoder
  FFT_OPTIMAL_DECODER_1S = 'fft_optimal_decoder_1s'
  FFT_OPTIMAL_DECODER_2S = 'fft_optimal_decoder_2s'
  FFT_OPTIMAL_DECODER_5S = 'fft_optimal_decoder_5s'
  FFT_OPTIMAL_DECODER_10S = 'fft_optimal_decoder_10s'
  FFT_OPTIMAL_DECODER_20S = 'fft_optimal_decoder_20s'
  # FFT with fixing outputs
  FFT_1S = 'fft_1s'
  FFT_2S = 'fft_2s'
  FFT_4S = 'fft_4s'
  FFT_5S = 'fft_5s'
  FFT_10S = 'fft_10s'
  FFT_20S = 'fft_20s'


_MATRIX_KEY_TO_INFO = {
    GradProcessorSpec.KOSKOLOVA_OPT_TWENTY_EPOCHS_100_STEPS: (
        MatrixMechanismInfo(
            path='TODO_GENERATE_THIS',
            num_epochs=20,
            steps_per_epoch=100,
            sensitivity=1.0,
            momentum=0.0,
        )
    ),
    GradProcessorSpec.PREFIX_SUM_OPT_SINGLE_EPOCH_2000_STEPS: (
        MatrixMechanismInfo(
            path='TODO_GENERATE_THIS',
            num_epochs=20,
            steps_per_epoch=100,
            sensitivity=1.0,
            momentum=0.0,
        )
    ),
    GradProcessorSpec.PREFIX_SUM_OPT_SINGLE_EPOCH_100_STEPS: (
        MatrixMechanismInfo(
            path='TODO_GENERATE_THIS',
            num_epochs=1,
            steps_per_epoch=100,
            sensitivity=1.0,
            momentum=0.0,
        )
    ),
    GradProcessorSpec.PREFIX_SUM_OPT_FIVE_EPOCHS_100_STEPS: MatrixMechanismInfo(
        path='TODO_GENERATE_THIS',
        num_epochs=20,
        steps_per_epoch=100,
        sensitivity=1.0,
        momentum=0.0,
    ),
    GradProcessorSpec.PREFIX_SUM_OPT_TEN_EPOCHS_100_STEPS: MatrixMechanismInfo(
        path='TODO_GENERATE_THIS',
        num_epochs=20,
        steps_per_epoch=100,
        sensitivity=1.0,
        momentum=0.0,
    ),
    GradProcessorSpec.PREFIX_SUM_OPT_TWENTY_EPOCHS_100_STEPS: (
        MatrixMechanismInfo(
            path='TODO_GENERATE_THIS',
            num_epochs=20,
            steps_per_epoch=100,
            sensitivity=1.0,
            momentum=0.0,
        )
    ),
    GradProcessorSpec.FULL_HONAKER: MatrixMechanismInfo(
        path='TODO_GENERATE_THIS',
        num_epochs=20,
        steps_per_epoch=100,
        sensitivity=1.0,
        momentum=0.0,
    ),
    GradProcessorSpec.FULL_HONAKER_TREE_COMPLETION_1S: MatrixMechanismInfo(
        path='TODO_GENERATE_THIS',
        num_epochs=20,
        steps_per_epoch=100,
        sensitivity=1.0,
        momentum=0.0,
    ),
    GradProcessorSpec.FULL_HONAKER_TREE_COMPLETION_2S: MatrixMechanismInfo(
        path='TODO_GENERATE_THIS',
        num_epochs=20,
        steps_per_epoch=100,
        sensitivity=1.0,
        momentum=0.0,
    ),
    GradProcessorSpec.FULL_HONAKER_TREE_COMPLETION_5S: MatrixMechanismInfo(
        path='TODO_GENERATE_THIS',
        num_epochs=20,
        steps_per_epoch=100,
        sensitivity=1.0,
        momentum=0.0,
    ),
    GradProcessorSpec.FULL_HONAKER_TREE_COMPLETION_10S: MatrixMechanismInfo(
        path='TODO_GENERATE_THIS',
        num_epochs=20,
        steps_per_epoch=100,
        sensitivity=1.0,
        momentum=0.0,
    ),
    GradProcessorSpec.FULL_HONAKER_TREE_COMPLETION_20S: MatrixMechanismInfo(
        path='TODO_GENERATE_THIS',
        num_epochs=20,
        steps_per_epoch=100,
        sensitivity=1.0,
        momentum=0.0,
    ),
    GradProcessorSpec.ONLINE_HONAKER: MatrixMechanismInfo(
        path='TODO_GENERATE_THIS',
        num_epochs=20,
        steps_per_epoch=100,
        sensitivity=1.0,
        momentum=0.0,
    ),
    GradProcessorSpec.ONLINE_HONAKER_TREE_COMPLETION_1S: MatrixMechanismInfo(
        path='TODO_GENERATE_THIS',
        num_epochs=20,
        steps_per_epoch=100,
        sensitivity=1.0,
        momentum=0.0,
    ),
    GradProcessorSpec.ONLINE_HONAKER_TREE_COMPLETION_2S: MatrixMechanismInfo(
        path='TODO_GENERATE_THIS',
        num_epochs=20,
        steps_per_epoch=100,
        sensitivity=1.0,
        momentum=0.0,
    ),
    GradProcessorSpec.ONLINE_HONAKER_TREE_COMPLETION_5S: MatrixMechanismInfo(
        path='TODO_GENERATE_THIS',
        num_epochs=20,
        steps_per_epoch=100,
        sensitivity=1.0,
        momentum=0.0,
    ),
    GradProcessorSpec.ONLINE_HONAKER_TREE_COMPLETION_10S: MatrixMechanismInfo(
        path='TODO_GENERATE_THIS',
        num_epochs=20,
        steps_per_epoch=100,
        sensitivity=1.0,
        momentum=0.0,
    ),
    GradProcessorSpec.ONLINE_HONAKER_TREE_COMPLETION_20S: MatrixMechanismInfo(
        path='TODO_GENERATE_THIS',
        num_epochs=20,
        steps_per_epoch=100,
        sensitivity=1.0,
        momentum=0.0,
    ),
    GradProcessorSpec.FFT_1S: MatrixMechanismInfo(
        path='TODO_GENERATE_THIS',
        num_epochs=20,
        steps_per_epoch=100,
        sensitivity=1.0,
        momentum=0.0,
    ),
    GradProcessorSpec.FFT_2S: MatrixMechanismInfo(
        path='TODO_GENERATE_THIS',
        num_epochs=20,
        steps_per_epoch=100,
        sensitivity=1.0,
        momentum=0.0,
    ),
    GradProcessorSpec.FFT_4S: MatrixMechanismInfo(
        path='TODO_GENERATE_THIS',
        num_epochs=20,
        steps_per_epoch=100,
        sensitivity=1.0,
        momentum=0.0,
    ),
    GradProcessorSpec.FFT_5S: MatrixMechanismInfo(
        path='TODO_GENERATE_THIS',
        num_epochs=20,
        steps_per_epoch=100,
        sensitivity=1.0,
        momentum=0.0,
    ),
    GradProcessorSpec.FFT_10S: MatrixMechanismInfo(
        path='TODO_GENERATE_THIS',
        num_epochs=20,
        steps_per_epoch=100,
        sensitivity=1.0,
        momentum=0.0,
    ),
    GradProcessorSpec.FFT_20S: MatrixMechanismInfo(
        path='TODO_GENERATE_THIS',
        num_epochs=20,
        steps_per_epoch=100,
        sensitivity=1.0,
        momentum=0.0,
    ),
    GradProcessorSpec.FFT_OPTIMAL_DECODER_5S: MatrixMechanismInfo(
        path='TODO_GENERATE_THIS',
        num_epochs=20,
        steps_per_epoch=100,
        sensitivity=1.0,
        momentum=0.0,
    ),
    GradProcessorSpec.FFT_OPTIMAL_DECODER_10S: MatrixMechanismInfo(
        path='TODO_GENERATE_THIS',
        num_epochs=20,
        steps_per_epoch=100,
        sensitivity=1.0,
        momentum=0.0,
    ),
    GradProcessorSpec.FFT_OPTIMAL_DECODER_20S: MatrixMechanismInfo(
        path='TODO_GENERATE_THIS',
        num_epochs=20,
        steps_per_epoch=100,
        sensitivity=1.0,
        momentum=0.0,
    ),
}


def _sensitivity_for_c(c: np.ndarray, contrib_matrix: np.ndarray) -> float:
  """Computes sensitivity for all positive C matrices."""
  x_matrix = c.T @ c
  return np.max(np.diag(contrib_matrix.T @ x_matrix @ contrib_matrix)) ** 0.5


def _check_factorization_against_spec(w_matrix, h_matrix, lr_tensor, mech_info):
  """Asserts consistency between the matrices and the specified mechanism."""
  del lr_tensor  # Currently unused.
  if w_matrix.shape[0] < (mech_info.num_epochs * mech_info.steps_per_epoch):
    raise ValueError(
        'Mismatched configuration between matrix mechanism '
        'info and loaded matrix. Loaded matrix supports '
        f'{w_matrix.shape[0]} steps, while mechanism is '
        f'specified for {mech_info.num_epochs} epochs with '
        f'{mech_info.steps_per_epoch} steps per epoch.'
    )
  num_steps = mech_info.steps_per_epoch * mech_info.num_epochs
  h_np = h_matrix.numpy()
  min_x_value = np.min(h_np.T @ h_np)

  h_all_positive = min_x_value >= -_NUMERICAL_STABILITY_CONSTANT
  if not h_all_positive:
    sensitivity = generate_noise.get_spectral_norm_sensitivity(
        h_np[:, :num_steps], num_steps, mech_info.num_epochs
    )
  else:
    contrib_matrix = (
        contrib_matrix_builders.epoch_participation_matrix_all_positive(
            num_steps, mech_info.num_epochs
        )
    )
    sensitivity = _sensitivity_for_c(h_np[:, :num_steps], contrib_matrix)

  if sensitivity > mech_info.sensitivity + _NUMERICAL_STABILITY_CONSTANT:
    raise ValueError(
        'H matrix has higher sensitivity for '
        f'{mech_info.num_epochs} epochs, '
        f'{mech_info.steps_per_epoch} than expected. Expected '
        f'sensitivity {mech_info.sensitivity} (within numerical '
        f'tolerance), computed sensitivity {sensitivity}.'
    )


def build_grad_processor(
    *,
    model_params: hk.Params,
    spec: GradProcessorSpec,
    l2_norm_clip: float,
    # It's not clear that we truly want to parameterize
    # this way. But it's close to the way we have
    # parameterized the code this will be calling, and it's
    # always possible to expose a different interface for
    # this 'privacy' value.
    l2_clip_noise_multiplier: float,
    num_epochs: int,
    steps_per_epoch: int,
    noise_seed: int,
    momentum: float,
) -> gradient_processors.DPGradientBatchProcessor:
  """Builds an instance of `gradient_processors.DPGradientBatchProcessor`."""
  del momentum  # Unused
  if spec == GradProcessorSpec.NO_PRIVACY:
    # We special-case this for efficiency of the implementation--we just need to
    # compute a mean.
    return gradient_processors.NoPrivacyGradientProcessor()
  elif spec in [GradProcessorSpec.DP_SGD]:
    model_params_struct = tff.structure.from_container(
        model_params, recursive=True
    )
    model_params_odict = tff.structure.to_odict(
        model_params_struct, recursive=True
    )
    model_params_tff_type = tff.types.type_from_tensors(model_params_odict)
    model_tensor_specs = tff.types.type_to_tf_tensor_specs(
        model_params_tff_type
    )
    aggregator_factory = (
        tff.aggregators.DifferentiallyPrivateFactory.gaussian_fixed(
            l2_clip_noise_multiplier, 1, l2_norm_clip
        )
    )
    aggregator = aggregator_factory.create(model_params_tff_type)
    return gradient_processors.DPAggregatorBackedGradientProcessor(
        dp_aggregator=aggregator, l2_norm_clip=l2_norm_clip
    )

  model_params_struct = tff.structure.from_container(
      model_params, recursive=True
  )
  model_params_odict = tff.structure.to_odict(
      model_params_struct, recursive=True
  )
  model_params_tff_type = tff.types.type_from_tensors(model_params_odict)
  model_tensor_specs = tff.types.type_to_tf_tensor_specs(model_params_tff_type)
  if spec in _MATRIX_KEY_TO_INFO:
    mech_info = _MATRIX_KEY_TO_INFO[spec]
    # (w, h) = (B, C) from multi-epoch paper: http://arxiv.org/abs/2211.06530
    w_matrix, h_matrix, lr_tensor = matrix_io.load_w_h_and_maybe_lr(
        mech_info.path
    )
    _check_factorization_against_spec(w_matrix, h_matrix, lr_tensor, mech_info)
    aggregator_factory = tff_aggregator.create_residual_prefix_sum_dp_factory(
        tensor_specs=model_tensor_specs,
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=l2_clip_noise_multiplier,
        w_matrix=w_matrix,
        h_matrix=h_matrix,
        # We use this as a *pure* noise generator now--see the comments on the
        # aggregator-backed gradient processor's docstring for the provenance of
        # this 1.
        clients_per_round=1,
        seed=noise_seed,
    )
    aggregator = aggregator_factory.create(model_params_tff_type)
    return gradient_processors.DPAggregatorBackedGradientProcessor(
        dp_aggregator=aggregator, l2_norm_clip=l2_norm_clip
    )
  else:
    raise ValueError(
        f'As-of-yet unsupported mechanism {spec} with n_epochs {num_epochs} '
        f'and steps per epoch {steps_per_epoch}; generate the appropriate data '
        'and plumb through grad_processor_builders.'
    )
