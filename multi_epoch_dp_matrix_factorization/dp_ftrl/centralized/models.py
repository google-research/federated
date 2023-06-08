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
"""Haiku models for centralized DPMatFac training."""
from collections.abc import Callable

import haiku as hk
from jax import numpy as jnp


def build_vgg_model(
    *,
    n_classes: int,
    dense_size: int,
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray],
) -> hk.Transformed:
  """Builds NCHW-formatted haiku VGG model.

  This implementation is a haiku replica of the PyTorch model defined in the
  commit https://github.com/google-research/DP-FTRL/commit/35b6134b. Notice
  that this model returns logits for n_classes outputs, not the result of a
  softmax.

  Args:
    n_classes: An integer specifying the number of output classes desires.
    dense_size: An integer specifying the input size of the final dense layer.
    activation_fn: A callable implementing the activation function desired.

  Returns:
    An instance of `haiku.Transformed`; two pure functions, implementing model
    parameter initialization abd forward pass.
  """

  def return_conv_block(*, out_channels):
    return [
        hk.Conv2D(
            output_channels=out_channels,
            kernel_shape=[3, 3],
            padding=[1, 1],
            # We are replicating PyTorch models which default to NCHW
            # layout.
            data_format='NCHW',
            # He normal.
            w_init=hk.initializers.VarianceScaling(
                2.0, 'fan_in', 'truncated_normal'
            ),
            b_init=hk.initializers.Constant(0.0),
        ),
        activation_fn,
        hk.Conv2D(
            output_channels=out_channels,
            kernel_shape=[3, 3],
            padding=[1, 1],
            data_format='NCHW',
            # He normal.
            w_init=hk.initializers.VarianceScaling(
                2.0, 'fan_in', 'truncated_normal'
            ),
            b_init=hk.initializers.Constant(0.0),
        ),
        activation_fn,
        # We must ensure that Haiku knows the channels come first.
        hk.MaxPool(window_shape=2, strides=2, channel_axis=1, padding='VALID'),
    ]

  def _forward(images):
    sequential_body = (
        return_conv_block(out_channels=32)
        + return_conv_block(out_channels=64)
        + return_conv_block(out_channels=128)
    )
    vgg = hk.Sequential(
        sequential_body
        + [
            hk.Flatten(),
            hk.Linear(
                dense_size,
                # Glorot normal.
                w_init=hk.initializers.VarianceScaling(
                    1.0, 'fan_avg', 'truncated_normal'
                ),
                b_init=hk.initializers.Constant(0.0),
            ),
            activation_fn,
            hk.Linear(
                n_classes,
                # Glorot normal.
                w_init=hk.initializers.VarianceScaling(
                    1.0, 'fan_avg', 'truncated_normal'
                ),
                b_init=hk.initializers.Constant(0.0),
            ),
        ]
    )
    return vgg(images)

  transformed_forward = hk.transform(_forward)
  return transformed_forward
