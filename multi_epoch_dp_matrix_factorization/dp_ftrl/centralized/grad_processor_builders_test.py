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
"""Tests for grad_processor_builders."""
import collections

from absl.testing import absltest
from jax import numpy as jnp

from multi_epoch_dp_matrix_factorization.dp_ftrl.centralized import grad_processor_builders
from multi_epoch_dp_matrix_factorization.dp_ftrl.centralized import gradient_processors


class GradProcessorBuildersTest(absltest.TestCase):

  def test_noprivacy_builds(self):
    model_params = collections.OrderedDict(a=jnp.zeros(shape=(100, 100)))
    processor = grad_processor_builders.build_grad_processor(
        model_params=model_params,
        spec=grad_processor_builders.GradProcessorSpec.NO_PRIVACY,
        l2_norm_clip=1.0,
        l2_clip_noise_multiplier=1.0,
        num_epochs=1,
        steps_per_epoch=100,
        noise_seed=0,
        momentum=0.9,
    )
    self.assertIsInstance(
        processor, gradient_processors.DPGradientBatchProcessor
    )


if __name__ == '__main__':
  absltest.main()
