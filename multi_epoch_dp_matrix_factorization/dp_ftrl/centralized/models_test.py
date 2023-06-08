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
"""Tests for models."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp

from multi_epoch_dp_matrix_factorization.dp_ftrl.centralized import models


def _return_batch(batch_size):
  # Existing code is all programmed to be channels-first.
  images = jnp.ones(shape=[batch_size, 3, 32, 32], dtype=jnp.float32)
  return images


class ModelsTest(parameterized.TestCase):

  def test_vgg_128_n_parameters(self):
    vgg_model = models.build_vgg_model(
        n_classes=10, dense_size=128, activation_fn=jax.nn.relu
    )
    params = vgg_model.init(jax.random.PRNGKey(2), _return_batch(10))
    flat_params, _ = jax.tree_util.tree_flatten(params)
    n_params = sum(jnp.size(x) for x in flat_params)
    # We pull this number from counting the parameters of the existing VGG
    # implementation in the centralized FTRL library.
    self.assertEqual(n_params, 550_570)

  @parameterized.named_parameters(*tuple((f'{n}', n) for n in range(1, 10)))
  def test_vgg_128_computes_logits_for_batch_size(self, batch_size):
    n_classes = 10
    vgg_model = models.build_vgg_model(
        n_classes=n_classes, dense_size=128, activation_fn=jax.nn.relu
    )
    rng = jax.random.PRNGKey(2)
    batch = _return_batch(batch_size)
    params = vgg_model.init(rng, batch)

    result = vgg_model.apply(params, rng, _return_batch(batch_size))
    self.assertEqual(result.shape, (batch_size, n_classes))

  @parameterized.named_parameters(*tuple((f'{n}', n) for n in range(1, 10)))
  def test_vgg_128_computes_logits_for_n_classes(self, n_classes):
    batch_size = 10
    vgg_model = models.build_vgg_model(
        n_classes=n_classes, dense_size=128, activation_fn=jax.nn.relu
    )
    rng = jax.random.PRNGKey(2)
    batch = _return_batch(batch_size)
    params = vgg_model.init(rng, batch)

    result = vgg_model.apply(params, rng, _return_batch(batch_size))
    self.assertEqual(result.shape, (batch_size, n_classes))


if __name__ == '__main__':
  absltest.main()
