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
"""Tests for dot_product_utils."""
import collections

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from one_shot_epe import dot_product_utils


GLOBAL_SEED = 0xBAD5EED


class DotProductUtilsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('vector', tf.zeros((3,))),
      ('matrix', tf.zeros((2, 3))),
      ('struct', [tf.zeros(()), dict(a=tf.zeros((1,)), b=tf.zeros((2,)))]),
  )
  def test_canary_randomness(self, model_weights):
    base_canary = dot_product_utils.packed_canary(
        model_weights, 0, global_seed=GLOBAL_SEED
    )

    different_id_canary = dot_product_utils.packed_canary(
        model_weights, 1, global_seed=GLOBAL_SEED
    )
    tf.nest.map_structure(
        self.assertNotAllEqual, base_canary, different_id_canary
    )

    different_seed_canary = dot_product_utils.packed_canary(
        model_weights, 0, global_seed=GLOBAL_SEED + 1
    )
    tf.nest.map_structure(
        self.assertNotAllEqual, base_canary, different_seed_canary
    )

    resampled_canary = dot_product_utils.packed_canary(
        model_weights, 0, global_seed=GLOBAL_SEED
    )
    tf.nest.map_structure(self.assertAllEqual, base_canary, resampled_canary)

  def test_canary_normalized(self):
    canary = dot_product_utils.packed_canary(
        tf.zeros((10,)), 0, global_seed=GLOBAL_SEED
    )
    self.assertAllClose(1.0, np.linalg.norm(canary.numpy()))

  @parameterized.named_parameters(
      ('vector', tf.zeros((17,)), 17),
      ('matrix', tf.zeros((5, 3)), 15),
      (
          'struct',
          [tf.zeros(()), dict(a=tf.zeros((6,)), b=tf.zeros((2, 4)))],
          13,
      ),
  )
  def test_cosine_distribution(self, model_weights, dim):
    # Expected squared cosine between arbitrary vector and random vector
    # chosen uniformly from the unit sphere is 1/d.
    model_weights = dot_product_utils.packed_canary(
        model_weights, 0, GLOBAL_SEED + 1
    )
    cosines = dot_product_utils.compute_negative_cosines_with_all_canaries(
        model_weights, 1000, GLOBAL_SEED
    )
    cos_sqs = np.square(cosines)
    self.assertAllClose(1 / dim, np.mean(cos_sqs), atol=2e-2)

  @parameterized.named_parameters(
      ('scalar', tf.constant(0), 1),
      ('scalar_tuple', (tf.constant(0),), 1),
      ('scalar_list', [tf.constant(0)], 1),
      (
          'scalar_dict',
          collections.OrderedDict(a=tf.constant(0), b=tf.constant(1)),
          2,
      ),
      ('vector_0', tf.constant([], tf.int32), 0),
      ('vector_1', tf.constant([0]), 1),
      ('vector_2', tf.constant([0, 1]), 2),
      ('matrix_0x1', tf.zeros((0, 1), tf.int32), 0),
      ('matrix_1x0', tf.zeros((1, 0), tf.int32), 0),
      ('matrix_1x1', tf.constant([[0]]), 1),
      ('matrix_2x3', tf.constant([[0, 1, 2], [3, 4, 5]]), 6),
      (
          'complex_struct',
          [
              0,
              (tf.constant([[1, 2], [3, 4]]), tf.constant([[]], tf.int32), 5),
              collections.OrderedDict(
                  a=tf.constant([6, 7]), b=tf.constant([[[8], [9]]])
              ),
          ],
          10,
      ),
  )
  def test_flatten(self, struct, dim):
    self.assertAllEqual(dot_product_utils.flatten(struct), list(range(dim)))


if __name__ == '__main__':
  tf.test.main()
