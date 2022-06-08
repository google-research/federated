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
"""Tests for initializers."""
from absl.testing import absltest
import tensorflow as tf

from dp_matrix_factorization import initializers


class InitializersTest(absltest.TestCase):

  def test_raises_empty_params(self):
    with self.assertRaises(ValueError):
      initializers.get_initial_h('binary_tree', {})

  def test_raises_unknown_initializer(self):
    with self.assertRaises(ValueError):
      initializers.get_initial_h('unk', {})

  def test_constructs_tensor_binary_tree(self):
    matrix = initializers.get_initial_h('binary_tree', {'log_2_leaves': 2})
    self.assertIsInstance(matrix, tf.Tensor)
    self.assertEqual(tf.rank(matrix), 2)

  def test_constructs_tensor_random_binary_tree_structure(self):
    matrix = initializers.get_initial_h('random_binary_tree_structure',
                                        {'log_2_leaves': 2})
    self.assertIsInstance(matrix, tf.Tensor)
    self.assertEqual(tf.rank(matrix), 2)

  def test_constructs_tensor_extended_binary_tree(self):
    matrix = initializers.get_initial_h('extended_binary_tree', {
        'log_2_leaves': 2,
        'num_extra_rows': 1
    })
    self.assertIsInstance(matrix, tf.Tensor)
    self.assertEqual(tf.rank(matrix), 2)

  def test_constructs_tensor_identity(self):
    matrix = initializers.get_initial_h('identity', {'log_2_leaves': 2})
    self.assertIsInstance(matrix, tf.Tensor)
    self.assertEqual(tf.rank(matrix), 2)

  def test_constructs_tensor_double_h(self):
    init_dim = 2
    matrix = initializers.get_initial_h('double_h_solution',
                                        {'h_to_double': tf.eye(init_dim)})
    self.assertIsInstance(matrix, tf.Tensor)
    self.assertEqual(tf.rank(matrix), 2)
    self.assertEqual(matrix.shape.as_list(), [4, 4])


if __name__ == '__main__':
  absltest.main()
