# Copyright 2018, The TensorFlow Federated Authors.
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

import numpy as np
import tensorflow as tf

from utils import tensor_utils


class TensorUtilsTest(tf.test.TestCase):

  def expect_ok_graph_mode(self, structure):
    with tf.Graph().as_default():
      result, error = tensor_utils.zero_all_if_any_non_finite(structure)
      with self.session() as sess:
        result, error = sess.run((result, error))
      try:
        tf.nest.map_structure(np.testing.assert_allclose, result, structure)
      except AssertionError:
        self.fail('Expected to get input {} back, but instead got {}'.format(
            structure, result))
      self.assertEqual(error, 0)

  def expect_ok_eager_mode(self, structure):
    result, error = tensor_utils.zero_all_if_any_non_finite(structure)
    try:
      tf.nest.map_structure(np.testing.assert_allclose, result, structure)
    except AssertionError:
      self.fail('Expected to get input {} back, but instead got {}'.format(
          structure, result))
    self.assertEqual(error, 0)

  def expect_zeros_graph_mode(self, structure, expected):
    with tf.Graph().as_default():
      result, error = tensor_utils.zero_all_if_any_non_finite(structure)
      with self.session() as sess:
        result, error = sess.run((result, error))
      try:
        tf.nest.map_structure(np.testing.assert_allclose, result, expected)
      except AssertionError:
        self.fail('Expected to get zeros, but instead got {}'.format(result))
      self.assertEqual(error, 1)

  def expect_zeros_eager_mode(self, structure, expected):
    result, error = tensor_utils.zero_all_if_any_non_finite(structure)
    try:
      tf.nest.map_structure(np.testing.assert_allclose, result, expected)
    except AssertionError:
      self.fail('Expected to get zeros, but instead got {}'.format(result))
    self.assertEqual(error, 1)

  def test_zero_all_if_any_non_finite_graph_mode(self):
    tf.config.experimental_run_functions_eagerly(False)
    self.expect_ok_graph_mode([])
    self.expect_ok_graph_mode([(), {}])
    self.expect_ok_graph_mode(1.1)
    self.expect_ok_graph_mode([1.0, 0.0])
    self.expect_ok_graph_mode([1.0, 2.0, {'a': 0.0, 'b': -3.0}])
    self.expect_zeros_graph_mode(np.inf, 0.0)
    self.expect_zeros_graph_mode((1.0, (2.0, np.nan)), (0.0, (0.0, 0.0)))
    self.expect_zeros_graph_mode((1.0, (2.0, {
        'a': 3.0,
        'b': [[np.inf], [np.nan]]
    })), (0.0, (0.0, {
        'a': 0.0,
        'b': [[0.0], [0.0]]
    })))

  def test_zero_all_if_any_non_finite_eager_mode(self):
    tf.config.experimental_run_functions_eagerly(True)
    self.expect_ok_eager_mode([])
    self.expect_ok_eager_mode([(), {}])
    self.expect_ok_eager_mode(1.1)
    self.expect_ok_eager_mode([1.0, 0.0])
    self.expect_ok_eager_mode([1.0, 2.0, {'a': 0.0, 'b': -3.0}])
    self.expect_zeros_eager_mode(np.inf, 0.0)
    self.expect_zeros_eager_mode((1.0, (2.0, np.nan)), (0.0, (0.0, 0.0)))
    self.expect_zeros_eager_mode((1.0, (2.0, {
        'a': 3.0,
        'b': [[np.inf], [np.nan]]
    })), (0.0, (0.0, {
        'a': 0.0,
        'b': [[0.0], [0.0]]
    })))


if __name__ == '__main__':
  tf.test.main()
