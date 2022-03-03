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

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_federated as tff

from data_poor_fl import coordinate_finalizers

MODEL_WEIGHTS_TYPE = tff.type_at_server(
    tff.to_type(tff.learning.ModelWeights(tf.float32, ())))


class CoordinateFinalizersTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('num_coordinates1', 1),
      ('num_coordinates2', 2),
      ('num_coordinates3', 3),
      ('num_coordinates5', 5),
  )
  def test_build_with_expected_state_length(self, num_coordinates):
    server_optimizer = tff.learning.optimizers.build_sgdm()
    base_finalizer = tff.learning.templates.build_apply_optimizer_finalizer(
        server_optimizer, MODEL_WEIGHTS_TYPE.member)
    finalizer = coordinate_finalizers.build_coordinate_finalizer(
        base_finalizer, num_coordinates=num_coordinates)
    state = finalizer.initialize()
    self.assertLen(state, num_coordinates)

  def test_single_coordinate_matches_base_finalizer(self):
    server_optimizer = tff.learning.optimizers.build_sgdm()
    base_finalizer = tff.learning.templates.build_apply_optimizer_finalizer(
        server_optimizer, MODEL_WEIGHTS_TYPE.member)
    coordinate_finalizer = coordinate_finalizers.build_coordinate_finalizer(
        base_finalizer, num_coordinates=1)

    base_state = base_finalizer.initialize()
    coordinate_state = coordinate_finalizer.initialize()
    self.assertAllClose(base_state, coordinate_state[0])

    weights = tff.learning.ModelWeights(1.0, ())
    update = 0.1
    base_output = base_finalizer.next(base_state, weights, update)
    coordinate_output = coordinate_finalizer.next(coordinate_state, [weights],
                                                  [update])
    self.assertAllClose(base_output.state, coordinate_output.state[0])
    self.assertAllClose(base_output.result.trainable,
                        coordinate_output.result[0].trainable)
    self.assertAllClose(base_output.measurements,
                        coordinate_output.measurements[0])

  def test_coordinate_finalizer_with_three_coordinates(self):
    server_optimizer = tff.learning.optimizers.build_sgdm()
    base_finalizer = tff.learning.templates.build_apply_optimizer_finalizer(
        server_optimizer, MODEL_WEIGHTS_TYPE.member)
    coordinate_finalizer = coordinate_finalizers.build_coordinate_finalizer(
        base_finalizer, num_coordinates=3)
    weights = [
        tff.learning.ModelWeights(1.0, ()),
        tff.learning.ModelWeights(2.0, ()),
        tff.learning.ModelWeights(3.0, ())
    ]
    updates = [4.0, 5.0, 6.0]

    coordinate_state = coordinate_finalizer.initialize()
    coordinate_output = coordinate_finalizer.next(coordinate_state, weights,
                                                  updates)
    actual_result = coordinate_output.result
    base_state = base_finalizer.initialize()
    list_of_base_state = [base_state, base_state, base_state]
    expected_result = [
        base_finalizer.next(a).result
        for a in zip(list_of_base_state, weights, updates)
    ]

    for a, b in zip(actual_result, expected_result):
      self.assertAllClose(a.trainable, b.trainable)


if __name__ == '__main__':
  tf.test.main()
