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
import tensorflow as tf

from lidp_auditing import constants
from lidp_auditing import data


class DataTest(tf.test.TestCase):

  def test_fashion_mnist(self):
    max_num_examples = 1000
    num_canaries = 10
    num_canaries_test = 5
    for canary_type in constants.CANARY_TYPES:
      n, datasets = data.load_fashion_mnist(
          canary_type=canary_type,
          num_canaries=num_canaries,
          num_canaries_test=num_canaries_test,
          seed=0,
          max_num_examples=max_num_examples,
          min_dimension=300,
          max_dimension=784,
          canary_class=5,
          normalize=True,
          synthetic=True,
      )
      train_data, test_data, canary_data, canary_test_data = datasets

      self.assertEqual(train_data.cardinality().numpy(), max_num_examples)
      self.assertEqual(test_data.cardinality().numpy(), max_num_examples)
      self.assertEqual(canary_data.cardinality().numpy(), num_canaries)
      self.assertEqual(
          canary_test_data.cardinality().numpy(), num_canaries_test
      )
      self.assertEqual(n, max_num_examples)


if __name__ == "__main__":
  tf.test.main()
