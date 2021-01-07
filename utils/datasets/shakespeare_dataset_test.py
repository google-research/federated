# Copyright 2019, Google LLC.
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

import collections
import tensorflow as tf

from utils.datasets import shakespeare_dataset


def _compute_length_of_dataset(ds):
  return ds.reduce(0, lambda x, _: x + 1)


class PreprocessFnTest(tf.test.TestCase):

  def test_to_ids(self):
    pad, _, bos, eos = shakespeare_dataset.get_special_tokens()
    to_tokens = shakespeare_dataset._build_tokenize_fn(split_length=5)
    tokens = self.evaluate(to_tokens({'snippets': tf.constant('abc')}))
    self.assertAllEqual(tokens, [bos, 64, 42, 21, eos])
    to_tokens = shakespeare_dataset._build_tokenize_fn(split_length=12)
    tokens = self.evaluate(to_tokens({'snippets': tf.constant('star wars')}))
    self.assertAllEqual(tokens,
                        [bos, 25, 5, 64, 46, 14, 26, 64, 46, 25, eos, pad])

  def test_last_id_not_oov(self):
    _, oov, bos, eos = shakespeare_dataset.get_special_tokens()
    to_tokens = shakespeare_dataset._build_tokenize_fn(split_length=5)
    tokens = to_tokens({'snippets': tf.constant('a\r~')})
    self.assertAllEqual(tokens, [bos, 64, 86, oov, eos])

  def test_split_target(self):
    example = self.evaluate(
        shakespeare_dataset._split_target(tf.constant([[1, 2, 3]])))
    self.assertAllEqual(([[1, 2]], [[2, 3]]), example)
    example = self.evaluate(
        shakespeare_dataset._split_target(tf.constant([[1, 2, 3], [4, 5, 6]])))
    self.assertAllEqual((
        [[1, 2], [4, 5]],
        [[2, 3], [5, 6]],
    ), example)

  def test_preprocess_fn(self):
    pad, _, bos, eos = shakespeare_dataset.get_special_tokens()
    initial_ds = tf.data.Dataset.from_tensor_slices(
        collections.OrderedDict(snippets=['a snippet', 'different snippet']))
    preprocess_fn = shakespeare_dataset.create_preprocess_fn(
        num_epochs=2, batch_size=2, shuffle_buffer_size=1, sequence_length=10)

    ds = preprocess_fn(initial_ds)
    expected_outputs = [
        # First batch.
        ([[bos, 64, 14, 25, 45, 66, 4, 4, 65, 5],
          [bos, 1, 66, 43, 43, 65, 46, 65, 45,
           5]], [[64, 14, 25, 45, 66, 4, 4, 65, 5, eos],
                 [1, 66, 43, 43, 65, 46, 65, 45, 5, 14]]),
        # Second batch.
        ([
            [25, 45, 66, 4, 4, 65, 5, eos, pad, pad],
            [bos, 64, 14, 25, 45, 66, 4, 4, 65, 5],
        ], [
            [45, 66, 4, 4, 65, 5, eos, pad, pad, pad],
            [64, 14, 25, 45, 66, 4, 4, 65, 5, eos],
        ]),
        # Third batch.
        ([[bos, 1, 66, 43, 43, 65, 46, 65, 45, 5],
          [25, 45, 66, 4, 4, 65, 5, eos, pad,
           pad]], [[1, 66, 43, 43, 65, 46, 65, 45, 5, 14],
                   [45, 66, 4, 4, 65, 5, eos, pad, pad, pad]]),
    ]
    for batch_num, actual in enumerate(ds):
      self.assertGreater(
          len(expected_outputs),
          0,
          msg='Actual output contains more than expected.\nActual: {!s}'.format(
              actual))
      expected = expected_outputs.pop(0)
      self.assertAllEqual(
          actual,
          expected,
          msg='Batch {:d} not equal. Actual: {!s}\nExpected: {!s}'.format(
              batch_num, actual, expected))
    self.assertLen(
        expected_outputs,
        0,
        msg='Not all expected output seen.\nLeft over expectations: {!s}'
        .format(expected_outputs))


class FederatedDatasetTest(tf.test.TestCase):

  def test_raises_negative_epochs_per_round(self):
    with self.assertRaisesRegex(
        ValueError,
        'train_client_epochs_per_round must be a positive integer.'):
      shakespeare_dataset.get_federated_datasets(
          train_client_batch_size=10, train_client_epochs_per_round=-1)


if __name__ == '__main__':
  tf.test.main()
