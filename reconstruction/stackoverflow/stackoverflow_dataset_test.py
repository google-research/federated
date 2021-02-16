# Copyright 2020, Google LLC.
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
"""Tests for stackoverflow_dataset.py."""

import collections

import numpy as np
import tensorflow as tf

from reconstruction.stackoverflow import stackoverflow_dataset

SINGLE_EXAMPLE_TEST_DATA = collections.OrderedDict(
    creation_date=(['2008-08-10 08:28:52.1 UTC']),
    title=(['unused title']),
    score=([tf.constant(0, dtype=tf.int64)]),
    tags=(['unused test tag']),
    tokens=(['one must imagine']),
    type=(['unused type']),
)

MULTI_EXAMPLE_TEST_DATA = collections.OrderedDict(
    creation_date=([
        '2008-08-10 08:28:52.1 UTC', '2007-03-01T13:00:00.67279Z',
        '2007-03-01T13:00:00.67279Z'
    ]),
    title=(['unused title', 'unused title', 'unused title']),
    score=(tf.constant([0, 0, 0], tf.int64)),
    tags=(['unused test tag', 'unused test tag', 'unused test tag']),
    tokens=(['one must imagine', 'imagine', 'one']),
    type=(['unused type', 'unused type', 'unused type']),
)

FEATURE_DTYPES = collections.OrderedDict(
    creation_date=tf.string,
    title=tf.string,
    score=tf.int64,
    tags=tf.string,
    tokens=tf.string,
    type=tf.string,
)


class StackoverflowDatasetTest(tf.test.TestCase):

  def test_creation_date_string_to_integer_returns_correct_values(self):
    creation_dates = tf.constant([
        '2009-06-15T13:45:30', '2020-12-30T22:58:55Z', '2007-03-01T13:00:00Z',
        '2007-03-01T13:00:00.67279Z', '2010-01-08 09:34:05 UTC',
        '2008-08-10 08:28:52.1 UTC', '2009-03-03 21:32:46.8 UTC'
    ], tf.string)
    creation_integers = stackoverflow_dataset._creation_date_string_to_integer(
        creation_dates)
    self.assertAllEqual(creation_integers, [
        20090615134530, 20201230225855, 20070301130000, 20070301130000,
        20100108093405, 20080810082852, 20090303213246
    ])

  def test_sort_timestamps_sorts_examples(self):
    sorted_examples = stackoverflow_dataset._sort_examples_by_date(
        MULTI_EXAMPLE_TEST_DATA)
    # This also ensures sort is stable, i.e. ties are ordered as they are
    # originally.
    expected_sorted_examples = collections.OrderedDict(
        creation_date=tf.constant([
            '2007-03-01T13:00:00.67279Z', '2007-03-01T13:00:00.67279Z',
            '2008-08-10 08:28:52.1 UTC'
        ]),
        title=tf.constant(['unused title', 'unused title', 'unused title']),
        score=tf.constant([0, 0, 0], tf.int64),
        tags=tf.constant(
            ['unused test tag', 'unused test tag', 'unused test tag']),
        tokens=tf.constant(['imagine', 'one', 'one must imagine']),
        type=tf.constant(['unused type', 'unused type', 'unused type']))

    self.assertAllEqual(sorted_examples.keys(), expected_sorted_examples.keys())
    for key in expected_sorted_examples:
      self.assertAllEqual(sorted_examples[key], expected_sorted_examples[key])

  def test_preprocess_fn_return_dataset_element_spec(self):
    ds = tf.data.Dataset.from_tensor_slices(SINGLE_EXAMPLE_TEST_DATA)
    preprocess_fn = stackoverflow_dataset.create_preprocess_fn(
        client_batch_size=32,
        max_sequence_length=10,
        max_elements_per_client=100,
        vocab=['one', 'must'],
        num_oov_buckets=1,
        feature_dtypes=FEATURE_DTYPES,
        sort_by_date=False)
    preprocessed_ds = preprocess_fn(ds)
    self.assertEqual(preprocessed_ds.element_spec,
                     (tf.TensorSpec(shape=[None, 10], dtype=tf.int64),
                      tf.TensorSpec(shape=[None, 10], dtype=tf.int64)))

  def test_preprocess_fn_bad_batch_size_fails(self):
    with self.assertRaisesRegex(ValueError, 'client_batch_size'):
      stackoverflow_dataset.create_preprocess_fn(
          client_batch_size=0,
          max_sequence_length=10,
          max_elements_per_client=100,
          vocab=['one', 'must'],
          num_oov_buckets=1,
          feature_dtypes=FEATURE_DTYPES)

  def test_preprocess_fn_bad_max_sequence_length_fails(self):
    with self.assertRaisesRegex(ValueError, 'max_sequence_length'):
      stackoverflow_dataset.create_preprocess_fn(
          client_batch_size=32,
          max_sequence_length=0,
          max_elements_per_client=100,
          vocab=['one', 'must'],
          num_oov_buckets=1,
          feature_dtypes=FEATURE_DTYPES)

  def test_preprocess_fn_bad_max_elements_fails(self):
    with self.assertRaisesRegex(ValueError, 'max_elements_per_client'):
      stackoverflow_dataset.create_preprocess_fn(
          client_batch_size=32,
          max_sequence_length=20,
          max_elements_per_client=0,
          vocab=['one', 'must'],
          num_oov_buckets=1,
          feature_dtypes=FEATURE_DTYPES)

  def test_preprocess_fn_return_dataset_element_spec_oov_buckets(self):
    ds = tf.data.Dataset.from_tensor_slices(SINGLE_EXAMPLE_TEST_DATA)
    preprocess_fn = stackoverflow_dataset.create_preprocess_fn(
        client_batch_size=32,
        max_sequence_length=10,
        max_elements_per_client=100,
        vocab=['one', 'must'],
        num_oov_buckets=10,
        feature_dtypes=FEATURE_DTYPES,
        sort_by_date=False)
    preprocessed_ds = preprocess_fn(ds)
    self.assertEqual(preprocessed_ds.element_spec,
                     (tf.TensorSpec(shape=[None, 10], dtype=tf.int64),
                      tf.TensorSpec(shape=[None, 10], dtype=tf.int64)))

  def test_preprocess_fn_return_dataset_element_spec_sort_by_date(self):
    ds = tf.data.Dataset.from_tensor_slices(MULTI_EXAMPLE_TEST_DATA)
    preprocess_fn = stackoverflow_dataset.create_preprocess_fn(
        client_batch_size=32,
        max_sequence_length=10,
        max_elements_per_client=100,
        vocab=['one', 'must'],
        num_oov_buckets=10,
        feature_dtypes=FEATURE_DTYPES,
        sort_by_date=True)
    preprocessed_ds = preprocess_fn(ds)
    self.assertEqual(preprocessed_ds.element_spec,
                     (tf.TensorSpec(shape=[None, 10], dtype=tf.int64),
                      tf.TensorSpec(shape=[None, 10], dtype=tf.int64)))

  def test_preprocess_fn_return_dataset_single_element_spec_sort_by_date(self):
    ds = tf.data.Dataset.from_tensor_slices(SINGLE_EXAMPLE_TEST_DATA)
    preprocess_fn = stackoverflow_dataset.create_preprocess_fn(
        client_batch_size=32,
        max_sequence_length=10,
        max_elements_per_client=100,
        vocab=['one', 'must'],
        num_oov_buckets=10,
        feature_dtypes=FEATURE_DTYPES,
        sort_by_date=True)
    preprocessed_ds = preprocess_fn(ds)
    self.assertEqual(preprocessed_ds.element_spec,
                     (tf.TensorSpec(shape=[None, 10], dtype=tf.int64),
                      tf.TensorSpec(shape=[None, 10], dtype=tf.int64)))

  def test_preprocess_fn_returns_correct_sequence(self):
    ds = tf.data.Dataset.from_tensor_slices(SINGLE_EXAMPLE_TEST_DATA)
    preprocess_fn = stackoverflow_dataset.create_preprocess_fn(
        client_batch_size=32,
        max_sequence_length=6,
        max_elements_per_client=100,
        vocab=['one', 'must'],
        num_oov_buckets=1,
        feature_dtypes=FEATURE_DTYPES,
        sort_by_date=False)

    preprocessed_ds = preprocess_fn(ds)
    element = next(iter(preprocessed_ds))

    # BOS is len(vocab)+2, EOS is len(vocab)+3, pad is 0, OOV is len(vocab)+1
    self.assertAllEqual(
        self.evaluate(element[0]), np.array([[4, 1, 2, 3, 5, 0]]))

  def test_preprocess_fn_returns_correct_sequence_oov_buckets(self):
    ds = tf.data.Dataset.from_tensor_slices(SINGLE_EXAMPLE_TEST_DATA)
    preprocess_fn = stackoverflow_dataset.create_preprocess_fn(
        client_batch_size=32,
        max_sequence_length=6,
        max_elements_per_client=100,
        vocab=['one', 'must'],
        num_oov_buckets=3,
        feature_dtypes=FEATURE_DTYPES,
        sort_by_date=False)
    preprocessed_ds = preprocess_fn(ds)
    element = next(iter(preprocessed_ds))
    # BOS is len(vocab)+3+1
    self.assertEqual(self.evaluate(element[0])[0][0], 6)
    self.assertEqual(self.evaluate(element[0])[0][1], 1)
    self.assertEqual(self.evaluate(element[0])[0][2], 2)
    # OOV is [len(vocab)+1, len(vocab)+2, len(vocab)+3]
    self.assertIn(self.evaluate(element[0])[0][3], [3, 4, 5])
    # EOS is len(vocab)+3+2
    self.assertEqual(self.evaluate(element[0])[0][4], 7)
    # pad is 0
    self.assertEqual(self.evaluate(element[0])[0][5], 0)

  def test_preprocess_fn_returns_correct_sequence_sort_by_date(self):
    ds = tf.data.Dataset.from_tensor_slices(MULTI_EXAMPLE_TEST_DATA)
    preprocess_fn = stackoverflow_dataset.create_preprocess_fn(
        client_batch_size=32,
        max_sequence_length=6,
        max_elements_per_client=100,
        vocab=['one', 'must'],
        num_oov_buckets=1,
        feature_dtypes=FEATURE_DTYPES,
        sort_by_date=True)
    preprocessed_ds = preprocess_fn(ds)
    element = next(iter(preprocessed_ds))

    # BOS is len(vocab)+2, EOS is len(vocab)+3, pad is 0, OOV is len(vocab)+1
    self.assertAllEqual(
        self.evaluate(element[0]),
        np.array([[4, 3, 5, 0, 0, 0], [4, 1, 5, 0, 0, 0], [4, 1, 2, 3, 5, 0]]))

  def test_preprocess_fn_returns_correct_sequence_sort_by_date_max_elements(
      self):
    ds = tf.data.Dataset.from_tensor_slices(MULTI_EXAMPLE_TEST_DATA)
    preprocess_fn = stackoverflow_dataset.create_preprocess_fn(
        client_batch_size=32,
        max_sequence_length=6,
        max_elements_per_client=2,
        vocab=['one', 'must'],
        num_oov_buckets=1,
        feature_dtypes=FEATURE_DTYPES,
        sort_by_date=True)
    preprocessed_ds = preprocess_fn(ds)
    element = next(iter(preprocessed_ds))

    # BOS is len(vocab)+2, EOS is len(vocab)+3, pad is 0, OOV is len(vocab)+1.
    # Last example is not included here due to `max_elements_per_client`.
    self.assertAllEqual(
        self.evaluate(element[0]),
        np.array([[4, 3, 5, 0, 0, 0], [4, 1, 2, 3, 5, 0]]))

  def test_preprocess_fn_returns_correct_sequence_sort_by_date_single_example(
      self):
    ds = tf.data.Dataset.from_tensor_slices(SINGLE_EXAMPLE_TEST_DATA)
    preprocess_fn = stackoverflow_dataset.create_preprocess_fn(
        client_batch_size=32,
        max_sequence_length=6,
        max_elements_per_client=100,
        vocab=['one', 'must'],
        num_oov_buckets=1,
        feature_dtypes=FEATURE_DTYPES,
        sort_by_date=True)
    preprocessed_ds = preprocess_fn(ds)
    element = next(iter(preprocessed_ds))

    # BOS is len(vocab)+2, EOS is len(vocab)+3, pad is 0, OOV is len(vocab)+1
    self.assertAllEqual(
        self.evaluate(element[0]), np.array([[4, 1, 2, 3, 5, 0]]))


if __name__ == '__main__':
  tf.test.main()
