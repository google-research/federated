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
import itertools
from absl.testing import parameterized

import tensorflow as tf
import tensorflow_federated as tff

from dp_visual_embeddings.tasks import task_data


def create_dataset_fn(client_id='5'):
  num_examples = tf.strings.to_number(client_id, out_type=tf.int64) + 1
  return tf.data.Dataset.range(num_examples)


def create_client_data(num_clients):
  client_ids = [str(x) for x in range(num_clients)]
  return tff.simulation.datasets.ClientData.from_clients_and_tf_fn(
      client_ids, create_dataset_fn)


class GetElementSpecTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('federated_no_preprocess', 'Federated', False),
      ('federated_preprocess', 'Federated', True),
      ('centralized_no_preprocess', 'Centralized', False),
      ('centralized_preprocess', 'Centralized', True),
  )
  def test_get_element_spec(self, dataset_type, preprocess):
    data = create_client_data(10)
    if dataset_type == 'Centralized':
      data = data.create_tf_dataset_from_all_clients()

    def convert_int64_to_int32(dataset):
      return dataset.map(lambda x: tf.cast(x, tf.int32))

    if preprocess:
      preprocess_fn = convert_int64_to_int32
      expected_type = tf.TensorSpec(shape=(), dtype=tf.int32, name=None)
    else:
      preprocess_fn = None
      expected_type = tf.TensorSpec(shape=(), dtype=tf.int64, name=None)

    actual_type = task_data._get_element_spec(data, preprocess_fn)
    self.assertEqual(actual_type, expected_type)


class EmbeddingTaskDatasetsTest(tf.test.TestCase, parameterized.TestCase):

  def test_raises_when_train_and_test_types_are_different_no_preprocessing(
      self):
    train_data = create_client_data(10)
    test_data = tf.data.Dataset.range(10, output_type=tf.int32)
    with self.assertRaisesRegex(
        ValueError,
        'train and test element structures after preprocessing must be equal'):
      task_data.EmbeddingTaskDatasets(
          train_data=train_data,
          validation_data=train_data,
          test_data=test_data)

  def test_raises_when_train_and_test_types_are_different_with_train_preprocessing(
      self):
    train_data = create_client_data(10)
    test_data = tf.data.Dataset.range(10)
    train_preprocess_fn = lambda x: x.map(lambda y: tf.cast(y, dtype=tf.int32))
    with self.assertRaisesRegex(
        ValueError,
        'train and test element structures after preprocessing must be equal'):
      task_data.EmbeddingTaskDatasets(
          train_data=train_data,
          validation_data=train_data,
          train_preprocess_fn=train_preprocess_fn,
          test_data=test_data)

  def test_raises_when_train_and_test_types_are_different_with_eval_preprocessing(
      self):
    train_data = create_client_data(10)
    test_data = tf.data.Dataset.range(10)
    eval_preprocess_fn = lambda x: x.map(lambda y: tf.cast(y, dtype=tf.int32))
    with self.assertRaisesRegex(
        ValueError,
        'train and test element structures after preprocessing must be equal'):
      task_data.EmbeddingTaskDatasets(
          train_data=train_data,
          validation_data=train_data,
          test_preprocess_fn=eval_preprocess_fn,
          test_data=test_data)

  def test_raises_when_test_and_validation_types_are_different(self):
    train_data = create_client_data(10)
    test_data = tf.data.Dataset.range(10, output_type=tf.int64)
    validation_data = tf.data.Dataset.range(10, output_type=tf.int32)
    with self.assertRaisesRegex(
        ValueError,
        'validation and test element structures after preprocessing must be '
        'equal'):
      task_data.EmbeddingTaskDatasets(
          train_data=train_data,
          test_data=test_data,
          validation_data=validation_data)

  @parameterized.named_parameters(('dync1', 1), ('dync2', 2))
  def test_constructs_without_eval_preprocess_fn(self, dynamic_clients):
    preprocess_fn = lambda x: x.map(lambda y: 2 * y)
    test_task_data = task_data.EmbeddingTaskDatasets(
        train_data=create_client_data(10),
        train_preprocess_fn=preprocess_fn,
        validation_data=create_client_data(5),
        test_data=create_client_data(2),
        dynamic_clients=dynamic_clients)
    train_preprocess_fn = test_task_data.train_preprocess_fn
    example_dataset = train_preprocess_fn(tf.data.Dataset.range(20))
    for i, x in enumerate(example_dataset):
      self.assertEqual(2 * i, x.numpy())

  @parameterized.named_parameters(('dync1', 1), ('dync2', 2))
  def test_constructs_with_test_preprocess_fn(self, dynamic_clients):
    train_preprocess_fn = lambda x: x.map(lambda y: 2 * y)
    eval_preprocess_fn = lambda x: x.map(lambda y: 3 * y)
    test_task_data = task_data.EmbeddingTaskDatasets(
        train_data=create_client_data(10),
        train_preprocess_fn=train_preprocess_fn,
        validation_data=create_client_data(5),
        validation_preprocess_fn=eval_preprocess_fn,
        test_data=create_client_data(2),
        test_preprocess_fn=eval_preprocess_fn,
        dynamic_clients=dynamic_clients)
    example_dataset = test_task_data.test_preprocess_fn(
        tf.data.Dataset.range(20))
    for i, x in enumerate(example_dataset):
      self.assertEqual(3 * i, x.numpy())

  def test_sample_train_clients_returns_train_datasets(self):
    train_data = create_client_data(10)
    test_task_data = task_data.EmbeddingTaskDatasets(
        train_data=train_data,
        validation_data=train_data,
        test_data=create_client_data(2))
    all_client_datasets = [
        train_data.create_tf_dataset_for_client(x)
        for x in train_data.client_ids
    ]
    all_client_datasets_as_lists = [
        list(ds.as_numpy_iterator()) for ds in all_client_datasets
    ]
    sampled_client_datasets = test_task_data._sample_train_clients(
        num_clients=3)
    for ds in sampled_client_datasets:
      ds_as_list = list(ds.as_numpy_iterator())
      self.assertIn(ds_as_list, all_client_datasets_as_lists)
    sampled_client_datasets = test_task_data._sample_validation_clients(
        num_clients=3)
    for ds in sampled_client_datasets:
      ds_as_list = list(ds.as_numpy_iterator())
      self.assertIn(ds_as_list, all_client_datasets_as_lists)

  def test_sample_train_clients_returns_preprocessed_train_datasets(self):
    train_preprocess_fn = lambda x: x.map(lambda y: 2 * y)
    eval_preprocess_fn = lambda x: x.map(lambda y: 5 * y)
    train_data = create_client_data(10)
    test_task_data = task_data.EmbeddingTaskDatasets(
        train_data=train_data,
        train_preprocess_fn=train_preprocess_fn,
        validation_data=train_data,
        validation_preprocess_fn=eval_preprocess_fn,
        test_data=create_client_data(2),
        test_preprocess_fn=eval_preprocess_fn)
    preprocess_train_data = train_data.preprocess(train_preprocess_fn)
    all_client_datasets = [
        preprocess_train_data.create_tf_dataset_for_client(x)
        for x in preprocess_train_data.client_ids
    ]
    all_client_datasets_as_lists = [
        list(ds.as_numpy_iterator()) for ds in all_client_datasets
    ]
    sampled_client_datasets = test_task_data._sample_train_clients(
        num_clients=5)
    for ds in sampled_client_datasets:
      ds_as_list = list(ds.as_numpy_iterator())
      self.assertIn(ds_as_list, all_client_datasets_as_lists)

  def test_sample_validation_clients_returns_preprocessed_validation_datasets(
      self):
    train_preprocess_fn = lambda x: x.map(lambda y: 2 * y)
    eval_preprocess_fn = lambda x: x.map(lambda y: 5 * y)
    train_data = create_client_data(10)
    test_task_data = task_data.EmbeddingTaskDatasets(
        train_data=train_data,
        train_preprocess_fn=train_preprocess_fn,
        validation_data=train_data,
        validation_preprocess_fn=eval_preprocess_fn,
        test_data=create_client_data(2),
        test_preprocess_fn=eval_preprocess_fn)
    preprocess_validation_data = train_data.preprocess(eval_preprocess_fn)
    all_client_datasets = [
        preprocess_validation_data.create_tf_dataset_for_client(x)
        for x in preprocess_validation_data.client_ids
    ]
    all_client_datasets_as_lists = [
        list(ds.as_numpy_iterator()) for ds in all_client_datasets
    ]
    sampled_client_datasets = test_task_data._sample_validation_clients(
        num_clients=5)
    for ds in sampled_client_datasets:
      ds_as_list = list(ds.as_numpy_iterator())
      self.assertIn(ds_as_list, all_client_datasets_as_lists)

  def test_sample_train_clients_dataset_computation_all(self):
    train_preprocess_fn = lambda x: x.map(lambda y: 2 * y)
    eval_preprocess_fn = lambda x: x.map(lambda y: 5 * y)
    train_data = create_client_data(10)
    test_task_data = task_data.EmbeddingTaskDatasets(
        train_data=train_data,
        train_preprocess_fn=train_preprocess_fn,
        validation_data=train_data,
        validation_preprocess_fn=eval_preprocess_fn,
        test_data=create_client_data(2),
        test_preprocess_fn=eval_preprocess_fn)
    all_client_datasets = [
        test_task_data.train_dataset_computation(x)
        for x in train_data.client_ids
    ]
    all_client_datasets_as_lists = [
        list(ds.as_numpy_iterator()) for ds in all_client_datasets
    ]
    sampled_client_datasets = test_task_data._sample_train_clients(
        num_clients=5)
    for ds in sampled_client_datasets:
      ds_as_list = list(ds.as_numpy_iterator())
      self.assertIn(ds_as_list, all_client_datasets_as_lists)

  @parameterized.named_parameters(('dync1', 1), ('dync2', 2))
  def test_sample_train_clients_dataset_computation_sample(
      self, dynamic_clients):
    train_preprocess_fn = lambda x: x.map(lambda y: 2 * y)
    eval_preprocess_fn = lambda x: x.map(lambda y: 5 * y)
    train_data = create_client_data(10)
    test_task_data = task_data.EmbeddingTaskDatasets(
        train_data=train_data,
        train_preprocess_fn=train_preprocess_fn,
        validation_data=train_data,
        validation_preprocess_fn=eval_preprocess_fn,
        test_data=create_client_data(2),
        test_preprocess_fn=eval_preprocess_fn,
        dynamic_clients=dynamic_clients)
    preprocess_train_data = train_data.preprocess(train_preprocess_fn)
    sampled_client_ids = test_task_data.sample_train_client_ids(num_clients=5)
    for cid in sampled_client_ids:
      ds = test_task_data.train_dataset_computation(cid)
      ds_as_list = list(ds.as_numpy_iterator())
      cids = cid.split(task_data._CLIENT_ID_JOINER)
      datasets = [
          preprocess_train_data.create_tf_dataset_for_client(c) for c in cids
      ]
      datasets_as_iter = [ds.as_numpy_iterator() for ds in datasets]
      expected = list(itertools.chain.from_iterable(datasets_as_iter))
      self.assertCountEqual(ds_as_list, expected)

  def test_sample_validation_clients_dataset_computation(self):
    train_preprocess_fn = lambda x: x.map(lambda y: 2 * y)
    eval_preprocess_fn = lambda x: x.map(lambda y: 5 * y)
    train_data = create_client_data(10)
    test_task_data = task_data.EmbeddingTaskDatasets(
        train_data=train_data,
        train_preprocess_fn=train_preprocess_fn,
        validation_data=train_data,
        validation_preprocess_fn=eval_preprocess_fn,
        test_data=create_client_data(2),
        test_preprocess_fn=eval_preprocess_fn)
    all_client_datasets = [
        test_task_data.validation_dataset_computation(x)
        for x in train_data.client_ids
    ]
    all_client_datasets_as_lists = [
        list(ds.as_numpy_iterator()) for ds in all_client_datasets
    ]
    sampled_client_datasets = test_task_data._sample_validation_clients(
        num_clients=5)
    for ds in sampled_client_datasets:
      ds_as_list = list(ds.as_numpy_iterator())
      self.assertIn(ds_as_list, all_client_datasets_as_lists)

  @parameterized.named_parameters(('dync1', 1), ('dync2', 2))
  def test_sample_validation_clients_dataset_computation_sample(
      self, dynamic_clients):
    train_preprocess_fn = lambda x: x.map(lambda y: 2 * y)
    eval_preprocess_fn = lambda x: x.map(lambda y: 5 * y)
    train_data = create_client_data(10)
    test_task_data = task_data.EmbeddingTaskDatasets(
        train_data=train_data,
        train_preprocess_fn=train_preprocess_fn,
        validation_data=train_data,
        validation_preprocess_fn=eval_preprocess_fn,
        test_data=create_client_data(2),
        test_preprocess_fn=eval_preprocess_fn,
        dynamic_clients=dynamic_clients)
    preprocess_validation_data = train_data.preprocess(eval_preprocess_fn)
    sampled_client_ids = test_task_data.sample_train_client_ids(num_clients=5)
    for cid in sampled_client_ids:
      ds = test_task_data.validation_dataset_computation(cid)
      ds_as_list = list(ds.as_numpy_iterator())
      cids = cid.split(task_data._CLIENT_ID_JOINER)
      datasets = [
          preprocess_validation_data.create_tf_dataset_for_client(c)
          for c in cids
      ]
      datasets_as_iter = [ds.as_numpy_iterator() for ds in datasets]
      expected = list(itertools.chain.from_iterable(datasets_as_iter))
      self.assertCountEqual(ds_as_list, expected)

  @parameterized.named_parameters(('dync1', 1), ('dync2', 2))
  def test_sample_train_clients_random_seed(self, dynamic_clients):
    train_data = create_client_data(100)
    test_task_data = task_data.EmbeddingTaskDatasets(
        train_data=train_data,
        validation_data=train_data,
        test_data=create_client_data(2),
        dynamic_clients=dynamic_clients)
    client_datasets1 = test_task_data._sample_train_clients(
        num_clients=5, random_seed=0)
    data1 = [list(ds.as_numpy_iterator()) for ds in client_datasets1]
    client_datasets2 = test_task_data._sample_train_clients(
        num_clients=5, random_seed=0)
    data2 = [list(ds.as_numpy_iterator()) for ds in client_datasets2]
    client_datasets3 = test_task_data._sample_train_clients(
        num_clients=5, random_seed=1)
    data3 = [list(ds.as_numpy_iterator()) for ds in client_datasets3]

    self.assertAllEqual(data1, data2)
    self.assertNotAllEqual(data1, data3)

  @parameterized.named_parameters(('dync1', 1), ('dync2', 2), ('dync5', 5))
  def test_dynamic_clients_dataset_computation_size(self, dynamic_clients):
    train_preprocess_fn = lambda x: x.map(lambda y: 2 * y)
    eval_preprocess_fn = lambda x: x.map(lambda y: 5 * y)
    train_data = create_client_data(20)
    test_task_data = task_data.EmbeddingTaskDatasets(
        train_data=train_data,
        train_preprocess_fn=train_preprocess_fn,
        validation_data=train_data,
        validation_preprocess_fn=eval_preprocess_fn,
        test_data=create_client_data(2),
        test_preprocess_fn=eval_preprocess_fn,
        dynamic_clients=dynamic_clients)
    sampled_client_ids = test_task_data.sample_train_client_ids(num_clients=3)
    for cid in sampled_client_ids:
      int_cids = [int(c) for c in cid.split(task_data._CLIENT_ID_JOINER)]
      self.assertLen(set(int_cids), len(int_cids))
      sum_size = sum(int_cids) + len(int_cids)
      ds = test_task_data.train_dataset_computation(cid)
      # The clients are created by `range` before merging
      self.assertLen(list(ds.as_numpy_iterator()), sum_size)
    sampled_client_ids = test_task_data.sample_validation_client_ids(
        num_clients=3)
    for cid in sampled_client_ids:
      int_cids = [int(c) for c in cid.split(task_data._CLIENT_ID_JOINER)]
      self.assertLen(set(int_cids), len(int_cids))
      sum_size = sum(int_cids) + len(int_cids)
      ds = test_task_data.validation_dataset_computation(cid)
      # The clients are created by `range` before merging
      self.assertLen(list(ds.as_numpy_iterator()), sum_size)

  def test_create_centralized_train_from_client_data(self):
    train_client_data = create_client_data(5)
    test_task_data = task_data.EmbeddingTaskDatasets(
        train_data=train_client_data,
        validation_data=train_client_data,
        test_data=create_client_data(3))
    train_data = test_task_data.get_centralized_train_data()
    self.assertCountEqual(
        list(train_data.as_numpy_iterator()),
        [0] * 5 + [1] * 4 + [2] * 3 + [3] * 2 + [4])

  def test_create_centralized_validation_from_client_data(self):
    train_client_data = create_client_data(5)
    val_client_data = create_client_data(4)
    test_task_data = task_data.EmbeddingTaskDatasets(
        train_data=train_client_data,
        validation_data=val_client_data,
        test_data=create_client_data(3))
    val_data = test_task_data.get_centralized_validation_data()
    self.assertCountEqual(
        list(val_data.as_numpy_iterator()), [0] * 4 + [1] * 3 + [2] * 2 + [3])

  def test_create_centralized_test_from_client_data(self):
    train_data = create_client_data(100)
    test_task_data = task_data.EmbeddingTaskDatasets(
        train_data=train_data,
        validation_data=train_data,
        test_data=create_client_data(3))
    test_data = test_task_data.get_centralized_test_data()
    self.assertSameElements(
        list(test_data.as_numpy_iterator()), [0, 0, 0, 1, 1, 2])

  def test_create_centralized_test_from_client_data_with_eval_preprocess(self):
    eval_preprocess_fn = lambda x: x.map(lambda y: 3 * y)
    train_data = create_client_data(100)
    test_task_data = task_data.EmbeddingTaskDatasets(
        train_data=train_data,
        validation_data=train_data,
        validation_preprocess_fn=eval_preprocess_fn,
        test_data=create_client_data(3),
        test_preprocess_fn=eval_preprocess_fn)
    test_data = test_task_data.get_centralized_test_data()
    self.assertSameElements(
        list(test_data.as_numpy_iterator()), [0, 0, 0, 3, 3, 6])

  def test_create_centralized_test_from_dataset(self):
    train_data = create_client_data(100)
    test_task_data = task_data.EmbeddingTaskDatasets(
        train_data=train_data,
        validation_data=train_data,
        test_data=tf.data.Dataset.range(7))
    test_data = test_task_data.get_centralized_test_data()
    self.assertSameElements(list(test_data.as_numpy_iterator()), list(range(7)))

  def test_create_centralized_test_from_dataset_with_eval_preprocess(self):
    eval_preprocess_fn = lambda x: x.map(lambda y: 3 * y)
    train_data = create_client_data(100)
    test_task_data = task_data.EmbeddingTaskDatasets(
        train_data=train_data,
        validation_data=train_data,
        validation_preprocess_fn=eval_preprocess_fn,
        test_data=tf.data.Dataset.range(7),
        test_preprocess_fn=eval_preprocess_fn)
    test_data = test_task_data.get_centralized_test_data()
    expected_data = [3 * a for a in range(7)]
    self.assertSameElements(list(test_data.as_numpy_iterator()), expected_data)

  @parameterized.named_parameters(
      ('num_clients1', 1),
      ('num_clients2', 4),
      ('num_clients3', 10),
  )
  def test_record_train_dataset_info(self, num_clients):
    train_data = create_client_data(num_clients)
    test_task_data = task_data.EmbeddingTaskDatasets(
        train_data=train_data,
        validation_data=train_data,
        test_data=create_client_data(2))
    actual_train_info = test_task_data._record_dataset_information()['train']
    expected_train_info = ['Train', 'Federated', str(num_clients)]
    self.assertEqual(actual_train_info, expected_train_info)

  @parameterized.named_parameters(
      ('test_config1', 'Federated', 4),
      ('test_config2', 'Federated', 15),
      ('test_config3', 'Centralized', 'N/A'),
  )
  def test_record_test_dataset_info(self, test_dataset_type, num_clients):
    if test_dataset_type == 'Federated':
      test_data = create_client_data(num_clients)
    else:
      test_data = tf.data.Dataset.range(5)
    train_data = create_client_data(1)
    test_task_data = task_data.EmbeddingTaskDatasets(
        train_data=train_data, validation_data=train_data, test_data=test_data)
    actual_test_info = test_task_data._record_dataset_information()['test']
    expected_test_info = ['Test', test_dataset_type, str(num_clients)]
    self.assertEqual(actual_test_info, expected_test_info)

  @parameterized.named_parameters(
      ('is_not_none', lambda x: x, True),
      ('is_none', None, False),
  )
  def test_summary_train_preprocess_fn(self, train_preprocess_fn, is_not_none):
    train_data = create_client_data(10)
    test_task_data = task_data.EmbeddingTaskDatasets(
        train_data=train_data,
        validation_data=train_data,
        train_preprocess_fn=train_preprocess_fn,
        test_data=create_client_data(2))
    summary_list = []
    test_task_data.summary(print_fn=summary_list.append)
    expected_train_preprocess_summary = 'Train Preprocess Function: {}'.format(
        is_not_none)
    self.assertEqual(summary_list[6], expected_train_preprocess_summary)

  @parameterized.named_parameters(
      ('is_not_none', lambda x: x, True),
      ('is_none', None, False),
  )
  def test_summary_test_preprocess_fn(self, eval_preprocess_fn, is_not_none):
    train_data = create_client_data(10)
    test_task_data = task_data.EmbeddingTaskDatasets(
        train_data=train_data,
        validation_data=train_data,
        test_preprocess_fn=eval_preprocess_fn,
        test_data=create_client_data(2))
    summary_list = []
    test_task_data.summary(print_fn=summary_list.append)
    expected_eval_preprocess_summary = 'Test Preprocess Function: {}'.format(
        is_not_none)
    self.assertEqual(summary_list[8], expected_eval_preprocess_summary)

  @parameterized.named_parameters(
      ('is_not_none', lambda x: x, True),
      ('is_none', None, False),
  )
  def test_summary_validation_preprocess_fn(self, eval_preprocess_fn,
                                            is_not_none):
    train_data = create_client_data(10)
    test_task_data = task_data.EmbeddingTaskDatasets(
        train_data=train_data,
        validation_data=train_data,
        validation_preprocess_fn=eval_preprocess_fn,
        test_data=create_client_data(2))
    summary_list = []
    test_task_data.summary(print_fn=summary_list.append)
    expected_eval_preprocess_summary = 'Validation Preprocess Function: {}'.format(
        is_not_none)
    self.assertEqual(summary_list[7], expected_eval_preprocess_summary)

  @parameterized.named_parameters(
      ('config1', 'Federated'),
      ('config2', 'Centralized'),
  )
  def test_data_summary_header_is_constant(self, test_type):
    if test_type == 'Federated':
      test_data = create_client_data(5)
    else:
      test_data = tf.data.Dataset.range(5)

    test_task_data = task_data.EmbeddingTaskDatasets(
        train_data=create_client_data(10),
        test_data=test_data,
        validation_data=create_client_data(4))
    data_summary = []
    test_task_data.summary(print_fn=data_summary.append)
    actual_header_values = data_summary[0].split()
    expected_header_values = [
        'Split', '|Dataset', 'Type', '|Number', 'of', 'Clients', '|'
    ]
    self.assertEqual(actual_header_values, expected_header_values)

  @parameterized.named_parameters(
      ('num_clients1', 1),
      ('num_clients2', 9),
      ('num_clients3', 7),
  )
  def test_summary_gives_correct_train_information(self, num_clients):
    train_data = create_client_data(num_clients)
    test_data = tf.data.Dataset.range(5)
    test_task_data = task_data.EmbeddingTaskDatasets(
        train_data=train_data, validation_data=train_data, test_data=test_data)
    data_summary = []
    test_task_data.summary(print_fn=data_summary.append)
    actual_train_summary = data_summary[2].split()
    expected_train_summary = [
        'Train', '|Federated', '|{}'.format(num_clients), '|'
    ]
    self.assertEqual(actual_train_summary, expected_train_summary)

  @parameterized.named_parameters(
      ('test_config1', 'Federated', 4),
      ('test_config2', 'Federated', 15),
      ('test_config3', 'Centralized', 'N/A'),
  )
  def test_summary_gives_correct_test_information(self, test_type, num_clients):
    train_data = create_client_data(5)
    if test_type == 'Federated':
      test_data = create_client_data(num_clients)
    else:
      test_data = tf.data.Dataset.range(5)
    test_task_data = task_data.EmbeddingTaskDatasets(
        train_data=train_data, validation_data=train_data, test_data=test_data)
    data_summary = []
    test_task_data.summary(print_fn=data_summary.append)
    actual_test_summary = data_summary[3].split()
    expected_test_summary = [
        'Test', '|{}'.format(test_type), '|{}'.format(num_clients), '|'
    ]
    self.assertEqual(actual_test_summary, expected_test_summary)

  @parameterized.named_parameters(
      ('validation_config1', 5),
      ('validation_config2', 23),
  )
  def test_summary_gives_correct_validation_information(self, num_clients):
    validation_data = create_client_data(num_clients)
    test_task_data = task_data.EmbeddingTaskDatasets(
        train_data=create_client_data(1),
        test_data=create_client_data(1),
        validation_data=validation_data)
    data_summary = []
    test_task_data.summary(print_fn=data_summary.append)
    actual_validation_summary = data_summary[4].split()
    expected_validation_summary = [
        'Validation', '|Federated', '|{}'.format(num_clients), '|'
    ]
    self.assertEqual(actual_validation_summary, expected_validation_summary)

  def test_summary_table_structure_with_validation(self):
    train_data = create_client_data(1)
    test_task_data = task_data.EmbeddingTaskDatasets(
        train_data=train_data, test_data=train_data, validation_data=train_data)
    data_summary = []
    test_task_data.summary(print_fn=data_summary.append)
    self.assertLen(data_summary, 9)

    table_len = len(data_summary[0])
    self.assertEqual(data_summary[1], '=' * table_len)
    for i in range(2, 5):
      self.assertLen(data_summary[i], table_len)
    self.assertEqual(data_summary[5], '_' * table_len)

  def test_raise_sample_clients_from_centralized(self):
    train_data = create_dataset_fn()
    test_task_data = task_data.EmbeddingTaskDatasets(
        train_data=train_data, test_data=train_data, validation_data=train_data)
    with self.assertRaisesRegex(TypeError,
                                'Cannot sample clients from train data'):
      test_task_data._sample_train_clients(1)
    with self.assertRaisesRegex(TypeError,
                                'Cannot sample clients from validation data'):
      test_task_data._sample_validation_clients(1)

  def test_create_centralized_train_with_preprocess(self):
    train_preprocess_fn = lambda x: x.map(lambda y: 3 * y)
    train_data = create_dataset_fn()
    test_task_data = task_data.EmbeddingTaskDatasets(
        train_data=train_data,
        validation_data=train_data,
        train_preprocess_fn=train_preprocess_fn,
        test_data=create_client_data(3))
    train_data = test_task_data.get_centralized_train_data()
    val_data = test_task_data.get_centralized_validation_data()
    test_data = test_task_data.get_centralized_test_data()
    self.assertSameElements(
        list(train_data.as_numpy_iterator()), list(range(0, 18, 3)))
    self.assertSameElements(
        list(val_data.as_numpy_iterator()), list(range(6)))
    self.assertSameElements(
        list(test_data.as_numpy_iterator()), [0, 0, 0, 1, 1, 2])

  def test_create_centralized_validation_with_preprocess(self):
    eval_preprocess_fn = lambda x: x.map(lambda y: 3 * y)
    train_data = create_dataset_fn()
    test_task_data = task_data.EmbeddingTaskDatasets(
        train_data=train_data,
        validation_data=train_data,
        validation_preprocess_fn=eval_preprocess_fn,
        test_data=create_client_data(3))
    train_data = test_task_data.get_centralized_train_data()
    val_data = test_task_data.get_centralized_validation_data()
    test_data = test_task_data.get_centralized_test_data()
    self.assertSameElements(
        list(train_data.as_numpy_iterator()), list(range(6)))
    self.assertSameElements(
        list(val_data.as_numpy_iterator()), list(range(0, 18, 3)))
    self.assertSameElements(
        list(test_data.as_numpy_iterator()), [0, 0, 0, 1, 1, 2])

if __name__ == '__main__':
  tf.test.main()
