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
"""Tests for emnist_task."""

import collections
from absl.testing import parameterized

import tensorflow as tf
import tensorflow_federated as tff

from dp_visual_embeddings import metrics
from dp_visual_embeddings.tasks import emnist_task


def _get_preprocess_spec():
  return tff.simulation.baselines.ClientSpec(
      num_epochs=1, batch_size=2, max_elements=3, shuffle_buffer_size=4)


def _get_expected_data_and_model_spec(image_height_and_width=28):
  return collections.OrderedDict(
      x=collections.OrderedDict(
          images=tf.TensorSpec(
              shape=(None, image_height_and_width, image_height_and_width, 1),
              dtype=tf.float32,
              name=None)),
      y=collections.OrderedDict(
          identity_names=tf.TensorSpec(
              shape=(None,), dtype=tf.string, name=None),
          identity_indices=tf.TensorSpec(
              shape=(None,), dtype=tf.int32, name=None)))


def _get_tff_task_for_testing(dynamic_clients=1):
  preprocess_spec = _get_preprocess_spec()
  task = emnist_task.get_emnist_embedding_task(
      preprocess_spec, preprocess_spec, dynamic_clients=dynamic_clients)
  return task


def _build_fake_inputs(batch_size=2, image_height_and_width=28, num_labels=3):
  images = tf.zeros(
      [batch_size, image_height_and_width, image_height_and_width, 1])
  identity_indices = tf.random.uniform([batch_size],
                                       maxval=num_labels,
                                       dtype=tf.int32,
                                       seed=42)
  return collections.OrderedDict(
      x=collections.OrderedDict(images=images),
      y=collections.OrderedDict(identity_indices=identity_indices))


class DatasetLibTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    train_client_data, test_client_data = tff.simulation.datasets.emnist.load_data(
        only_digits=False)
    self.train_client_data = train_client_data
    self.test_client_data = test_client_data

  def test_get_emnist_client_data(self):
    train_client_data = self.train_client_data
    train_client_dataset = train_client_data.create_tf_dataset_for_client(
        train_client_data.client_ids[0])
    self.assertIn(emnist_task._IMAGE_KEY, train_client_dataset.element_spec)
    self.assertIn(emnist_task._IDENTITY_KEY, train_client_dataset.element_spec)

    test_client_dataset = self.test_client_data.create_tf_dataset_for_client(
        self.test_client_data.client_ids[0])
    self.assertIn(emnist_task._IMAGE_KEY, test_client_dataset.element_spec)
    self.assertIn(emnist_task._IDENTITY_KEY, test_client_dataset.element_spec)


class TaskLibTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    preprocess_spec = _get_preprocess_spec()
    self.default_task = emnist_task.get_emnist_embedding_task(
        preprocess_spec, preprocess_spec)

  @parameterized.named_parameters(('dyn1', 1), ('dyn3', 3))
  def test_get_emnist_embedding_task_datasets(self, dynamic_clients):
    preprocess_spec = _get_preprocess_spec()
    datasets = emnist_task._get_emnist_embedding_task_datasets(
        preprocess_spec, preprocess_spec, dynamic_clients=dynamic_clients)
    self.assertLen(datasets.train_data.client_ids, 6800)
    self.assertIsNotNone(datasets.train_preprocess_fn)
    self.assertIsNotNone(datasets.validation_preprocess_fn)
    self.assertIsNotNone(datasets.test_preprocess_fn)
    self.assertEqual(_get_expected_data_and_model_spec(),
                     datasets.element_type_structure)

  def test_get_emnist_embedding_task(self):
    task = self.default_task
    task_model = task.federated_model_fn()
    self.assertIsInstance(task_model, tff.learning.Model)
    self.assertEqual(_get_expected_data_and_model_spec(), task_model.input_spec)

  def test_get_emnist_embedding_task_image_scale(self):
    task = self.default_task
    example_clients = task.datasets._sample_train_clients(
        num_clients=1, random_seed=42)
    min_val, max_val = 0, 0
    for batch in example_clients[0]:
      min_val = tf.minimum(min_val, tf.reduce_min(batch['x']['images']))
      max_val = tf.maximum(max_val, tf.reduce_max(batch['x']['images']))
    self.assertNear(min_val, -1, 0.1)
    self.assertNear(max_val, 1, 0.1)

  def test_get_inference_model(self):
    task = self.default_task
    model = task.inference_model

    images = tf.zeros([2, 28, 28, 1])
    embeddings = model(images, training=False)
    self.assertSequenceEqual(embeddings.numpy().shape, (2, 128))


class TFFModelTest(tf.test.TestCase, parameterized.TestCase):

  def _test_model_forward_pass(self, tff_model, num_labels, dynamic_clients):
    del dynamic_clients
    batch_size = 4
    # Input data is all zero, so the corresponding output is also zero.
    batch_input = _build_fake_inputs(
        batch_size=batch_size, num_labels=num_labels)

    batch_output = tff_model.forward_pass(batch_input)
    self.assertAllClose(
        tf.zeros([batch_size, emnist_task._NUM_LABELS]),
        batch_output.predictions[0])
    self.assertAllClose(
        tf.zeros([batch_size, emnist_task._EMBEDDING_DIM_SIZE]),
        batch_output.predictions[1])
    self.assertEqual(batch_size, batch_output.num_examples)

  @parameterized.named_parameters(('static', 3, 1), ('dync', 1, 3))
  def test_forward_pass_of_tff_model(self, num_labels, dynamic_clients):
    task = _get_tff_task_for_testing(dynamic_clients=dynamic_clients)
    self._test_model_forward_pass(
        task.federated_model_fn(),
        num_labels=num_labels,
        dynamic_clients=dynamic_clients)

  @parameterized.named_parameters(('statc', 3, 1), ('dync', 1, 3))
  def test_forward_pass_of_embedding_model(self, num_labels, dynamic_clients):
    task = _get_tff_task_for_testing(dynamic_clients=dynamic_clients)
    self._test_model_forward_pass(
        task.embedding_model_fn(),
        num_labels=num_labels,
        dynamic_clients=dynamic_clients)

  def _test_report_local_unfinalized_metrics(self, tff_model, num_labels):
    initial_outputs = tff_model.report_local_unfinalized_metrics()
    self.assertAllEqual(
        [0.0, 0.], initial_outputs['embedding_categorical_accuracy_metric'])
    self.assertAllEqual([[0] * metrics._DEFAULT_NUM_THRESHOLDS] * 4,
                        initial_outputs['recall_at_far_0p1'])
    self.assertAllEqual([[0] * metrics._DEFAULT_NUM_THRESHOLDS] * 4,
                        initial_outputs['recall_at_far_1e-3'])
    self.assertAllEqual([0., 0.], initial_outputs['loss'])
    self.assertAllEqual([0], initial_outputs['num_batches'])
    self.assertAllEqual([0], initial_outputs['num_examples'])
    self.assertLen(initial_outputs, 6)

    batch_size = 3
    batch_input = _build_fake_inputs(
        batch_size=batch_size, num_labels=num_labels)
    tff_model.forward_pass(batch_input)

    first_batch_outputs = tff_model.report_local_unfinalized_metrics()
    self.assertAllEqual(
        [0.0, batch_size],
        first_batch_outputs['embedding_categorical_accuracy_metric'])
    self.assertAllEqual(
        [[metrics._DEFAULT_NUM_THRESHOLDS]] * 4,
        [t.shape for t in first_batch_outputs['recall_at_far_0p1']])
    self.assertAllEqual(
        [[metrics._DEFAULT_NUM_THRESHOLDS]] * 4,
        [t.shape for t in first_batch_outputs['recall_at_far_1e-3']])
    self.assertAllEqual([1], first_batch_outputs['num_batches'])
    self.assertAllEqual([3], first_batch_outputs['num_examples'])
    self.assertLen(first_batch_outputs, 6)

  def test_report_local_unfinalized_metrics_of_tff_model(self):
    task = _get_tff_task_for_testing()
    self._test_report_local_unfinalized_metrics(
        task.federated_model_fn(), num_labels=3)

  def test_report_local_unfinalized_metrics_of_embedding_model(self):
    task = _get_tff_task_for_testing()
    self._test_report_local_unfinalized_metrics(
        task.embedding_model_fn(), num_labels=3)


if __name__ == '__main__':
  tff.google.backends.native.set_local_cpp_execution_context()
  tf.test.main()
