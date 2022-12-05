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
"""Tests for trainer."""
import functools
import os
from unittest import mock
from absl.testing import parameterized

import tensorflow as tf
import tensorflow_federated as tff

from dp_visual_embeddings import trainer
from dp_visual_embeddings.models import build_model
from dp_visual_embeddings.tasks import emnist_task


def _get_emnist_test_task(trainable_conv=True):
  train_client_spec = tff.simulation.baselines.ClientSpec(
      num_epochs=4, batch_size=2, max_elements=4, shuffle_buffer_size=8)
  eval_client_spec = tff.simulation.baselines.ClientSpec(
      num_epochs=1, batch_size=4, max_elements=4, shuffle_buffer_size=1)
  return emnist_task.get_emnist_embedding_task(
      train_client_spec,
      eval_client_spec,
      model_backbone=build_model.ModelBackbone.MOBILESMALL,
      dynamic_clients=2,
      trainable_conv=trainable_conv)


class TrainerTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    # Create and save an inference model to be loaded as pretrained model.
    task = _get_emnist_test_task(trainable_conv=True)
    self.save_infer_model = task.inference_model
    rootpath = self.create_tempdir()
    self.save_infer_path = os.path.join(rootpath, 'infer')
    self.save_infer_model.save(self.save_infer_path)

  def test_get_process_types(self):
    self.assertListEqual(list(trainer.ProcessType), trainer.get_process_types())

  def test_configure_optimizers(self):
    server_optimizer_fn, client_optimizer_fn = trainer.configure_optimizers()
    self.assertTrue(callable(server_optimizer_fn))
    self.assertTrue(callable(client_optimizer_fn))
    self.assertIsInstance(server_optimizer_fn(), tf.keras.optimizers.SGD)
    self.assertIsInstance(client_optimizer_fn(), tf.keras.optimizers.SGD)

  def test_train_and_eval_fed_partial(self):
    task = _get_emnist_test_task()
    server_optimizer, client_optimizer, learning_rate_fn = (
        trainer.configure_client_scheduled_optimizers(
            client_learning_rate=0.01))
    aggregator = trainer.configure_aggregator()
    process_type = trainer.ProcessType.FEDPARTIAL
    process = trainer.build_train_process(
        process_type,
        aggregator=aggregator,
        task=task,
        server_optimizer=server_optimizer,
        client_optimizer=client_optimizer,
        client_learning_rate_fn=learning_rate_fn)
    eval_fn = trainer.build_eval_fn(
        process_type, train_process=process, task=task)
    export_fn = trainer.build_export_fn(
        process_type, task=task, train_process=process)
    export_fn = functools.partial(export_fn, export_dir=self.get_temp_dir())
    trainer.train_and_eval(
        task,
        train_process=process,
        evaluation_fn=eval_fn,
        export_fn=export_fn,
        total_rounds=3,
        clients_per_round=1,
        rounds_per_eval=2,
        metrics_managers=[tff.program.LoggingReleaseManager()])

  def test_train_and_eval_fed_partial_freeze_with_saved_inference(self):
    task = _get_emnist_test_task(trainable_conv=False)
    server_optimizer, client_optimizer, learning_rate_fn = (
        trainer.configure_client_scheduled_optimizers(
            client_learning_rate=0.01))
    aggregator = trainer.configure_aggregator()
    process_type = trainer.ProcessType.FEDPARTIAL
    process = trainer.build_train_process(
        process_type,
        task=task,
        aggregator=aggregator,
        server_optimizer=server_optimizer,
        client_optimizer=client_optimizer,
        client_learning_rate_fn=learning_rate_fn,
        pretrained_model_path=self.save_infer_path,
        pretrained_output_size=0,
    )
    eval_fn = trainer.build_eval_fn(
        process_type, train_process=process, task=task)
    export_fn = trainer.build_export_fn(
        process_type, task=task, train_process=process)
    export_fn = functools.partial(export_fn, export_dir=self.get_temp_dir())
    trainer.train_and_eval(
        task,
        train_process=process,
        evaluation_fn=eval_fn,
        export_fn=export_fn,
        total_rounds=2,
        clients_per_round=1,
        rounds_per_eval=1,
        metrics_managers=[tff.program.LoggingReleaseManager()])

  def test_train_and_eval_fed_partial_with_saved_inference(self):
    task = _get_emnist_test_task()
    server_optimizer, client_optimizer, learning_rate_fn = (
        trainer.configure_client_scheduled_optimizers(
            client_learning_rate=0.01))
    aggregator = trainer.configure_aggregator()
    process_type = trainer.ProcessType.FEDPARTIAL
    process = trainer.build_train_process(
        process_type,
        task=task,
        aggregator=aggregator,
        server_optimizer=server_optimizer,
        client_optimizer=client_optimizer,
        client_learning_rate_fn=learning_rate_fn,
        pretrained_model_path=self.save_infer_path,
        pretrained_output_size=0,
    )
    eval_fn = trainer.build_eval_fn(
        process_type, train_process=process, task=task)
    export_fn = trainer.build_export_fn(
        process_type, task=task, train_process=process)
    export_fn = functools.partial(export_fn, export_dir=self.get_temp_dir())
    state = trainer.train_and_eval(
        task,
        train_process=process,
        evaluation_fn=eval_fn,
        export_fn=export_fn,
        total_rounds=0,
        clients_per_round=1,
        rounds_per_eval=2,
        metrics_managers=[tff.program.LoggingReleaseManager()])
    state_model_weights = process.get_model_weights(state)
    state_model_variables = (
        state_model_weights.trainable + state_model_weights.non_trainable)
    self.assertAllClose(self.save_infer_model.weights, state_model_variables)

  def test_train_and_eval_fed_partial_reconst(self):
    task = _get_emnist_test_task()
    server_optimizer, client_optimizer, learning_rate_fn = (
        trainer.configure_client_scheduled_optimizers(
            client_learning_rate=0.01))
    aggregator = trainer.configure_aggregator()
    process_type = trainer.ProcessType.FEDPARTIAL
    process = trainer.build_train_process(
        process_type,
        aggregator=aggregator,
        task=task,
        server_optimizer=server_optimizer,
        client_optimizer=client_optimizer,
        client_learning_rate_fn=learning_rate_fn,
        reconst_iters=1)
    trainer.train_and_eval(
        task,
        train_process=process,
        evaluation_fn=None,
        export_fn=lambda x: None,
        total_rounds=3,
        clients_per_round=1,
        rounds_per_eval=2,
        metrics_managers=[tff.program.LoggingReleaseManager()])


class AggregatorTest(tf.test.TestCase, parameterized.TestCase):

  @mock.patch.object(
      tff.aggregators.DifferentiallyPrivateFactory,
      'gaussian_adaptive',
      wraps=tff.aggregators.DifferentiallyPrivateFactory.gaussian_adaptive)
  def test_adaptive_dpsgd_call(self, mock_method):
    task = _get_emnist_test_task()
    trainer.configure_aggregator(
        aggregator_type=trainer.AggregatorType.DPSGD,
        model_fn=task.embedding_model_fn,
        clip_norm=0.1,
        noise_multiplier=1e-3,
        report_goal=100,
        target_unclipped_quantile=0.5)
    mock_method.assert_called()

  @mock.patch.object(
      tff.aggregators.DifferentiallyPrivateFactory,
      'gaussian_fixed',
      wraps=tff.aggregators.DifferentiallyPrivateFactory.gaussian_fixed)
  def test_dpsgd_call(self, mock_method):
    task = _get_emnist_test_task()
    trainer.configure_aggregator(
        aggregator_type=trainer.AggregatorType.DPSGD,
        model_fn=task.embedding_model_fn,
        clip_norm=0.1,
        noise_multiplier=1e-3,
        report_goal=100,
        target_unclipped_quantile=None)
    mock_method.assert_called()

  @mock.patch.object(
      tff.aggregators.DifferentiallyPrivateFactory,
      'tree_aggregation',
      wraps=tff.aggregators.DifferentiallyPrivateFactory.tree_aggregation)
  def test_dpftrl_call(self, mock_method):
    task = _get_emnist_test_task()
    trainer.configure_aggregator(
        aggregator_type=trainer.AggregatorType.DPFTRL,
        model_fn=task.embedding_model_fn,
        clip_norm=0.1,
        noise_multiplier=1e-3,
        report_goal=100,
        target_unclipped_quantile=None)
    mock_method.assert_called()

  @mock.patch.object(
      tff.aggregators.robust,
      'clipping_factory',
      wraps=tff.aggregators.robust.clipping_factory)
  def test_clipping_call(self, mock_method):
    task = _get_emnist_test_task()
    trainer.configure_aggregator(
        aggregator_type=None,
        model_fn=task.embedding_model_fn,
        clip_norm=0.1,
        noise_multiplier=None,
        report_goal=100,
        target_unclipped_quantile=None)
    mock_method.assert_called()

  @parameterized.named_parameters(('dpsgd', trainer.AggregatorType.DPSGD),
                                  ('dpftrl', trainer.AggregatorType.DPFTRL))
  def test_clipping_raise(self, aggregator_type):
    task = _get_emnist_test_task()
    with self.assertRaisesRegex(ValueError, 'Clipping only'):
      trainer.configure_aggregator(
          aggregator_type=aggregator_type,
          model_fn=task.embedding_model_fn,
          clip_norm=0.1,
          noise_multiplier=None,
          report_goal=100,
          target_unclipped_quantile=None)

  @parameterized.named_parameters(
      ('adaptive_dpsgd', trainer.AggregatorType.DPSGD, 0.5, True),
      ('dpsgd', trainer.AggregatorType.DPSGD, None, False),
      ('dpftrl', trainer.AggregatorType.DPFTRL, None, True))
  def test_train_and_eval_fed_partial(self, aggregator_type,
                                      target_unclipped_quantile,
                                      trainable_conv):
    task = _get_emnist_test_task(trainable_conv=trainable_conv)
    server_optimizer, client_optimizer, learning_rate_fn = (
        trainer.configure_client_scheduled_optimizers(
            client_learning_rate=0.01))
    aggregator = trainer.configure_aggregator(
        aggregator_type=aggregator_type,
        model_fn=task.embedding_model_fn,
        clip_norm=0.1,
        noise_multiplier=1e-3,
        report_goal=100,
        target_unclipped_quantile=target_unclipped_quantile)
    process_type = trainer.ProcessType.FEDPARTIAL
    process = trainer.build_train_process(
        process_type,
        aggregator=aggregator,
        task=task,
        server_optimizer=server_optimizer,
        client_optimizer=client_optimizer,
        client_learning_rate_fn=learning_rate_fn)
    eval_fn = trainer.build_eval_fn(
        process_type, train_process=process, task=task)
    export_fn = trainer.build_export_fn(
        process_type, task=task, train_process=process)
    export_fn = functools.partial(export_fn, export_dir=self.get_temp_dir())
    trainer.train_and_eval(
        task,
        train_process=process,
        evaluation_fn=eval_fn,
        export_fn=export_fn,
        total_rounds=3,
        clients_per_round=1,
        rounds_per_eval=2,
        metrics_managers=[tff.program.LoggingReleaseManager()])


if __name__ == '__main__':
  tf.test.main()
