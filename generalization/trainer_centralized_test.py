# Copyright 2021, Google LLC.
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
"""End-to-end tests for centralized trainer tasks."""

import collections
import os.path

from absl.testing import parameterized
import pandas as pd
import tensorflow as tf

from generalization.tasks import cifar100_image
from generalization.tasks import emnist_character
from generalization.tasks import shakespeare_character
from generalization.tasks import stackoverflow_word
from generalization.tasks import training_specs
from generalization.utils import centralized_training_loop
from generalization.utils import metric_utils

CIFAR100_IMAGE_TEST_FLAGS = collections.OrderedDict()
EMNIST_CHARACTER_TEST_FLAGS = collections.OrderedDict(
    model='cnn', merge_case=True)
SHAKESPEARE_CHARACTER_TEST_FLAGS = collections.OrderedDict()
STACKOVERFLOW_WORD_TEST_FLAGS = collections.OrderedDict()


class CentralizedTasksTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('cifar100_image', cifar100_image.configure_training_centralized, None, 4,
       CIFAR100_IMAGE_TEST_FLAGS, False),
      ('emnist_character', emnist_character.configure_training_centralized, 0.2,
       None, EMNIST_CHARACTER_TEST_FLAGS, False),
      ('shakespeare_character',
       shakespeare_character.configure_training_centralized, 0.2, None,
       SHAKESPEARE_CHARACTER_TEST_FLAGS, False),
      ('stackoverflow_word', stackoverflow_word.configure_training_centralized,
       None, 4, STACKOVERFLOW_WORD_TEST_FLAGS, True))
  def test_run_centralized(self, config_fn, unpart_clients_proportion,
                           train_val_ratio_intra_client, test_flags, has_test):
    task_spec = training_specs.TaskSpecCentralized(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        batch_size=2,
        centralized_shuffle_buffer_size=4,
        sql_database=None,
        unpart_clients_proportion=unpart_clients_proportion,
        train_val_ratio_intra_client=train_val_ratio_intra_client,
        part_clients_subsampling_rate=0.01,
        include_unpart_train_for_val=False,
        max_elements_per_client=8,
        part_clients_per_eval=2,
        unpart_clients_per_eval=2,
        test_clients_for_eval=2,
        resample_eval_clients=False,
        eval_client_batch_size=16,
        shared_random_seed=1)
    num_epochs = 1
    runner_spec = config_fn(task_spec, **test_flags)

    root_output_dir = self.get_temp_dir()
    exp_name = 'test_run_centralized'
    epochs_per_checkpoint = 1

    checkpoint_callback, metrics_callbacks = metric_utils.configure_default_callbacks(
        root_output_dir=root_output_dir,
        experiment_name=exp_name,
        epochs_per_checkpoint=epochs_per_checkpoint)

    centralized_training_loop.run(
        keras_model=runner_spec.keras_model,
        train_dataset=runner_spec.train_dataset,
        num_epochs=num_epochs,
        steps_per_epoch=2,
        part_train_eval_fn=runner_spec.part_train_eval_fn,
        part_val_fn=runner_spec.part_val_fn,
        unpart_fn=runner_spec.unpart_fn,
        test_fn=runner_spec.test_fn,
        checkpoint_callback=checkpoint_callback,
        metrics_callbacks=metrics_callbacks)

    self.assertTrue(tf.io.gfile.exists(root_output_dir))

    summary_dir = os.path.join(root_output_dir, 'logdir', exp_name)
    self.assertTrue(tf.io.gfile.exists(summary_dir))
    self.assertLen(tf.io.gfile.listdir(summary_dir), 1)

    results_dir = os.path.join(root_output_dir, 'results', exp_name)
    self.assertTrue(tf.io.gfile.exists(results_dir))
    metrics_file = os.path.join(results_dir, 'experiment.metrics.csv')
    self.assertTrue(tf.io.gfile.exists(metrics_file))

    metrics_csv = pd.read_csv(metrics_file)

    expected_rows = num_epochs + 1 if has_test else num_epochs
    self.assertLen(
        metrics_csv.index,
        expected_rows,
        msg='The output metrics CSV should have {} rows.'.format(expected_rows))

    possible_prefixes = [
        metric_utils.PART_TRAIN_EVAL_METRICS_PREFIX,
        metric_utils.PART_VAL_METRICS_PREFIX, metric_utils.UNPART_METRICS_PREFIX
    ]
    if has_test:
      possible_prefixes.append(metric_utils.TEST_METRICS_PREFIX)

    for prefix in possible_prefixes:
      prefixed_metric = prefix + 'loss/avg'
      self.assertIn(
          prefixed_metric,
          metrics_csv.columns,
          msg=f'The output metrics CSV should have a column "{prefixed_metric}"'
          'if training is successful.')


if __name__ == '__main__':
  tf.test.main()
