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
"""End-to-end tests for federated CIFAR-10 task."""

import collections
import os.path

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_federated as tff

from fedopt_guide.cifar10_resnet import federated_cifar10


def iterative_process_builder(model_fn, client_weight_fn=None):
  return tff.learning.build_federated_averaging_process(
      model_fn=model_fn,
      client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.1),
      server_optimizer_fn=lambda: tf.keras.optimizers.SGD(1.0),
      client_weighting=client_weight_fn)


class FederatedTasksTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('cifar10', federated_cifar10.run_federated))
  def test_run_federated(self, run_federated_fn):
    total_rounds = 1
    shared_args = collections.OrderedDict(
        client_epochs_per_round=1,
        client_batch_size=32,
        clients_per_round=1,
        client_datasets_random_seed=1,
        total_rounds=total_rounds,
        iterative_process_builder=iterative_process_builder,
        rounds_per_checkpoint=10,
        rounds_per_eval=10)
    root_output_dir = self.get_temp_dir()
    exp_name = 'test_run_federated'
    shared_args['root_output_dir'] = root_output_dir
    shared_args['experiment_name'] = exp_name

    run_federated_fn(**shared_args)

    results_dir = os.path.join(root_output_dir, 'results', exp_name)
    self.assertTrue(tf.io.gfile.exists(results_dir))

    scalar_manager = tff.simulation.CSVMetricsManager(
        os.path.join(results_dir, 'experiment.metrics.csv'))
    fieldnames, metrics = scalar_manager.get_metrics()

    self.assertIn(
        'train/train/loss',
        fieldnames,
        msg='The output metrics should have a `train/loss` column if training '
        'is successful.')
    self.assertIn(
        'eval/loss',
        fieldnames,
        msg='The output metrics should have a `eval/loss` column if validation'
        ' metrics computation is successful.')
    self.assertIn(
        'test/loss',
        fieldnames,
        msg='The output metrics should have a `test/loss` column if test '
        'metrics computation is successful.')
    self.assertLen(
        metrics,
        total_rounds + 1,
        msg='The number of rows in the metrics CSV should be the number of '
        'training rounds + 1 (as there is an extra row for validation/test set'
        'metrics after training has completed.')


if __name__ == '__main__':
  tf.test.main()
