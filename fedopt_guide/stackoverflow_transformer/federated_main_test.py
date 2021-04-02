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

import os
import tensorflow as tf
import tensorflow_federated as tff
from fedopt_guide.stackoverflow_transformer import federated_main


def iterative_process_builder(model_fn, client_weight_fn=None):
  return tff.learning.build_federated_averaging_process(
      model_fn=model_fn,
      client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.1),
      server_optimizer_fn=lambda: tf.keras.optimizers.SGD(1.0),
      client_weighting=client_weight_fn)


class FederatedMainTest(tf.test.TestCase):

  def test_run_federated(self):
    root_output_dir = self.create_tempdir()
    exp_name = 'test_run_federated'
    total_rounds = 1
    federated_main.run_federated(
        iterative_process_builder=iterative_process_builder,
        client_epochs_per_round=1,
        client_batch_size=4,
        clients_per_round=1,
        max_elements_per_user=16,
        total_rounds=total_rounds,
        vocab_size=10,
        client_datasets_random_seed=1,
        num_validation_examples=100,
        max_val_test_batches=1000,
        dim_embed=3,
        dim_model=3,
        dim_hidden=3,
        num_heads=1,
        experiment_name=exp_name,
        root_output_dir=root_output_dir,
        rounds_per_checkpoint=10,
        rounds_per_eval=10)

    results_dir = os.path.join(root_output_dir, 'results', exp_name)
    self.assertTrue(tf.io.gfile.exists(results_dir))

    scalar_manager = tff.simulation.CSVMetricsManager(
        os.path.join(results_dir, 'experiment.metrics.csv'))
    fieldnames, metrics = scalar_manager.get_metrics()

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
