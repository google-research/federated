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
"""End-to-end tests for federated trainer tasks."""

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_federated as tff

from optimization.tasks import cifar100
from optimization.tasks import emnist
from optimization.tasks import emnist_ae
from optimization.tasks import shakespeare
from optimization.tasks import stackoverflow_nwp
from optimization.tasks import stackoverflow_tp
from optimization.tasks import training_specs


def iterative_process_builder(model_fn):
  return tff.learning.build_federated_averaging_process(
      model_fn=model_fn,
      client_optimizer_fn=tf.keras.optimizers.SGD,
      server_optimizer_fn=tf.keras.optimizers.SGD)


class FederatedTasksTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('cifar100', 'cifar100', cifar100.configure_training),
      ('emnist_cr', 'emnist_cr', emnist.configure_training),
      ('emnist_ae', 'emnist_ae', emnist_ae.configure_training),
      ('shakespeare', 'shakespeare', shakespeare.configure_training),
      ('stackoverflow_nwp', 'stackoverflow_nwp',
       stackoverflow_nwp.configure_training),
      ('stackoverflow_tp', 'stackoverflow_tp',
       stackoverflow_tp.configure_training),
  )
  def test_run_federated(self, task_name, config_fn):
    task_spec = training_specs.TaskSpec(
        iterative_process_builder=iterative_process_builder,
        client_epochs_per_round=1,
        client_batch_size=32,
        clients_per_round=1,
        client_datasets_random_seed=1)
    runner_spec = config_fn(task_spec)

    tff.simulation.run_simulation(
        process=runner_spec.iterative_process,
        client_selection_fn=runner_spec.client_datasets_fn,
        total_rounds=1)


if __name__ == '__main__':
  tf.test.main()
