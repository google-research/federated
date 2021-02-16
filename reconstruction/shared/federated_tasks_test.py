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
"""End-to-end tests for federated reconstruction training tasks."""

import collections
import functools
import os.path
from typing import Any, Callable, List, Optional, Tuple

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_federated as tff

from reconstruction import evaluation_computation
from reconstruction import reconstruction_model
from reconstruction import reconstruction_utils
from reconstruction import training_process
from reconstruction.movielens import federated_movielens
from reconstruction.shared import federated_evaluation
from reconstruction.stackoverflow import federated_stackoverflow

# Task-specific flags in addition to shared flags.
STACKOVERFLOW_FINETUNE_FLAGS = collections.OrderedDict(
    max_elements_per_user=10,
    vocab_size=500,
    latent_size=50,
    embedding_size=50,
    global_variables_only=True)
STACKOVERFLOW_FLAGS = collections.OrderedDict(
    max_elements_per_user=10,
    vocab_size=500,
    latent_size=50,
    embedding_size=50,
    global_variables_only=False)
MOVIELENS_FLAGS = collections.OrderedDict(
    split_by_user=True,
    split_train_fraction=0.8,
    split_val_fraction=0.1,
    normalize_ratings=False,
    max_examples_per_user=10,
    num_items=50,
    num_latent_factors=5,
    add_biases=False,
    l2_regularization=0.0,
    spreadout_lambda=0.0,
    accuracy_threshold=0.5,
    dataset_path='reconstruction/movielens/testdata',
    global_variables_only=False,
)


def iterative_process_builder(
    model_fn: Callable[[], reconstruction_model.ReconstructionModel],
    loss_fn: Callable[[], List[tf.keras.losses.Loss]],
    metrics_fn: Optional[Callable[[], List[tf.keras.metrics.Metric]]] = None,
    client_weight_fn: Optional[Callable[[Any], tf.Tensor]] = None,
    dataset_split_fn_builder: Callable[
        ..., reconstruction_utils.DatasetSplitFn] = reconstruction_utils
    .build_dataset_split_fn,
    task_name: str = 'stackoverflow_nwp',
    dp_noise_multiplier: Optional[float] = None,
    dp_zeroing: bool = True,
    clients_per_round: int = 5,
) -> tff.templates.IterativeProcess:
  """Creates an iterative process using a given TFF `model_fn`."""

  # Get aggregation factory for DP, if needed.
  aggregation_factory = None
  client_weighting = client_weight_fn
  if dp_noise_multiplier is not None:
    aggregation_factory = tff.learning.dp_aggregator(
        noise_multiplier=dp_noise_multiplier,
        clients_per_round=float(clients_per_round),
        zeroing=dp_zeroing)
    # DP is only implemented for uniform weighting.
    client_weighting = lambda _: 1.0

  if task_name == 'stackoverflow_nwp_finetune':
    # The returned iterative process would be basically the same as the one
    # created by the standard `tff.learning.build_federated_averaging_process`.
    client_epochs_per_round = 1

    # No need to split the client data as the model has only global variables.
    def dataset_split_fn(
        client_dataset: tf.data.Dataset,
        round_num: tf.Tensor) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
      del round_num
      return client_dataset.repeat(0), client_dataset.repeat(
          client_epochs_per_round)

  else:
    dataset_split_fn = dataset_split_fn_builder(
        recon_epochs_max=1,
        recon_epochs_constant=1,
        recon_steps_max=1,
        post_recon_epochs=1,
        post_recon_steps_max=1,
        split_dataset=False)

  return training_process.build_federated_reconstruction_process(
      model_fn=model_fn,
      loss_fn=loss_fn,
      metrics_fn=metrics_fn,
      server_optimizer_fn=lambda: tf.keras.optimizers.SGD(1.0),
      client_optimizer_fn=lambda: tf.keras.optimizers.SGD(1.0),
      reconstruction_optimizer_fn=lambda: tf.keras.optimizers.SGD(1.0),
      client_weight_fn=client_weighting,
      dataset_split_fn=dataset_split_fn,
      aggregation_factory=aggregation_factory)


def evaluation_computation_builder(
    model_fn: Callable[[], reconstruction_model.ReconstructionModel],
    loss_fn: Callable[[], tf.losses.Loss],
    metrics_fn: Callable[[], List[tf.metrics.Metric]],
    dataset_split_fn_builder: Callable[
        ..., reconstruction_utils.DatasetSplitFn] = reconstruction_utils
    .build_dataset_split_fn,
    task_name: str = 'stackoverflow_nwp',
) -> tff.Computation:
  """Creates an evaluation computation using federated reconstruction."""

  # For a `stackoverflow_nwp_finetune` task, the first dataset returned by
  # `dataset_split_fn` is used for fine-tuning global variables. For other
  # tasks, the first dataset is used for reconstructing local variables.
  dataset_split_fn = dataset_split_fn_builder(
      recon_epochs_max=1,
      recon_epochs_constant=1,
      recon_steps_max=1,
      post_recon_epochs=1,
      post_recon_steps_max=1,
      split_dataset=True)

  if task_name == 'stackoverflow_nwp_finetune':
    return federated_evaluation.build_federated_finetune_evaluation(
        model_fn=model_fn,
        loss_fn=loss_fn,
        metrics_fn=metrics_fn,
        finetune_optimizer_fn=lambda: tf.keras.optimizers.SGD(1.0),
        dataset_split_fn=dataset_split_fn)

  return evaluation_computation.build_federated_reconstruction_evaluation(
      model_fn=model_fn,
      loss_fn=loss_fn,
      metrics_fn=metrics_fn,
      reconstruction_optimizer_fn=lambda: tf.keras.optimizers.SGD(1.0),
      dataset_split_fn=dataset_split_fn)


class FederatedTasksTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('stackoverflow_nwp', 'stackoverflow_nwp',
       federated_stackoverflow.run_federated, STACKOVERFLOW_FLAGS, False),
      ('stackoverflow_nwp_finetune', 'stackoverflow_nwp_finetune',
       federated_stackoverflow.run_federated, STACKOVERFLOW_FINETUNE_FLAGS,
       False),
      ('movielens_mf', 'movielens_mf', federated_movielens.run_federated,
       MOVIELENS_FLAGS, False),
      ('movielens_mf_dp', 'movielens_mf', federated_movielens.run_federated,
       MOVIELENS_FLAGS, True),
  )
  def test_run_federated(self, task_name, run_federated_fn, task_flags,
                         enable_dp):
    total_rounds = 1
    clients_per_round = 20
    root_output_dir = self.get_temp_dir()
    exp_name = 'test_run_federated'
    dp_noise_multiplier = None
    dp_zeroing = True
    if enable_dp:
      dp_noise_multiplier = 1.0
    shared_args = collections.OrderedDict(
        client_batch_size=10,
        clients_per_round=1,
        total_rounds=total_rounds,
        iterative_process_builder=functools.partial(
            iterative_process_builder,
            task_name=task_name,
            dp_noise_multiplier=dp_noise_multiplier,
            dp_zeroing=dp_zeroing,
            clients_per_round=clients_per_round),
        evaluation_computation_builder=functools.partial(
            evaluation_computation_builder, task_name=task_name),
        rounds_per_checkpoint=10,
        rounds_per_eval=10,
        root_output_dir=root_output_dir,
        experiment_name=exp_name)

    run_federated_fn(**shared_args, **task_flags)

    results_dir = os.path.join(root_output_dir, 'results', exp_name)
    self.assertTrue(tf.io.gfile.exists(results_dir))

    metrics_manager = tff.simulation.CSVMetricsManager(
        os.path.join(results_dir, 'experiment.metrics.csv'))
    fieldnames, metrics = metrics_manager.get_metrics()

    self.assertIn(
        'train/loss',
        fieldnames,
        msg='The output metrics should have a `train/loss` column if training '
        'is successful.')
    self.assertIn(
        'eval/loss',
        fieldnames,
        msg='The output metrics should have a `train/loss` column if validation'
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
