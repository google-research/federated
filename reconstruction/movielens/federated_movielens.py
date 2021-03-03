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
"""Federated MovieLens matrix factorization runner library."""

import functools
import os
from typing import Callable, List, Optional

from absl import logging
import tensorflow as tf
import tensorflow_federated as tff
from reconstruction.movielens import models
from reconstruction.movielens import movielens_dataset
from reconstruction.shared import federated_trainer_utils
from utils import training_loop


def run_federated(
    iterative_process_builder: Callable[..., tff.templates.IterativeProcess],
    evaluation_computation_builder: Callable[..., tff.Computation],
    *,  # Caller passes below args by name.
    client_batch_size: int,
    clients_per_round: int,
    global_variables_only: bool,
    # Begin task-specific parameters.
    split_by_user: bool,
    split_train_fraction: float,
    split_val_fraction: float,
    normalize_ratings: bool,
    max_examples_per_user: Optional[int],
    num_items: int,
    num_latent_factors: int,
    add_biases: bool,
    l2_regularization: float,
    spreadout_lambda: float,
    accuracy_threshold: float,
    dataset_path:
    str = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip',
    # End task-specific parameters.
    total_rounds: int,
    experiment_name: Optional[str] = 'federated_ml_mf',
    root_output_dir: Optional[str] = '/tmp/fed_recon',
    **kwargs):
  """Runs an iterative process on the MovieLens matrix factorization task.

  This method will load and pre-process dataset and construct a model used for
  the task. It then uses `iterative_process_builder` to create an iterative
  process that it applies to the task, using
  `federated_research.utils.training_loop`.

  This algorithm only sends updates for item embeddings every round. User
  embeddings are reconstructed every round based on the latest item embeddings
  and user data.

  We assume that the iterative process has the following functional type
  signatures:

   *   `initialize`: `( -> S@SERVER)` where `S` represents the server state.
   *   `next`: `<S@SERVER, {B*}@CLIENTS> -> <S@SERVER, T@SERVER>` where `S`
        represents the server state, `{B*}` represents the client datasets,
        and `T` represents a python `Mapping` object.

  Args:
    iterative_process_builder: A function that accepts a no-arg `model_fn`, a
      `loss_fn`, a `metrics_fn`, and a `client_weight_fn`, and returns a
      `tff.templates.IterativeProcess`. The `model_fn` must return a
      `reconstruction_model.ReconstructionModel`. See `federated_trainer.py` for
      an example.
    evaluation_computation_builder: A function that accepts a no-arg `model_fn`,
      a loss_fn`, and a `metrics_fn`, and returns a `tff.Computation` for
      federated reconstruction evaluation. The `model_fn` must return a
      `reconstruction_model.ReconstructionModel`. See `federated_trainer.py` for
      an example.
    client_batch_size: An integer representing the batch size used on clients.
    clients_per_round: An integer representing the number of clients
      participating in each round.
    global_variables_only: If True, the `ReconstructionModel` contains
      all model variables as global variables. This can be useful for
      baselines involving aggregating all variables.
    split_by_user: Whether to split MovieLens data into train/val/test by user
      ID or by timestamp. If True, `movielens_dataset.split_tf_datasets` is used
      to partition the set of users into disjoint train/val/test sets. If False,
      `movielens_dataset.split_ratings_df` is used to split each user's data
      into train/val/test portions, so that each user shows up in each data
      partition. Setting to False can be useful for comparing with server-side
      training, since server matrix factorization requires test users to have
      been seen before (since otherwise we don't have trained user embeddings
      for these users).
    split_train_fraction: The fraction of data to use for the train set.
    split_val_fraction: The fraction of data to use for the val set.
      `split_train_fraction` and `split_val_fraction` should sum to no more than
      1. 1 - split_train_fraction - split_val_fraction of the data is left for
      test.
    normalize_ratings: Whether to normalize ratings in 1-5 to be in {-1, -0.5,
      0, 0.5, 1} via a linear scaling.
    max_examples_per_user: If not None, limit the number of rating examples for
      each user to this many examples.
    num_items: Number of items in the preferences matrix.
    num_latent_factors: Dimensionality of the learned user/item embeddings used
      to factorize the preferences matrix.
    add_biases: If True, add three bias terms: (1) user-specific bias, (2)
      item-specific bias, and (3) global bias.
    l2_regularization: The constant to use to scale L2 regularization on all
      weights, including the factorized matrices and the (optional) biases. A
      value of 0.0 indicates no regularization.
    spreadout_lambda: Scaling constant for spreadout regularization on item
      embeddings. This ensures that item embeddings are generally spread far
      apart, and that random items have dissimilar embeddings. See
      `models.EmbeddingSpreadoutRegularizer` for details. A value of 0.0
      indicates no regularization.
    accuracy_threshold: Threshold to use to determine whether a prediction is
      considered correct for metrics.
    dataset_path: URL or local path to the MovieLens 1M dataset. If a URL is
      passed, it is expected to be a .zip archive that will be extracted.
    total_rounds: The number of federated training rounds.
    experiment_name: The name of the experiment being run. This will be appended
      to the `root_output_dir` for purposes of writing outputs.
    root_output_dir: The name of the root output directory for writing
      experiment outputs.
    **kwargs: Additional arguments configuring the training loop. For details on
      supported arguments, see training_loop.py`.
  """

  logging.info('Copying MovieLens data.')
  if tf.io.gfile.exists('/tmp/ml-1m'):
    tf.io.gfile.rmtree('/tmp/ml-1m')

  if dataset_path.startswith('http'):
    movielens_dataset.download_and_extract_data(dataset_path, '/tmp')
  else:
    tf.io.gfile.makedirs('/tmp/ml-1m/')
    tf.io.gfile.copy(
        os.path.join(dataset_path, 'ratings.dat'),
        '/tmp/ml-1m/ratings.dat',
        overwrite=True)
    tf.io.gfile.copy(
        os.path.join(dataset_path, 'movies.dat'),
        '/tmp/ml-1m/movies.dat',
        overwrite=True)
    tf.io.gfile.copy(
        os.path.join(dataset_path, 'users.dat'),
        '/tmp/ml-1m/users.dat',
        overwrite=True)
  logging.info('Finished copying MovieLens data.')

  ratings_df, _, _ = movielens_dataset.load_movielens_data(
      normalize_ratings=normalize_ratings)

  # Split the ratings into training/val/test.
  if split_by_user:
    tf_datasets = movielens_dataset.create_tf_datasets(
        ratings_df=ratings_df,
        personal_model=True,
        batch_size=client_batch_size,
        max_examples_per_user=max_examples_per_user,
        num_local_epochs=1)

    tf_train_datasets, tf_val_datasets, tf_test_datasets = movielens_dataset.split_tf_datasets(
        tf_datasets,
        train_fraction=split_train_fraction,
        val_fraction=split_val_fraction)
  else:
    train_ratings_df, val_ratings_df, test_ratings_df = movielens_dataset.split_ratings_df(
        ratings_df,
        train_fraction=split_train_fraction,
        val_fraction=split_val_fraction)

    tf_train_datasets = movielens_dataset.create_tf_datasets(
        ratings_df=train_ratings_df,
        personal_model=True,
        batch_size=client_batch_size,
        max_examples_per_user=max_examples_per_user,
        num_local_epochs=1)
    tf_val_datasets = movielens_dataset.create_tf_datasets(
        ratings_df=val_ratings_df,
        personal_model=True,
        batch_size=client_batch_size,
        max_examples_per_user=max_examples_per_user,
        num_local_epochs=1)
    tf_test_datasets = movielens_dataset.create_tf_datasets(
        ratings_df=test_ratings_df,
        personal_model=True,
        batch_size=client_batch_size,
        max_examples_per_user=max_examples_per_user,
        num_local_epochs=1)

  model_fn = models.build_reconstruction_model(
      functools.partial(
          models.get_matrix_factorization_model,
          num_users=1,
          num_items=num_items,
          num_latent_factors=num_latent_factors,
          personal_model=True,
          add_biases=add_biases,
          l2_regularization=l2_regularization,
          spreadout_lambda=spreadout_lambda),
      global_variables_only=global_variables_only)

  loss_fn = models.get_loss_fn()
  metrics_fn = models.get_metrics_fn(accuracy_threshold=accuracy_threshold)

  training_process = iterative_process_builder(
      model_fn, loss_fn=loss_fn, metrics_fn=metrics_fn)
  evaluation_computation = evaluation_computation_builder(
      model_fn, loss_fn=loss_fn, metrics_fn=metrics_fn)

  def client_datasets_fn_from_tf_datasets(
      tf_datasets: List[tf.data.Dataset],
      clients_per_round: int,
  ) -> Callable[[int], List[tf.data.Dataset]]:
    """Produces a sampling function for train/val/test from a list of datasets."""
    sample_clients_fn = federated_trainer_utils.build_list_sample_fn(
        list(range(len(tf_datasets))), size=clients_per_round, replace=False)

    def client_datasets_fn(round_num):
      sampled_clients = sample_clients_fn(round_num)
      return [tf_datasets[client_id] for client_id in sampled_clients]

    return client_datasets_fn

  # Create client sampling functions for each of train/val/test.
  train_client_datasets_fn = client_datasets_fn_from_tf_datasets(
      tf_train_datasets, clients_per_round=clients_per_round)
  val_client_datasets_fn = client_datasets_fn_from_tf_datasets(
      tf_val_datasets, clients_per_round=clients_per_round)
  test_client_datasets_fn = client_datasets_fn_from_tf_datasets(
      tf_test_datasets, clients_per_round=clients_per_round)

  # Create final evaluation functions to pass to `training_loop`.
  val_fn = federated_trainer_utils.build_eval_fn(
      evaluation_computation=evaluation_computation,
      client_datasets_fn=val_client_datasets_fn,
      get_model=training_process.get_model_weights)
  test_fn = federated_trainer_utils.build_eval_fn(
      evaluation_computation=evaluation_computation,
      client_datasets_fn=test_client_datasets_fn,
      get_model=training_process.get_model_weights)
  test_fn = functools.partial(test_fn, round_num=0)

  logging.info('Starting training loop.')
  training_loop.run(
      iterative_process=training_process,
      client_datasets_fn=train_client_datasets_fn,
      validation_fn=val_fn,
      test_fn=test_fn,
      total_rounds=total_rounds,
      experiment_name=experiment_name,
      root_output_dir=root_output_dir,
      **kwargs)
