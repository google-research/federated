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
"""Federated CIFAR-10 classification library using TFF."""

import functools
from typing import Callable, Optional
from absl import logging

import tensorflow as tf
import tensorflow_federated as tff

from fedopt_guide import training_loop
from utils.datasets import cifar10_dataset
from utils.models import resnet_models

CIFAR_SHAPE = (32, 32, 3)
NUM_CLASSES = 100


def run_federated(
    iterative_process_builder: Callable[..., tff.templates.IterativeProcess],
    client_epochs_per_round: int,
    client_batch_size: int,
    clients_per_round: int,
    client_datasets_random_seed: Optional[int] = None,
    crop_size: Optional[int] = 24,
    total_rounds: Optional[int] = 1500,
    experiment_name: Optional[str] = 'federated_cifar10',
    root_output_dir: Optional[str] = '/tmp/fed_opt',
    uniform_weighting: Optional[bool] = False,
    **kwargs):
  """Runs an iterative process on the CIFAR-10 classification task.

  This method will load and pre-process dataset and construct a model used for
  the task. It then uses `iterative_process_builder` to create an iterative
  process that it applies to the task, using
  `federated_research.utils.training_loop`.
  We assume that the iterative process has the following functional type
  signatures:
    *   `initialize`: `( -> S@SERVER)` where `S` represents the server state.
    *   `next`: `<S@SERVER, {B*}@CLIENTS> -> <S@SERVER, T@SERVER>` where `S`
        represents the server state, `{B*}` represents the client datasets,
        and `T` represents a python `Mapping` object.
  The iterative process must also have a callable attribute `get_model_weights`
  that takes as input the state of the iterative process, and returns a
  `tff.learning.ModelWeights` object.

  Args:
    iterative_process_builder: A function that accepts a no-arg `model_fn`, and
      returns a `tff.templates.IterativeProcess`. The `model_fn` must return a
      `tff.learning.Model`.
    client_epochs_per_round: An integer representing the number of epochs of
      training performed per client in each training round.
    client_batch_size: An integer representing the batch size used on clients.
    clients_per_round: An integer representing the number of clients
      participating in each round.
    client_datasets_random_seed: An optional int used to seed which clients are
      sampled at each round. If `None`, no seed is used.
    crop_size: An optional integer representing the resulting size of input
      images after preprocessing.
    total_rounds: The number of federated training rounds.
    experiment_name: The name of the experiment being run. This will be appended
      to the `root_output_dir` for purposes of writing outputs.
    root_output_dir: The name of the root output directory for writing
      experiment outputs.
    uniform_weighting: Whether to weigh clients uniformly. If false, clients are
      weighted by the number of samples.
    **kwargs: Additional arguments configuring the training loop. For details on
      supported arguments, see `federated_research/utils/training_utils.py`.
  """

  crop_shape = (crop_size, crop_size, 3)

  cifar_train, _ = cifar10_dataset.get_federated_datasets(
      train_client_epochs_per_round=client_epochs_per_round,
      train_client_batch_size=client_batch_size,
      crop_shape=crop_shape)

  _, cifar_test = cifar10_dataset.get_centralized_datasets(
      crop_shape=crop_shape)

  input_spec = cifar_train.create_tf_dataset_for_client(
      cifar_train.client_ids[0]).element_spec

  model_builder = functools.partial(
      resnet_models.create_resnet18,
      input_shape=crop_shape,
      num_classes=NUM_CLASSES)

  loss_builder = tf.keras.losses.SparseCategoricalCrossentropy
  metrics_builder = lambda: [tf.keras.metrics.SparseCategoricalAccuracy()]

  def tff_model_fn() -> tff.learning.Model:
    return tff.learning.from_keras_model(
        keras_model=model_builder(),
        input_spec=input_spec,
        loss=loss_builder(),
        metrics=metrics_builder())

  if uniform_weighting:
    client_weight_fn = tff.learning.ClientWeighting.UNIFORM
  else:
    client_weight_fn = tff.learning.ClientWeighting.NUM_EXAMPLES

  training_process = iterative_process_builder(tff_model_fn, client_weight_fn)

  client_datasets_fn = tff.simulation.build_uniform_client_sampling_fn(
      dataset=cifar_train,
      clients_per_round=clients_per_round,
      random_seed=client_datasets_random_seed)

  evaluate_fn = tff.learning.build_federated_evaluation(tff_model_fn)

  def validation_fn(model_weights, round_num):
    del round_num
    return evaluate_fn(model_weights, [cifar_test])

  def test_fn(model_weights):
    return evaluate_fn(model_weights, [cifar_test])

  logging.info('Training model:')
  logging.info(model_builder().summary())

  training_loop.run(
      iterative_process=training_process,
      train_client_datasets_fn=client_datasets_fn,
      evaluation_fn=validation_fn,
      test_fn=test_fn,
      total_rounds=total_rounds,
      experiment_name=experiment_name,
      root_output_dir=root_output_dir,
      **kwargs)
