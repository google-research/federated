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
"""Federated experiments on the Google Landmark dataset using TFF."""

from typing import Callable, Optional
from absl import logging
import tensorflow as tf
import tensorflow_federated as tff

from fedopt_guide import training_loop
from fedopt_guide.gld23k_mobilenet import dataset
from fedopt_guide.gld23k_mobilenet import mobilenet_v2


def run_federated(
    iterative_process_builder: Callable[..., tff.templates.IterativeProcess],
    client_epochs_per_round: int,
    client_batch_size: int,
    clients_per_round: int,
    max_elements_per_user: int,
    image_size: int,
    num_groups: int = 8,
    total_rounds: int = 3000,
    dataset_type: dataset.DatasetType = dataset.DatasetType.GLD23K,
    experiment_name: str = 'federated_gld23k',
    root_output_dir: str = '/tmp/fedopt_guide',
    dropout_prob: Optional[float] = None,
    client_datasets_random_seed: Optional[int] = None,
    **kwargs) -> None:
  """Runs an iterative process on the Google Landmark dataset.

  This method will load and pre-process dataset and construct a model used for
  the task. It then uses `iterative_process_builder` to create an iterative
  process that it applies to the task, using
  `federated_research/fedopt_guide/training_loop`.

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
    iterative_process_builder: A function that accepts a no-arg `model_fn`, a
      `client_weight_fn` and returns a `tff.templates.IterativeProcess`. The
      `model_fn` must return a `tff.learning.Model`.
    client_epochs_per_round: An integer representing the number of epochs of
      training performed per client in each training round.
    client_batch_size: An integer representing the batch size used on clients.
    clients_per_round: An integer representing the number of clients
      participating in each round.
    max_elements_per_user: The maximum number of elements processed for each
      client's dataset. This has be to a positive value or -1 (which means that
      all elements are taken for training).
    image_size: The height and width of images after preprocessing.
    num_groups: The number of groups in the GroupNorm layers of MobilenetV2.
    total_rounds: The number of federated training rounds.
    dataset_type: A `dataset.DatasetType` specifying which dataset is used for
      experiments.
    experiment_name: The name of the experiment being run. This will be appended
      to the `root_output_dir` for purposes of writing outputs.
    root_output_dir: The name of the root output directory for writing
      experiment outputs.
    dropout_prob: Probability of setting a weight to zero in the dropout layer
      of MobilenetV2. Must be in the range [0, 1). Setting it to None (default)
      or zero means no dropout.
    client_datasets_random_seed: An optional int used to seed which clients are
      sampled at each round. If `None`, no seed is used.
    **kwargs: Additional arguments configuring the training loop. For details on
      supported arguments, see
      `federated_research/fedopt_guide/training_utils.py`.
  """
  num_classes, shuffle_buffer_size = dataset.get_dataset_stats(dataset_type)

  train_data, _ = tff.simulation.datasets.gldv2.load_data(
      gld23k=True if dataset_type == dataset.DatasetType.GLD23K else False)
  _, test_data = dataset.get_centralized_datasets(
      image_size=image_size,
      batch_size=client_batch_size,
      dataset_type=dataset_type)

  if dropout_prob and (dropout_prob < 0 or dropout_prob >= 1):
    raise ValueError(
        f'Expected a value in [0, 1) for `dropout_prob`, found {dropout_prob}.')

  def model_builder() -> tf.keras.Model:
    return mobilenet_v2.create_mobilenet_v2(
        input_shape=(image_size, image_size, 3),
        num_groups=num_groups,
        num_classes=num_classes,
        dropout_prob=dropout_prob)

  loss_builder = tf.keras.losses.SparseCategoricalCrossentropy
  metrics_builder = lambda: [tf.keras.metrics.SparseCategoricalAccuracy()]
  input_spec = test_data.element_spec

  def model_fn() -> tff.learning.Model:
    return tff.learning.from_keras_model(
        keras_model=model_builder(),
        input_spec=input_spec,
        loss=loss_builder(),
        metrics=metrics_builder())

  training_process = iterative_process_builder(
      model_fn=model_fn, client_weight_fn=None)

  preprocessing_fn = dataset.get_preprocessing_fn(
      image_size=image_size,
      batch_size=client_batch_size,
      num_epochs=client_epochs_per_round,
      max_elements=max_elements_per_user,
      shuffle_buffer_size=shuffle_buffer_size)

  @tff.tf_computation(tf.string)
  def train_dataset_computation(client_id):
    client_train_data = train_data.dataset_computation(client_id)
    return preprocessing_fn(client_train_data)

  trainer = tff.simulation.compose_dataset_computation_with_iterative_process(
      dataset_computation=train_dataset_computation, process=training_process)

  # `compose_dataset_computation_with_iterative_process` does not inherit the
  # `get_model_weights` attribute from the `training_process`.
  if not hasattr(training_process, 'get_model_weights'):
    raise ValueError(
        'The `iterative_process_builder` must create an iterative process '
        'that has an attribute `get_model_weights`. It is a `tff.Computation` '
        'that accepts as input the state of an iterative process, and returns '
        'the model weights part from the state. If you use '
        '`tff.learning.build_federated_averaging_process`, it should already '
        'satisfy this requirement.')
  else:
    trainer.get_model_weights = training_process.get_model_weights

  client_ids_fn = tff.simulation.build_uniform_sampling_fn(
      train_data.client_ids,
      size=clients_per_round,
      replace=False,
      random_seed=client_datasets_random_seed)
  # We convert the output to a list (instead of an np.ndarray) so that it can
  # be used as input to the iterative process.
  client_ids_fn_as_list = lambda x: list(client_ids_fn(x))

  evaluate_fn = tff.learning.build_federated_evaluation(model_fn)

  def validation_fn(model_weights, round_num):
    del round_num
    return evaluate_fn(model_weights, [test_data])

  def test_fn(model_weights):
    return evaluate_fn(model_weights, [test_data])

  logging.info('Training model:')
  logging.info(model_builder().summary())

  training_loop.run(
      iterative_process=trainer,
      train_client_datasets_fn=client_ids_fn_as_list,
      evaluation_fn=validation_fn,
      test_fn=test_fn,
      total_rounds=total_rounds,
      experiment_name=experiment_name,
      root_output_dir=root_output_dir,
      **kwargs)
