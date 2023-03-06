# Copyright 2023, Google LLC.
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
"""EMNIST bandits task."""
from collections.abc import Callable
import enum
import functools
from typing import Any, Optional
from absl import logging

import tensorflow as tf
import tensorflow_federated as tff

from bandits.tasks import emnist_preprocessing
from bandits.tasks import task_utils


MAX_CLIENT_DATASET_SIZE = 418


class ModelType(enum.Enum):
  LENET = enum.auto()
  LINEAR = enum.auto()


def get_model_types():
  return list(ModelType)


def _create_original_fedavg_cnn_model(
    only_digits: bool = True, seed: Optional[int] = None
) -> tf.keras.Model:
  """The CNN model used in https://arxiv.org/abs/1602.05629.

  The number of parameters when `only_digits=True` is (1,663,370), which matches
  what is reported in the paper. This model returns logits (before softmax), and
  expects `from_logits=True` when used together with
  `tf.keras.losses.SparseCategoricalCrossentropy`.

  Args:
    only_digits: If True, uses a final layer with 10 outputs, for use with the
      digits only EMNIST dataset. If False, uses 62 outputs for the larger
      dataset.
    seed: A random seed governing the model initialization and layer randomness.
      If not `None`, then the global random seed will be set before constructing
      the tensor initializer, in order to guarantee the same model is produced.

  Returns:
    A `tf.keras.Model`.
  """
  data_format = 'channels_last'
  if seed is not None:
    tf.random.set_seed(seed)
  initializer = tf.keras.initializers.GlorotNormal(seed=seed)

  max_pool = functools.partial(
      tf.keras.layers.MaxPooling2D,
      pool_size=(2, 2),
      padding='same',
      data_format=data_format,
  )
  conv2d = functools.partial(
      tf.keras.layers.Conv2D,
      kernel_size=5,
      padding='same',
      data_format=data_format,
      activation=tf.nn.relu,
      kernel_initializer=initializer,
  )
  model = tf.keras.models.Sequential([
      conv2d(filters=32, input_shape=(28, 28, 1)),
      max_pool(),
      conv2d(filters=64),
      max_pool(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(
          512, activation=tf.nn.relu, kernel_initializer=initializer
      ),
      tf.keras.layers.Dense(
          10 if only_digits else 62, kernel_initializer=initializer
      ),
  ])

  return model


# TODO(b/215566681): set the rewards to be 0/-1 instead of 1/0 so that the
# greedy algorithm won't stuck when initialized to all zero.
def _get_keras_linear_model(
    only_digits: bool = True,
    normalize_input: bool = True,
    zero_init: bool = True,
) -> tf.keras.Model:
  """Returns a linear model for EMNIST.

  Args:
    only_digits: Controls the output size for EMNIST classification: digits only
      (10 classes), or characters (62 classes).
    normalize_input: Whether to normalize the vectorized input to have unit L2
      norm.
    zero_init: Whether to initialize the linear model to all zero.
  """
  output_size = 10 if only_digits else 62
  layers = [
      tf.keras.layers.Reshape(input_shape=(28, 28, 1), target_shape=(28 * 28,))
  ]
  if normalize_input:
    layers += [task_utils.VecNormLayer()]
  if zero_init:
    layers += [tf.keras.layers.Dense(output_size, kernel_initializer='zeros')]
  else:
    layers += [tf.keras.layers.Dense(output_size)]
  return tf.keras.models.Sequential(layers)


def create_emnist_bandits_model_fn(
    bandits_data_spec: Any,
    *,
    model: ModelType = ModelType.LENET,
    only_digits: bool = True,
    loss: tf.keras.losses.Loss = task_utils.BanditsMSELoss(),
) -> Callable[[], tff.learning.models.VariableModel]:
  """Returns `model_fn` for bandits process in TFF simulation.

  Args:
    bandits_data_spec: The format of bandits data.
    model: The type of model, can be LeNet or a linear model.
    only_digits: If true, the model predicts 10 digits for EMNIST; if False, the
      model predicts 62 characters.
    loss: The loss for the returned `model_fn`.
  """
  if isinstance(loss, task_utils.MultiLabelCELoss):
    raise ValueError(
        'Use task_utils.SupervisedCELoss instead of task_utils.MultiLabelCELoss'
        'for EMNIST with sparse labels.'
    )
  if isinstance(loss, task_utils.MultiLabelMSELoss):
    raise ValueError(
        'Use task_utils.SupervisedMSELoss instead of'
        'task_utils.MultiLabelMSELoss for EMNIST with sparse labels.'
    )

  def keras_model_fn():
    if model == ModelType.LENET:
      return _create_original_fedavg_cnn_model(only_digits=only_digits)
    elif model == ModelType.LINEAR:
      return _get_keras_linear_model(only_digits=only_digits)
    else:
      raise ValueError(f'Unknown EMNIST model type: {model}')

  def model_fn():
    return tff.learning.from_keras_model(
        keras_model=keras_model_fn(),
        loss=loss,
        input_spec=bandits_data_spec,
        metrics=[
            task_utils.WrapCategoricalAccuracy(),
            task_utils.WeightCategoricalAccuracy(
                name='bandits_weighted_accuracy'
            ),
        ],
    )

  return model_fn


def create_emnist_preprocessed_datasets(
    *,
    train_client_batch_size: int = 16,
    test_client_batch_size: int = 128,
    train_client_epochs_per_round: int = 1,
    train_shuffle_buffer_size: Optional[int] = MAX_CLIENT_DATASET_SIZE,
    train_client_max_elements: Optional[int] = None,
    only_digits: bool = False,
    use_synthetic_data: bool = False,
    label_distribution_shift: bool = False,
    population_client_selection: Optional[str] = None,
) -> tff.simulation.baselines.BaselineTaskDatasets:
  """Returns preprocessed *supervised* client datasets in TFF simulation.

  The datasets will be used to generate bandits data for training and
  evaluation.

  Args:
    train_client_batch_size: The batch size for train clients.
    test_client_batch_size: The batch size for test clients.
    train_client_epochs_per_round: The number of epochs each train client should
      iterate over their local dataset, via `tf.data.Dataset.repeat`. Must be
      set to a positive integer.
    train_shuffle_buffer_size: An integer representing the shuffle buffer size
      (as in `tf.data.Dataset.shuffle`) for each train client's dataset. By
      default, this is set to the largest dataset size among all clients. If set
      to some integer less than or equal to 1, no shuffling occurs. If set to
      None, a value will be chosen based on `tff.simulation.baselines` default.
    train_client_max_elements: The maximum number of samples per client via
      `tf.data.Dataset.take`. If `None`, all the client data will be used.
    only_digits: A boolean representing whether to take the digits-only
      EMNIST-10 (with only 10 labels) or the full EMNIST-62 dataset with digits
      and characters (62 labels). If set to True, we use EMNIST-10, otherwise we
      use EMNIST-62.
    use_synthetic_data: Use synthetic data for fast and robust unit tests.
    label_distribution_shift: Label distribution shift by shifting the original
      integer label 36-61 (i.e., characters a-z) to 10-35 (i.e., characters A-Z
      ).
    population_client_selection: Use a subset of clients to form the training
      population; can be useful for distribution shift settings. Should be in
      the format of "start_index-end_index" for [start_index, end_index) of all
      the sorted clients in a simulation dataset. For example, "0-1000" will
      select the first 1000 clients.
  """
  if use_synthetic_data:
    synthetic_data = tff.simulation.datasets.emnist.get_synthetic()
    emnist_train = synthetic_data
    emnist_test = synthetic_data
  else:
    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(
        only_digits=only_digits
    )
  if population_client_selection is not None:
    total_population = len(emnist_train.client_ids)
    emnist_train = task_utils.clientdata_for_select_clients(
        emnist_train, population_client_selection
    )
    logging.info(
        'Use %d of %d clients selected by %s',
        len(emnist_train.client_ids),
        total_population,
        population_client_selection,
    )

  train_client_spec = tff.simulation.baselines.ClientSpec(
      num_epochs=train_client_epochs_per_round,
      batch_size=train_client_batch_size,
      max_elements=train_client_max_elements,
      shuffle_buffer_size=train_shuffle_buffer_size,
  )
  eval_client_spec = tff.simulation.baselines.ClientSpec(
      num_epochs=1,
      batch_size=test_client_batch_size,
      max_elements=None,
      shuffle_buffer_size=1,
  )
  train_preprocess_fn = emnist_preprocessing.create_preprocess_fn(
      train_client_spec, label_distribution_shift=label_distribution_shift
  )
  eval_preprocess_fn = emnist_preprocessing.create_preprocess_fn(
      eval_client_spec, label_distribution_shift=False
  )
  task_datasets = tff.simulation.baselines.BaselineTaskDatasets(
      train_data=emnist_train,
      test_data=emnist_test,
      validation_data=None,
      train_preprocess_fn=train_preprocess_fn,
      eval_preprocess_fn=eval_preprocess_fn,
  )
  return task_datasets
