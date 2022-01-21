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
"""Library for creating periodic distribution shift tasks on EMNIST."""

# TODO(b/193904908): add unit tests.

import enum
from typing import Optional, Union

import tensorflow as tf
import tensorflow_federated as tff

from periodic_distribution_shift.datasets import emnist_preprocessing
from periodic_distribution_shift.models import keras_utils_dual_branch_kmeans
from periodic_distribution_shift.tasks import dist_shift_task
from periodic_distribution_shift.tasks import dist_shift_task_data


class CharacterRecognitionModel(enum.Enum):
  """Enum for EMNIST character recognition models."""
  SINGLE_BRANCH_CNN = 'single_branch_cnn'
  DUAL_BRANCH_CNN = 'dual_branch_cnn'


_CHARACTER_RECOGNITION_MODELS = [e.value for e in CharacterRecognitionModel]


def create_single_branch_cnn_model() -> tf.keras.Model:
  """Create a single-branch convolutional network.

  Returns:
    An uncompiled `tf.keras.Model`.
  """
  data_format = 'channels_last'
  image = tf.keras.layers.Input(shape=(None, None, None), name='image')
  group = tf.keras.layers.Input(shape=(None,), name='group')

  model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(
          32,
          kernel_size=(3, 3),
          activation='relu',
          data_format=data_format,
          input_shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(
          64, kernel_size=(3, 3), activation='relu', data_format=data_format),
      tf.keras.layers.MaxPool2D(pool_size=(2, 2), data_format=data_format),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
  ])

  feature = model(image)
  n_classes = 62
  output = tf.keras.layers.Dense(n_classes, activation=tf.nn.softmax)(feature)

  return tf.keras.Model(inputs=[image, group], outputs=output)


def create_dual_branch_cnn_model() -> tf.keras.Model:
  """Create a dual-branch convolutional network.

  Returns:
    An uncompiled `tf.keras.Model`.
  """
  data_format = 'channels_last'
  image = tf.keras.layers.Input(shape=(None, None, None), name='image')
  group = tf.keras.layers.Input(shape=(None,), name='group')

  model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(
          32,
          kernel_size=(3, 3),
          activation='relu',
          data_format=data_format,
          input_shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(
          64, kernel_size=(3, 3), activation='relu', data_format=data_format),
      tf.keras.layers.MaxPool2D(pool_size=(2, 2), data_format=data_format),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
  ])

  feature = model(image)
  final_fc_list = [
      tf.keras.layers.Dense(62, activation=None, name='branch_1'),
      tf.keras.layers.Dense(62, activation=None, name='branch_2')
  ]

  pred1 = final_fc_list[0](feature)
  pred2 = final_fc_list[1](feature)

  output = [pred1, pred2, feature]

  return tf.keras.Model(inputs=[image, group], outputs=output)


def _get_character_recognition_model(
    model_id: Union[str, CharacterRecognitionModel]) -> tf.keras.Model:
  """Constructs a `tf.keras.Model` for character recognition."""
  try:
    model_enum = CharacterRecognitionModel(model_id)
  except ValueError:
    raise ValueError('The model argument must be one of {}, found {}'.format(
        _CHARACTER_RECOGNITION_MODELS, model_id))

  if model_enum == CharacterRecognitionModel.SINGLE_BRANCH_CNN:
    keras_model = create_single_branch_cnn_model()
  elif model_enum == CharacterRecognitionModel.DUAL_BRANCH_CNN:
    keras_model = create_dual_branch_cnn_model()
  else:
    raise ValueError('The model id must be one of {}, found {}'.format(
        _CHARACTER_RECOGNITION_MODELS, model_id))
  return keras_model


def filter_chars(sample):
  return tf.math.greater(sample['label'], tf.constant(9))


def create_character_recognition_task_from_datasets(
    train_client_spec: tff.simulation.baselines.ClientSpec,
    eval_client_spec: Optional[tff.simulation.baselines.ClientSpec],
    train_data: tff.simulation.datasets.ClientData,
    test_data: tff.simulation.datasets.ClientData,
    aggregated_kmeans: bool = False,
    label_smooth_w: float = 0.,
    label_smooth_eps: float = 1.,
    batch_majority_voting: bool = False) -> dist_shift_task.DistShiftTask:
  """Creates a baseline task for character recognition on EMNIST.

  Args:
    train_client_spec: A `tff.simulation.baselines.ClientSpec` specifying how to
      preprocess train client data.
    eval_client_spec: An optional `tff.simulation.baselines.ClientSpec`
      specifying how to preprocess evaluation client data. If set to `None`, the
      evaluation datasets will use a batch size of 64 with no extra
      preprocessing.
    train_data: A `tff.simulation.datasets.ClientData` used for training.
    test_data: A `tff.simulation.datasets.ClientData` used for testing.
    aggregated_kmeans: Whether to use aggregated k-means. If set to `True`, we
      will create a dual branch model, and use k-means based on the feautres to
      select branches in the forward pass.
    label_smooth_w: Weight of label smoothing regularization on the unselected
      branch. Only effective when `aggregated_kmeans = True`.
    label_smooth_eps: Epsilon of the label smoothing for the unselected branch.
      The value should be within 0 to 1, where 1 enforces the prediction to be
      uniform on all labels, and 0 falls back to cross entropy loss on one-hot
      label. The label smoothing regularization is defined as
      `L_{CE}(g(x), (1 - epsilon) * y + epsilon * 1/n)`, where L_{CE} is the
      cross entropy loss, g(x) is the prediction, epsilon represents the
      smoothness. Only effective when `aggregated_kmeans = True`.
    batch_majority_voting: Whether to use batch-wise majority voting to select
      the branch during test time. If set to True, we select the branch
      according to the majority within the minibatch during inference.
      Otherwise, we select the branch for each sample. Only effective when
      `aggregated_kmeans = True`.

  Returns:
    A `dist_shift_task.DistShiftTask`.
  """
  if label_smooth_w < 0:
    raise ValueError(f'label_smooth_w should be a non-negative number. '
                     f'Got {label_smooth_w}.')
  if label_smooth_eps > 1. or label_smooth_eps < 0.:
    raise ValueError(f'label_smooth_eps should be within [0, 1]. '
                     f'Got value {label_smooth_eps}.')

  emnist_task = 'character_recognition'

  if eval_client_spec is None:
    eval_client_spec = tff.simulation.baselines.ClientSpec(
        num_epochs=1, batch_size=64, shuffle_buffer_size=1)

  train_preprocess_fn = emnist_preprocessing.create_preprocess_fn(
      train_client_spec, emnist_task=emnist_task)
  eval_preprocess_fn = emnist_preprocessing.create_preprocess_fn(
      eval_client_spec, emnist_task=emnist_task)

  # Create the splits for validation.
  dataset_dict = {}
  full_validation_set = test_data.create_tf_dataset_from_all_clients()
  chars_val_set = full_validation_set.filter(filter_chars)

  _, digits_test_data = tff.simulation.datasets.emnist.load_data(
      only_digits=False)
  digits_val_set = digits_test_data.create_tf_dataset_from_all_clients()
  dataset_dict['full'] = eval_preprocess_fn(full_validation_set)
  dataset_dict['digits'] = eval_preprocess_fn(digits_val_set)
  dataset_dict['chars'] = eval_preprocess_fn(chars_val_set)

  task_datasets = dist_shift_task_data.DistShiftDatasets(
      train_data=train_data,
      test_data=test_data,
      validation_data_dict=dataset_dict,
      train_preprocess_fn=train_preprocess_fn,
      eval_preprocess_fn=eval_preprocess_fn)

  def model_fn() -> tff.learning.Model:
    if aggregated_kmeans:
      return keras_utils_dual_branch_kmeans.from_keras_model(
          keras_model=_get_character_recognition_model('dual_branch_cnn'),
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          input_spec=task_datasets.element_type_structure,
          metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
          from_logits=True,
          uniform_reg=label_smooth_w,
          label_smoothing=label_smooth_eps,
          batch_majority_voting=batch_majority_voting,
      )
    else:
      return tff.learning.keras_utils.from_keras_model(
          keras_model=_get_character_recognition_model('single_branch_cnn'),
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          input_spec=task_datasets.element_type_structure,
          metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

  return dist_shift_task.DistShiftTask(task_datasets, model_fn)


def char_filter_fn(sample):
  if tf.math.equal(sample['prefix'], tf.constant('g1_')):
    return True
  else:
    return tf.math.greater(sample['label'], tf.constant(9))


def create_character_recognition_task(
    train_client_spec: tff.simulation.baselines.ClientSpec,
    eval_client_spec: Optional[tff.simulation.baselines.ClientSpec] = None,
    cache_dir: Optional[str] = None,
    use_synthetic_data: bool = False,
    aggregated_kmeans: bool = False,
    label_smooth_w: float = 0.,
    label_smooth_eps: float = 1.0,
    batch_majority_voting: bool = False) -> dist_shift_task.DistShiftTask:
  """Creates a baseline task for character recognition on EMNIST.

  The goal of the task is to minimize the sparse categorical crossentropy
  between the output labels of the model and the true label of the image. When
  `only_digits = True`, there are 10 possible labels (the digits 0-9), while
  when `only_digits = False`, there are 62 possible labels (both numbers and
  letters).

  This classification can be done using a number of different models, specified
  using the `model_id` argument. Below we give a list of the different models
  that can be used:

  *   `model_id = cnn_dropout`: A moderately sized convolutional network. Uses
  two convolutional layers, a max pooling layer, and dropout, followed by two
  dense layers.
  *   `model_id = cnn`: A moderately sized convolutional network, without any
  dropout layers. Matches the architecture of the convolutional network used
  by (McMahan et al., 2017) for the purposes of testing the FedAvg algorithm.
  *   `model_id = 2nn`: A densely connected network with 2 hidden layers, each
  with 200 hidden units and ReLU activations.

  Args:
    train_client_spec: A `tff.simulation.baselines.ClientSpec` specifying how to
      preprocess train client data.
    eval_client_spec: An optional `tff.simulation.baselines.ClientSpec`
      specifying how to preprocess evaluation client data. If set to `None`, the
      evaluation datasets will use a batch size of 64 with no extra
      preprocessing.
    cache_dir: An optional directory to cache the downloadeded datasets. If
      `None`, they will be cached to `~/.tff/`.
    use_synthetic_data: A boolean indicating whether to use synthetic EMNIST
      data. This option should only be used for testing purposes, in order to
      avoid downloading the entire EMNIST dataset.
    aggregated_kmeans: Whether to use aggregated k-means. If set to `True`, we
      will create a dual branch model, and use k-means based on the feautres to
      select branches in the forward pass.
    label_smooth_w: Weight of label smoothing regularization on the unselected
      branch. Only effective when `aggregated_kmeans = True`.
    label_smooth_eps: Epsilon of the label smoothing for the unselected branch.
      The value should be within 0 to 1, where 1 enforces the prediction to be
      uniform on all labels, and 0 falls back to cross entropy loss on one-hot
      label. The label smoothing regularization is defined as
      `L_{CE}(g(x), (1 - epsilon) * y + epsilon * 1/n)`, where L_{CE} is the
      cross entropy loss, g(x) is the prediction, epsilon represents the
      smoothness. Only effective when `aggregated_kmeans = True`.
    batch_majority_voting: Whether to use batch-wise majority voting to select
      the branch during test time. If set to True, we select the branch
      according to the majority within the minibatch during inference.
      Otherwise, we select the branch for each sample. Only effective when
      `aggregated_kmeans = True`.

  Returns:
    A `dist_shift_task.DistShiftTask`.
  """

  if use_synthetic_data:
    synthetic_data = tff.simulation.datasets.emnist.get_synthetic()
    emnist_train = synthetic_data
    emnist_test = synthetic_data
  else:
    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(
        only_digits=False, cache_dir=cache_dir)
  return create_character_recognition_task_from_datasets(
      train_client_spec,
      eval_client_spec,
      emnist_train,
      emnist_test,
      label_smooth_w=label_smooth_w,
      label_smooth_eps=label_smooth_eps,
      batch_majority_voting=batch_majority_voting,
      aggregated_kmeans=aggregated_kmeans)
