# Copyright 2021, Google LLC.
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
"""Library for creating periodic distribution shift tasks on CIFAR-100."""

# TODO(b/193904908): add unit tests.

import enum
from typing import Optional, Tuple, Union

import tensorflow as tf
import tensorflow_federated as tff

from periodic_distribution_shift.datasets import cifar_classification_preprocessing
from periodic_distribution_shift.models import dual_branch_resnet_models
from periodic_distribution_shift.models import keras_utils_dual_branch_kmeans
from periodic_distribution_shift.tasks import dist_shift_task
from periodic_distribution_shift.tasks import dist_shift_task_data
from utils.datasets import cifar10_dataset


class ResnetModel(enum.Enum):
  """Enum for ResNet classification models."""
  RESNET18 = 'resnet18'
  RESNET34 = 'resnet34'
  RESNET50 = 'resnet50'
  RESNET101 = 'resnet101'
  RESNET152 = 'resnet152'


_NUM_CLASSES = 110
_RESNET_MODELS = [e.value for e in ResnetModel]
DEFAULT_CROP_HEIGHT = 24
DEFAULT_CROP_WIDTH = 24


def _get_resnet_model(
    model_id: Union[str, ResnetModel],
    input_shape: Tuple[int, int, int],
    dual_branch: bool = False,
) -> tf.keras.Model:
  """Constructs a `tf.keras.Model` for digit recognition."""
  try:
    model_enum = ResnetModel(model_id)
  except ValueError:
    raise ValueError('The model argument must be one of {}, found {}'.format(
        ResnetModel, model_id))

  if model_enum == ResnetModel.RESNET18:
    keras_model_fn = dual_branch_resnet_models.create_resnet18
  elif model_enum == ResnetModel.RESNET34:
    keras_model_fn = dual_branch_resnet_models.create_resnet34
  elif model_enum == ResnetModel.RESNET50:
    keras_model_fn = dual_branch_resnet_models.create_resnet50
  elif model_enum == ResnetModel.RESNET101:
    keras_model_fn = dual_branch_resnet_models.create_resnet101
  elif model_enum == ResnetModel.RESNET152:
    keras_model_fn = dual_branch_resnet_models.create_resnet152
  else:
    raise ValueError('The model id must be one of {}, found {}'.format(
        _RESNET_MODELS, model_enum))
  return keras_model_fn(
      input_shape=input_shape,
      num_classes=_NUM_CLASSES,
      dual_branch=dual_branch)


def create_image_classification_task_with_datasets(
    train_client_spec: tff.simulation.baselines.ClientSpec,
    eval_client_spec: Optional[tff.simulation.baselines.ClientSpec],
    model_id: Union[str, ResnetModel],
    crop_height: int,
    crop_width: int,
    cache_dir: Optional[str] = None,
    aggregated_kmeans: bool = False,
    label_smooth_w: float = 0.,
    label_smooth_eps: float = 0.,
    batch_majority_voting: bool = False,
) -> dist_shift_task.DistShiftTask:
  """Creates a baseline task for image classification on CIFAR-100.

  Args:
    train_client_spec: A `tff.simulation.baselines.ClientSpec` specifying how to
      preprocess train client data.
    eval_client_spec: An optional `tff.simulation.baselines.ClientSpec`
      specifying how to preprocess evaluation client data. If set to `None`, the
      evaluation datasets will use a batch size of 64 with no extra
      preprocessing.
    model_id: A string identifier for a digit recognition model. Must be one of
      `resnet18`, `resnet34`, `resnet50`, `resnet101` and `resnet152. These
      correspond to various ResNet architectures. Unlike standard ResNet
      architectures though, the batch normalization layers are replaced with
      group normalization.
    crop_height: An integer specifying the desired height for cropping images.
      Must be between 1 and 32 (the height of uncropped CIFAR-100 images). By
      default, this is set to
      `tff.simulation.baselines.cifar100.DEFAULT_CROP_HEIGHT`.
    crop_width: An integer specifying the desired width for cropping images.
      Must be between 1 and 32 (the width of uncropped CIFAR-100 images). By
      default this is set to
      `tff.simulation.baselines.cifar100.DEFAULT_CROP_WIDTH`.
    cache_dir: An optional directory to cache the downloadeded datasets. If
      `None`, they will be cached to `~/.tff/`.
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
  if crop_height < 1 or crop_width < 1 or crop_height > 32 or crop_width > 32:
    raise ValueError('The crop_height and crop_width must be between 1 and 32.')
  crop_shape = (crop_height, crop_width, 3)

  if eval_client_spec is None:
    eval_client_spec = tff.simulation.baselines.ClientSpec(
        num_epochs=1, batch_size=64, shuffle_buffer_size=1)

  train_cifar100, eval_cifar100 = tff.simulation.datasets.cifar100.load_data(
      cache_dir)
  _, eval_cifar10 = cifar10_dataset.load_cifar10_federated(num_clients=500)

  train_cifar100_prep = cifar_classification_preprocessing.create_preprocess_fn(
      train_client_spec, crop_shape=crop_shape, is_cifar10=False)

  train_cifar100 = train_cifar100.preprocess(train_cifar100_prep)

  eval_cifar10_prep = cifar_classification_preprocessing.create_preprocess_fn(
      eval_client_spec, crop_shape=crop_shape, is_cifar10=True)
  eval_cifar100_prep = cifar_classification_preprocessing.create_preprocess_fn(
      eval_client_spec, crop_shape=crop_shape, is_cifar10=False)
  eval_cifar10 = eval_cifar10.preprocess(eval_cifar10_prep)
  eval_cifar10 = eval_cifar10.create_tf_dataset_from_all_clients()
  eval_cifar100 = eval_cifar100.preprocess(eval_cifar100_prep)
  eval_cifar100 = eval_cifar100.create_tf_dataset_from_all_clients()
  eval_set = eval_cifar10.concatenate(eval_cifar100)

  # Create the splits for validation.
  dataset_dict = {}
  dataset_dict['full'] = eval_set
  dataset_dict['cifar10'] = eval_cifar10
  dataset_dict['cifar100'] = eval_cifar100

  task_datasets = dist_shift_task_data.DistShiftDatasets(
      train_data=train_cifar100,
      test_data=dataset_dict['full'],
      validation_data_dict=dataset_dict,
  )

  def model_fn() -> tff.learning.Model:
    if aggregated_kmeans:
      return keras_utils_dual_branch_kmeans.from_keras_model(
          keras_model=_get_resnet_model(model_id, crop_shape, dual_branch=True),
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
          keras_model=_get_resnet_model(model_id, crop_shape),
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          input_spec=task_datasets.element_type_structure,
          metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

  return dist_shift_task.DistShiftTask(task_datasets, model_fn)


def create_image_classification_task(
    train_client_spec: tff.simulation.baselines.ClientSpec,
    eval_client_spec: Optional[tff.simulation.baselines.ClientSpec] = None,
    model_id: Union[str, ResnetModel] = 'resnet18',
    crop_height: int = DEFAULT_CROP_HEIGHT,
    crop_width: int = DEFAULT_CROP_WIDTH,
    cache_dir: Optional[str] = None,
    aggregated_kmeans: bool = False,
    label_smooth_w: float = 0.,
    label_smooth_eps: float = 0.,
    batch_majority_voting: bool = False,
) -> dist_shift_task.DistShiftTask:
  """Creates a baseline task for image classification on CIFAR-100.

  The goal of the task is to minimize the sparse categorical crossentropy
  between the output labels of the model and the true label of the image.

  Args:
    train_client_spec: A `tff.simulation.baselines.ClientSpec` specifying how to
      preprocess train client data.
    eval_client_spec: An optional `tff.simulation.baselines.ClientSpec`
      specifying how to preprocess evaluation client data. If set to `None`, the
      evaluation datasets will use a batch size of 64 with no extra
      preprocessing.
    model_id: A string identifier for a digit recognition model. Must be one of
      `resnet18`, `resnet34`, `resnet50`, `resnet101` and `resnet152. These
      correspond to various ResNet architectures. Unlike standard ResNet
      architectures though, the batch normalization layers are replaced with
      group normalization.
    crop_height: An integer specifying the desired height for cropping images.
      Must be between 1 and 32 (the height of uncropped CIFAR-100 images). By
      default, this is set to
      `tff.simulation.baselines.cifar100.DEFAULT_CROP_HEIGHT`.
    crop_width: An integer specifying the desired width for cropping images.
      Must be between 1 and 32 (the width of uncropped CIFAR-100 images). By
      default this is set to
      `tff.simulation.baselines.cifar100.DEFAULT_CROP_WIDTH`.
    cache_dir: An optional directory to cache the downloadeded datasets. If
      `None`, they will be cached to `~/.tff/`.
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

  return create_image_classification_task_with_datasets(
      train_client_spec,
      eval_client_spec,
      model_id,
      crop_height,
      crop_width,
      label_smooth_w=label_smooth_w,
      batch_majority_voting=batch_majority_voting,
      label_smooth_eps=label_smooth_eps,
      aggregated_kmeans=aggregated_kmeans,
      cache_dir=cache_dir)
