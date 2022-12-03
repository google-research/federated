# Copyright 2022, Google LLC.
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
"""Library providing methods for working with the Landmark dataset and tasks."""

import collections
from typing import Any, Optional

import tensorflow as tf
import tensorflow_federated as tff

from dp_visual_embeddings import tff_input
from dp_visual_embeddings.models import build_model
from dp_visual_embeddings.models import keras_utils
from dp_visual_embeddings.tasks import task_data
from dp_visual_embeddings.tasks import task_utils
from dp_visual_embeddings.tasks.task_data import EmbeddingTaskDatasets


# Dataset stats: 2,028 landmarks, 1262 users and 164,172 images.
_LANDMARK_NUM_LABELS = 2028
_LANDMARK_MAX_NUM_LABELS_PER_USER = 399

_LANDMARK_IMAGE_SHAPE = (224, 224, 3)
_EMBEDDING_DIM_SIZE = 128
_IMAGE_KEY = 'image/decoded'
_IDENTITY_KEY = 'class'


def _get_landmark_embedding_task_datasets(
    train_preprocess_spec: tff.simulation.baselines.ClientSpec,
    eval_preprocess_spec: tff.simulation.baselines.ClientSpec,
    *,
    use_client_softmax: bool = True,
    dynamic_clients: int = 1,
    gld23k: bool = False,
) -> EmbeddingTaskDatasets:
  """Provides EmbeddingTaskDatasets object for the Landmark dataset."""
  del use_client_softmax
  train_data_by_user, test_dataset = tff.simulation.datasets.gldv2.load_data(
      gld23k=gld23k)

  @tf.function
  def serializable_test_dataset_fn(_: str) -> tf.data.Dataset:
    _, test_dataset = tff.simulation.datasets.gldv2.load_data(gld23k=gld23k)
    return test_dataset

  train_preprocess_fn = tff_input.get_integer_label_preprocess_fn(
      train_preprocess_spec,
      image_shape=_LANDMARK_IMAGE_SHAPE,
      resize_images=True,
      image_key=_IMAGE_KEY,
      identity_key=_IDENTITY_KEY)
  validation_preprocess_fn = tff_input.get_integer_label_preprocess_fn(
      eval_preprocess_spec,
      image_shape=_LANDMARK_IMAGE_SHAPE,
      resize_images=True,
      image_key=_IMAGE_KEY,
      identity_key=_IDENTITY_KEY)
  test_preprocess_fn = tff_input.get_integer_label_preprocess_fn(
      eval_preprocess_spec,
      image_shape=_LANDMARK_IMAGE_SHAPE,
      resize_images=True,
      image_key=_IMAGE_KEY,
      identity_key=_IDENTITY_KEY)

  return EmbeddingTaskDatasets(
      train_data=train_data_by_user,
      validation_data=train_data_by_user,
      test_data=test_dataset,
      train_preprocess_fn=train_preprocess_fn,
      validation_preprocess_fn=validation_preprocess_fn,
      test_preprocess_fn=test_preprocess_fn,
      dynamic_clients=dynamic_clients)


class LandmarkTask(task_utils.EmbeddingTask):
  """Variant of the embedding model.  task for the Landmark training."""

  def __init__(self,
               datasets: task_data.EmbeddingTaskDatasets,
               use_client_softmax: bool = True,
               dynamic_clients: int = 1,
               trainable_conv: bool = True):
    self._dynamic_clients = dynamic_clients
    self._use_client_softmax = use_client_softmax
    self._trainable_conv = trainable_conv
    super().__init__('Landmark', datasets)

  @property
  def model_input_spec(self) -> collections.OrderedDict[str, Any]:
    image_shape = (None,) + _LANDMARK_IMAGE_SHAPE
    return collections.OrderedDict(
        x=collections.OrderedDict(
            images=tf.TensorSpec(
                shape=image_shape, dtype=tf.float32, name=None)),
        y=collections.OrderedDict(
            identity_names=tf.TensorSpec(
                shape=(None,), dtype=tf.string, name=None),
            identity_indices=tf.TensorSpec(
                shape=(None,), dtype=tf.int64, name=None)))

  def keras_model_and_variables_fn(
      self,
      model_output_size: Optional[int] = None) -> keras_utils.EmbeddingModel:
    if model_output_size is None:
      model_output_size = _LANDMARK_NUM_LABELS
    return build_model.classification_training_model(
        build_model.ModelBackbone.MOBILENET2,
        input_shape=_LANDMARK_IMAGE_SHAPE,
        num_identities=model_output_size,
        embedding_dim_size=_EMBEDDING_DIM_SIZE,
        trainable_conv=self._trainable_conv)

  def keras_model_fn(self) -> tf.keras.Model:
    return self.keras_model_and_variables_fn().model

  @property
  def inference_model(self) -> tf.keras.Model:
    return build_model.embedding_model(
        build_model.ModelBackbone.MOBILENET2,
        input_shape=_LANDMARK_IMAGE_SHAPE,
        embedding_dim_size=_EMBEDDING_DIM_SIZE,
        trainable_conv=self._trainable_conv)


def get_landmark_embedding_task(
    train_preprocess_spec: tff.simulation.baselines.ClientSpec,
    eval_preprocess_spec: tff.simulation.baselines.ClientSpec,
    *,
    use_client_softmax: bool = True,
    dynamic_clients: int = 1,
    trainable_conv: bool = True,
    gld23k: bool = False) -> task_utils.EmbeddingTask:
  """Provides BaselineTask object for the Landmark data.

  Args:
    train_preprocess_spec: A `tff.simulation.baselines.ClientSpec` specifying
      how to preprocess train client data.
    eval_preprocess_spec: An optional `tff.simulation.baselines.ClientSpec`
      specifying how to preprocess evaluation client data.
    use_client_softmax: A boolean indicating whether to use client-based mapping
      to generate labels from landmark labels for training. It is used to
      implement client-sampled softmax for embedding model. .
    dynamic_clients: Dynamically merge clients to super clients for user-level
      DP.
    trainable_conv: Whether to train the weights of convolutional layers. Can be
      used to reduce the size of trainable parameters when set to False.
    gld23k: Whether to load a smaller landmark dataset, for testing purposes.

  Returns:
    A `task_utils.EmbeddingTask` with defined model and datasets.
  """
  datasets = _get_landmark_embedding_task_datasets(
      train_preprocess_spec,
      eval_preprocess_spec,
      use_client_softmax=use_client_softmax,
      dynamic_clients=dynamic_clients,
      gld23k=gld23k)

  return LandmarkTask(
      datasets=datasets,
      use_client_softmax=use_client_softmax,
      dynamic_clients=dynamic_clients,
      trainable_conv=trainable_conv)
