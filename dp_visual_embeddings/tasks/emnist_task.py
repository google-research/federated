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
"""Library providing methods for working with the EMNIST dataset and tasks."""

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

_NUM_LABELS = 62
_IMAGE_SHAPE = (28, 28, 1)
_EMBEDDING_DIM_SIZE = 128
_IMAGE_KEY = 'pixels'
_IDENTITY_KEY = 'label'
_CLIENT_ID_SEP = ';'
_TRAIN_IND = 'TR'
_TEST_IND = 'TE'


def _reshape_images(element):
  return collections.OrderedDict(
      # The EMNIST pixels are in [0, 1]
      pixels=tf.expand_dims(element['pixels'] * 255., axis=-1),
      label=tf.cast(element['label'], dtype=tf.int32))


def _get_emnist_embedding_task_datasets(
    train_preprocess_spec: tff.simulation.baselines.ClientSpec,
    eval_preprocess_spec: tff.simulation.baselines.ClientSpec,
    *,
    use_client_softmax: bool = True,
    dynamic_clients: int = 1,
) -> EmbeddingTaskDatasets:
  """Provides EmbeddingTaskDatasets object for the EMNIST dataset."""
  del use_client_softmax
  train_data_by_user, test_data_by_user = tff.simulation.datasets.emnist.load_data(
      only_digits=False)

  client_ids = (
      [x + _CLIENT_ID_SEP + _TRAIN_IND for x in train_data_by_user.client_ids] +
      [x + _CLIENT_ID_SEP + _TEST_IND for x in test_data_by_user.client_ids])

  @tf.function
  def merge_dataset_fn(client_id: str) -> tf.data.Dataset:
    split_client_id = tf.strings.split(client_id, _CLIENT_ID_SEP)
    base_client_id = split_client_id[0]
    client_type = split_client_id[-1]
    if tf.math.equal(client_type, _TRAIN_IND):
      ds = train_data_by_user.serializable_dataset_fn(base_client_id)
    else:
      ds = test_data_by_user.serializable_dataset_fn(base_client_id)
    return ds.map(_reshape_images)

  @tf.function
  def serializable_train_dataset_fn(client_id: str) -> tf.data.Dataset:
    ds = merge_dataset_fn(client_id)
    return ds.filter(lambda x: x[_IDENTITY_KEY] < 36)

  @tf.function
  def serializable_test_dataset_fn(client_id: str) -> tf.data.Dataset:
    ds = merge_dataset_fn(client_id)
    return ds.filter(lambda x: x[_IDENTITY_KEY] > 35)

  train_clients_data = (
      tff.simulation.datasets.ClientData.from_clients_and_tf_fn(
          client_ids, serializable_dataset_fn=serializable_train_dataset_fn))

  test_clients_data = (
      tff.simulation.datasets.ClientData.from_clients_and_tf_fn(
          client_ids, serializable_dataset_fn=serializable_test_dataset_fn))

  train_preprocess_fn = tff_input.get_integer_label_preprocess_fn(
      train_preprocess_spec,
      image_shape=_IMAGE_SHAPE,
      resize_images=False,
      image_key=_IMAGE_KEY,
      identity_key=_IDENTITY_KEY)
  validation_preprocess_fn = tff_input.get_integer_label_preprocess_fn(
      eval_preprocess_spec,
      image_shape=_IMAGE_SHAPE,
      resize_images=False,
      image_key=_IMAGE_KEY,
      identity_key=_IDENTITY_KEY)
  test_preprocess_fn = tff_input.get_integer_label_preprocess_fn(
      eval_preprocess_spec,
      image_shape=_IMAGE_SHAPE,
      resize_images=False,
      image_key=_IMAGE_KEY,
      identity_key=_IDENTITY_KEY)

  return EmbeddingTaskDatasets(
      train_data=train_clients_data,
      # Use the test clients as validation clients. This will condider the
      # pairs of images within each client in the sampled virtual clients,
      # which can be a proxy of the centralized evaluation of all image pairs.
      validation_data=test_clients_data,
      test_data=test_clients_data.create_tf_dataset_from_all_clients(seed=42),
      train_preprocess_fn=train_preprocess_fn,
      validation_preprocess_fn=validation_preprocess_fn,
      test_preprocess_fn=test_preprocess_fn,
      dynamic_clients=dynamic_clients)


class EMNISTTask(task_utils.EmbeddingTask):
  """Variant of the embedding model task for the EMNIST training."""

  def __init__(self,
               datasets: task_data.EmbeddingTaskDatasets,
               model_backbone: build_model.ModelBackbone = build_model
               .ModelBackbone.LENET,
               use_client_softmax: bool = True,
               dynamic_clients: int = 1,
               trainable_conv: bool = True):
    self._dynamic_clients = dynamic_clients
    self._model_backbone = model_backbone
    self._use_client_softmax = use_client_softmax
    self._trainable_conv = trainable_conv
    super().__init__('EMNIST', datasets)

  @property
  def model_input_spec(self) -> collections.OrderedDict[str, Any]:
    image_shape = (None,) + _IMAGE_SHAPE
    return collections.OrderedDict(
        x=collections.OrderedDict(
            images=tf.TensorSpec(
                shape=image_shape, dtype=tf.float32, name=None)),
        y=collections.OrderedDict(
            identity_names=tf.TensorSpec(
                shape=(None,), dtype=tf.string, name=None),
            identity_indices=tf.TensorSpec(
                shape=(None,), dtype=tf.int32, name=None)))

  def keras_model_and_variables_fn(
      self,
      model_output_size: Optional[int] = None) -> keras_utils.EmbeddingModel:
    if model_output_size is None:
      model_output_size = _NUM_LABELS
    return build_model.classification_training_model(
        model_backbone=self._model_backbone,
        input_shape=_IMAGE_SHAPE,
        num_identities=model_output_size,
        embedding_dim_size=_EMBEDDING_DIM_SIZE,
        trainable_conv=self._trainable_conv)

  def keras_model_fn(self) -> tf.keras.Model:
    return self.keras_model_and_variables_fn().model

  @property
  def inference_model(self) -> tf.keras.Model:
    return build_model.embedding_model(
        model_backbone=self._model_backbone,
        input_shape=_IMAGE_SHAPE,
        embedding_dim_size=_EMBEDDING_DIM_SIZE,
        trainable_conv=self._trainable_conv)


def get_emnist_embedding_task(
    train_preprocess_spec: tff.simulation.baselines.ClientSpec,
    eval_preprocess_spec: tff.simulation.baselines.ClientSpec,
    *,
    use_client_softmax: bool = True,
    dynamic_clients: int = 1,
    model_backbone: build_model.ModelBackbone = build_model.ModelBackbone.LENET,
    trainable_conv: bool = True) -> task_utils.EmbeddingTask:
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
    model_backbone: Selects between different convolutional "backbones:" the
      main part of the model.
    trainable_conv: Whether to train the weights of convolutional layers. Can be
      used to reduce the size of trainable parameters when set to False.

  Returns:
    A `task_utils.EmbeddingTask` with defined model and datasets.
  """
  datasets = _get_emnist_embedding_task_datasets(
      train_preprocess_spec,
      eval_preprocess_spec,
      use_client_softmax=use_client_softmax,
      dynamic_clients=dynamic_clients)

  return EMNISTTask(
      datasets=datasets,
      model_backbone=model_backbone,
      use_client_softmax=use_client_softmax,
      dynamic_clients=dynamic_clients,
      trainable_conv=trainable_conv)
