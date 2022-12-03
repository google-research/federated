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
"""Contains builder function for federated training tasks."""
import enum
from typing import Optional, Union

import tensorflow_federated as tff

from dp_visual_embeddings.models import build_model
from dp_visual_embeddings.tasks import emnist_task
from dp_visual_embeddings.tasks import inaturalist_task
from dp_visual_embeddings.tasks import landmark_task
from dp_visual_embeddings.tasks import task_utils


class TaskType(enum.Enum):
  LANDMARK = enum.auto()
  EMNIST = enum.auto()
  INAT = enum.auto()


def get_task_types():
  return list(TaskType)


def _check_positive(value: Union[int, float]):
  if value <= 0:
    raise ValueError(f'Got {value} for positive input.')


def _check_positive_or_none(value: Union[int, float, None]):
  if value is not None:
    _check_positive(value)


def configure_task(
    task_type: TaskType,
    *,
    model_backbone: build_model.ModelBackbone = build_model.ModelBackbone
    .MOBILENET2,
    # Training algorithms
    use_client_softmax: bool = True,
    dynamic_clients: Optional[int] = 5,
    trainable_conv: bool = True,
    # Training parameters
    client_batch_size: int,
    eval_batch_size: int,
    train_max_examples_per_client: Optional[int] = None,
    eval_max_examples_per_client: Optional[int] = None,
    client_epochs_per_round: int = 1,
    client_shuffle_buffer_size: Optional[int] = None
) -> task_utils.EmbeddingTask:
  """Returns a task that defines data and model in TFF simulation.

  Args:
    task_type: Which task (dataset and model) to use in the experiment. Must be
      one of choices available via the `get_task_types()` method.
    model_backbone: Selects between different convolutional "backbones:" the
      main part of the model.
    use_client_softmax: Whether to use client-based mapping to generate labels
      from identities for training. It is used to implement client-sampled
      softmax for embedding model.
    dynamic_clients: Dynamically merge clients to super clients for user-level
      DP.
    trainable_conv: Whether to train the weights of convolutional layers. Can be
      used to reduce the size of trainable parameters when set to False.
    client_batch_size: A positive integer for the batch size of local training
      on each client.
    eval_batch_size: A positive integer for the batch size in online eval.
    train_max_examples_per_client: An optional poistive integer to cap the
      maximum number of examples on each client in training. If None, all
      examples will be used.
    eval_max_examples_per_client: An optional poistive integer to cap the
      maximum number of examples on each client in online eval. If None, all
      examples will be used.
    client_epochs_per_round: A positive integer for the number of epochs of
      local training on each client.
    client_shuffle_buffer_size: Size of buffer used to shuffle examples in
      client data input pipeline. No shuffling will be done if 0. The default,
      if None, is `10 * client_batch_size`.
  """
  _check_positive(client_batch_size)
  _check_positive_or_none(train_max_examples_per_client)
  _check_positive_or_none(eval_max_examples_per_client)
  _check_positive(client_epochs_per_round)
  if client_shuffle_buffer_size is None:
    client_shuffle_buffer_size = 10 * client_batch_size
  train_client_spec = tff.simulation.baselines.ClientSpec(
      num_epochs=client_epochs_per_round,
      batch_size=client_batch_size,
      max_elements=train_max_examples_per_client,
      shuffle_buffer_size=client_shuffle_buffer_size)
  eval_client_spec = tff.simulation.baselines.ClientSpec(
      num_epochs=1,
      batch_size=eval_batch_size,
      max_elements=eval_max_examples_per_client,
      shuffle_buffer_size=1)
  if task_type == TaskType.LANDMARK:
    if model_backbone != build_model.ModelBackbone.MOBILENET2:
      raise ValueError('Can only use MobileNet on Landmark.')
    task = landmark_task.get_landmark_embedding_task(
        train_client_spec,
        eval_client_spec,
        use_client_softmax=use_client_softmax,
        dynamic_clients=dynamic_clients,
        trainable_conv=trainable_conv)
  elif task_type == TaskType.EMNIST:
    task = emnist_task.get_emnist_embedding_task(
        train_client_spec,
        eval_client_spec,
        model_backbone=model_backbone,
        use_client_softmax=use_client_softmax,
        dynamic_clients=dynamic_clients,
        trainable_conv=trainable_conv)
  elif task_type == TaskType.INAT:
    if model_backbone != build_model.ModelBackbone.MOBILENET2:
      raise ValueError('Can only use MobileNet on iNaturalist.')
    task = inaturalist_task.get_inat_embedding_task(
        train_client_spec,
        eval_client_spec,
        use_client_softmax=use_client_softmax,
        dynamic_clients=dynamic_clients,
        trainable_conv=trainable_conv)
  else:
    raise ValueError(
        'Unrecognized value for task_type argument {}'.format(task_type))
  return task
