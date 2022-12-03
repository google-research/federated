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
"""Task utilities."""

import abc
import collections
from collections.abc import Callable
import os
from typing import Any, Optional
from absl import logging

import tensorflow as tf
import tensorflow_federated as tff

from dp_visual_embeddings import losses
from dp_visual_embeddings import metrics
from dp_visual_embeddings.models import embedding_model
from dp_visual_embeddings.models import keras_utils
from dp_visual_embeddings.tasks import task_data
from dp_visual_embeddings.utils import exporting_program_state_manager

ManagerFunctionType = Callable[[str, str, int, tff.program.CSVSaveMode],
                               tuple[tff.program.FileProgramStateManager,
                                     list[tff.program.ReleaseManager]]]


def _configure_release_managers(
    root_output_dir: str,
    experiment_name: str) -> list[tff.program.ReleaseManager]:
  """Configures checkpoint and metrics managers.

  Args:
    root_output_dir: A string representing the root output directory for the
      training simulation. All metrics and checkpoints will be logged to
      subdirectories of this directory.
    experiment_name: A unique identifier for the current training simulation,
      used to create appropriate subdirectories of `root_output_dir`.

  Returns:
    A list of `tff.program.ReleaseManager` instances.
  """
  summary_dir = os.path.join(root_output_dir, 'logdir', experiment_name)
  if tf.executing_eagerly():
    tensorboard_manager = tff.program.TensorBoardReleaseManager(summary_dir)

  logging_manager = tff.program.LoggingReleaseManager()

  logging.info('Writing...')
  logging.info('    TensorBoard summaries to: %s', summary_dir)
  return [logging_manager, tensorboard_manager]


class EmbeddingTask(object, metaclass=abc.ABCMeta):
  """Specification for an embedding  task.

  Attributes:
    name: The name of the task (e.g. 'EMNIST').
    federated_model_fn: A no-arg callable returning a `tff.learning.Model` used
      for the task. Note that `federated_model_fn().input_spec` must match
      `datasets.element_type_structure`.
    datasets: A `task_data.EmbeddingTaskDatasets` object specifying
      dataset-related aspects of the task, including training data and
      preprocessing functions.
    configure_managers: A callable returning TFF checkpoint and metrics
      managers.
  """

  def __init__(self, name: str, datasets: task_data.EmbeddingTaskDatasets):
    self.name = name
    self.datasets = datasets

    dataset_element_spec = self.datasets.element_type_structure
    if dataset_element_spec != self.model_input_spec:
      raise ValueError(
          'Dataset element spec and model input spec do not match.'
          'Found dataset element spec {}, but model input spec {}'.format(
              dataset_element_spec, self.model_input_spec))

  def configure_federated_checkpoint_manager(
      self, root_output_dir: str, experiment_name: str,
      export_fn: Callable[[Any, str], None],
      rounds_per_export: int) -> tff.program.FileProgramStateManager:
    """Configures manager for TFF program state checkpoints.

    Args:
      root_output_dir: A string representing the root output directory for the
        training simulation. All checkpoints will be logged to subdirectories of
        this directory.
      experiment_name: A unique identifier for the current training simulation,
        used to create appropriate subdirectories of `root_output_dir`.
      export_fn: A callable to convert and save the server state to a directory.
        This is meant to handle converting the TFF `ServerState` to the
        exportable format, such as a Keras model, and saving it to the directory
        specified in the second argument.
      rounds_per_export: Sets the interval between exports.

    Returns:
      The `ProgramStateManager` to checkpoint TFF server states.
    """
    program_state_dir = os.path.join(root_output_dir, 'checkpoints',
                                     experiment_name)
    logging.info('Writing checkpoints to: %s', program_state_dir)
    export_root_dir = os.path.join(root_output_dir, 'export', experiment_name)
    logging.info('Writing exported models to: %s', export_root_dir)
    return exporting_program_state_manager.ExportingProgramStateManager(
        program_state_dir, export_fn, rounds_per_export, export_root_dir)

  def configure_release_managers(
      self, root_output_dir: str,
      experiment_name: str) -> list[tff.program.ReleaseManager]:
    """Configures metrics release managers.

    Args:
      root_output_dir: A string representing the root output directory for the
        training simulation. All metrics and checkpoints will be logged to
        subdirectories of this directory.
      experiment_name: A unique identifier for the current training simulation,
        used to create appropriate subdirectories of `root_output_dir`.

    Returns:
      A `tff.program.FileProgramStateManager`, and a list of
      `tff.program.ReleaseManager` instances.
    """
    return _configure_release_managers(
        root_output_dir=root_output_dir, experiment_name=experiment_name)

  @abc.abstractmethod
  def keras_model_fn(self) -> tf.keras.Model:
    """Builds the keras model for use in training.

    The embedding model typically differs between training-time and test-time.
    This variant will be used for centralized training, or to be wrapped in a
    TFF model for federated training.

    Returns:
      The training-mode Keras model.
    """
    raise NotImplementedError('Must be implemented in subclasses.')

  @abc.abstractmethod
  def keras_model_and_variables_fn(
      self,
      model_output_size: Optional[int] = None) -> keras_utils.EmbeddingModel:
    """Returns the keras model and variables for use in (partial) training.

    The embedding model will typically differ between training-time and
    test-time. This variant will be used for centralized training, or to be
    wrapped in a TFF model for federated training.

    The extra returned variables indicated `global_variables` to be federated,
    and `client_variables` that are only used locally on clients.

    Args:
      model_output_size: Number of outputs/labels for the last layer of the
        training model. Useful when the pretraining and TFF models use different
        last layers.

    Returns:
      A tuple of (training-mode Keras model, global_variables, client_variables)
    """
    raise NotImplementedError('Must be implemented in subclasses.')

  @property
  @abc.abstractmethod
  def model_input_spec(self) -> collections.OrderedDict[str, Any]:
    """Specifies the shapes and types of input fields for the model."""
    raise NotImplementedError('Must be implemented in subclasses.')

  def get_loss(
      self,
      reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.AUTO
  ) -> tf.keras.losses.Loss:
    return losses.EmbeddingLoss(reduction=reduction)

  def get_metrics(self) -> list[tf.keras.metrics.Metric]:
    return [
        metrics.EmbeddingCategoricalAccuracy(),
        metrics.EmbeddingRecallAtFAR(far=1e-3, name='recall_at_far_1e-3'),
        metrics.EmbeddingRecallAtFAR(far=0.1, name='recall_at_far_0p1'),
    ]

  def federated_model_fn(self) -> tff.learning.Model:
    """Returns a TFF model for federated training for this task."""
    return tff.learning.from_keras_model(
        keras_model=self.keras_model_fn(),
        loss=self.get_loss(),
        input_spec=self.model_input_spec,
        metrics=self.get_metrics())

  def embedding_model_fn(self) -> embedding_model.Model:
    """Returns a TFF model for federated (partial) training for this task.

    The `client_variables` for the head of embedding model are only used on
    local clients; the `global_variables` for the features extractors are
    federated.
    """
    model_with_variables = self.keras_model_and_variables_fn()
    return keras_utils.from_keras_model(
        keras_model=model_with_variables.model,
        global_variables=model_with_variables.global_variables,
        client_variables=model_with_variables.client_variables,
        loss=self.get_loss(),
        input_spec=self.model_input_spec,
        metrics=self.get_metrics())

  @property
  @abc.abstractmethod
  def inference_model(self) -> tf.keras.Model:
    """Returns a variant of the model for use in eval & inference.

    The embedding model typically differs between training-time and test-time.
    This variant will be used for centralized eval and export for inference.

    Returns:
      The inference/eval-mode Keras model.
    """
    raise NotImplementedError('Must be implemented in subclasses.')
