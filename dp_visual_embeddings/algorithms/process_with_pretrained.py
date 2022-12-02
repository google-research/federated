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
"""Use pre-trained model to initialize a TFF training process."""
from collections.abc import Callable
from typing import Optional
from absl import logging

import tensorflow as tf
import tensorflow_federated as tff

from dp_visual_embeddings.models import keras_utils


def _load_from_keras_model(
    model_path: str,
    model_with_variables: keras_utils.EmbeddingModel) -> list[tf.Variable]:
  """Returns `keras_utils.EmbeddingModel` initialized from a saved keras model.
  """
  keras_model = tf.keras.models.load_model(model_path)
  src_vars = keras_model.weights
  dst_vars = model_with_variables.model.weights
  return tf.nest.map_structure(lambda v, w: v.assign(w), dst_vars, src_vars)


def build_process_with_pretrained(
    train_process: tff.learning.templates.LearningProcess,
    pretrained_model_fn: Callable[[], keras_utils.EmbeddingModel],
    pretrained_model_path: Optional[str] = None,
) -> tff.learning.templates.LearningProcess:
  """Uses a pretrained model to initialize the training process.

  Args:
    train_process: A TFF learning process for model training, for example, a
      variant of Federated Averaging.
    pretrained_model_fn: A no-arg function that returns a
      `keras_utils.EmbeddingModel`, where `keras_utils.EmbeddingModel.model`
      matches the structure of saved keras model from pretraining (if provided),
      and `keras_utils.EmbeddingModel.global_variables` matches the training
      model weights in `train_process`.
    pretrained_model_path: Optional path to model saved by Keras `model.save`.

  Returns:
    A process that uses a pretrained model to initialize TFF training.
  """
  if pretrained_model_path is None:
    return train_process
  logging.info('Loading pretrained model from %s', pretrained_model_path)

  @tff.tf_computation
  def load_global_model_weights(state):
    load_model = pretrained_model_fn()
    model_variables = _load_from_keras_model(pretrained_model_path, load_model)
    # TODO(b/227775900): explicit control_dependencies is necessary for TFF
    # computation without `tf.function`.
    with tf.control_dependencies(model_variables):
      return train_process.set_model_weights(state, load_model.global_variables)

  @tff.federated_computation
  def init_state():
    state = train_process.initialize()
    return tff.federated_map(load_global_model_weights, state)

  return tff.learning.templates.LearningProcess(
      initialize_fn=init_state,
      next_fn=train_process.next,
      get_model_weights=train_process.get_model_weights,
      set_model_weights=train_process.set_model_weights)


def build_fedavg_process_with_pretrained(
    train_process: tff.learning.templates.LearningProcess,
    pretrained_model_path: Optional[str] = None,
) -> tff.learning.templates.LearningProcess:
  """Uses a pretrained model to initialize the training process.

  Args:
    train_process: A TFF learning process for model training, for example, a
      variant of Federated Averaging.
    pretrained_model_path: Optional path to model saved by Keras `model.save`.

  Returns:
    A process that uses a pretrained model to initialize TFF training.
  """
  if pretrained_model_path is None:
    return train_process
  logging.info('Loading pretrained model from %s', pretrained_model_path)

  @tff.tf_computation
  def load_global_model_weights(state):
    keras_model = tf.keras.models.load_model(pretrained_model_path)
    model_weights = tff.learning.ModelWeights.from_model(keras_model)
    # TODO(b/227775900): explicit control_dependencies is necessary for TFF
    # computation without `tf.function`.
    with tf.control_dependencies(model_weights.trainable +
                                 model_weights.non_trainable):
      return train_process.set_model_weights(state, model_weights)

  @tff.federated_computation
  def init_state():
    state = train_process.initialize()
    return tff.federated_map(load_global_model_weights, state)

  return tff.learning.templates.LearningProcess(
      initialize_fn=init_state,
      next_fn=train_process.next,
      get_model_weights=train_process.get_model_weights,
      set_model_weights=train_process.set_model_weights)
