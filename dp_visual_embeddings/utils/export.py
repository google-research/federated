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
"""Code to export a trained model."""

import array
from collections.abc import Callable, Iterable
from typing import Optional, Union

from absl import logging
import tensorflow as tf
import tensorflow_federated as tff


def _name_weights(variables: Iterable[tf.Variable],
                  weights: Iterable[array.array]) -> dict[str, array.array]:
  return dict(
      tf.nest.map_structure(lambda var, weight: (var.name, weight), variables,
                            weights))


def _var_dict(var_list: Iterable[tf.Variable]) -> dict[str, tf.Variable]:
  """dict mapping from each variable's name to itself."""
  return {var.name: var for var in var_list}


def _assign_where_present(dst_var_dict: dict[str, tf.Variable],
                          src_weight_dict: dict[str, Union[array.array,
                                                           tf.Variable]]):
  """Assigns weights to variables where a key is present in both dicts."""
  assign_ops = []
  for name, src_weight in src_weight_dict.items():
    if name in dst_var_dict:
      assign_ops.append(dst_var_dict[name].assign(src_weight))
  return assign_ops


def export_state(model_fn: Callable[[], tff.learning.Model],
                 model_weights: tff.learning.ModelWeights,
                 export_model: tf.keras.Model,
                 export_dir: Optional[str] = None) -> tf.keras.Model:
  """Saves the model with given weights as SavedModel.

  Given the results of a federated optimization in `state`, this will save the
  weights so they can be used for inference in the graph provided by
  `export_model`. Weights are assigned to the export model where variables of
  the same name are present in both the `tff.learning.Model` returned by
  `model_fn` and `export_model`, to allow for differences in model structure
  between training time and test time.

  Args:
    model_fn: A no-arg function returns the model.
    model_weights: The trained model weights from TFF simulation.
    export_model: A keras model, used to write the `SavedModel` with ths same
      graph and weights from `state`.
    export_dir: Location to write the `SavedModel`. If `None`, `export_model`
      will be returned without being saved on hard disk.

  Returns:
    The `export_model`, to which `model_weights` have been assigned to
    the variables.
  """

  # Instantiate tff.learning.Model to get the variable names corresponding to
  # the weights in the server state.
  tff_model = model_fn()
  trainable_weights = _name_weights(tff_model.trainable_variables,
                                    model_weights.trainable)
  non_trainable_weights = _name_weights(tff_model.non_trainable_variables,
                                        model_weights.non_trainable)
  all_weights = trainable_weights
  all_weights.update(non_trainable_weights)

  export_var_dict = _var_dict(export_model.weights)
  assign_ops = _assign_where_present(export_var_dict, all_weights)
  if len(assign_ops) < len(export_var_dict):
    logging.info('weights names %s', all_weights.keys())
    logging.info('shapes %s', [tf.shape(x) for x in all_weights.values()])
    logging.info('var_dict names %s', export_var_dict.keys())
    logging.info('shapes %s', [tf.shape(x) for x in export_var_dict.values()])
    raise AssertionError('Cannot export all variables.')
  if export_dir:
    logging.info('Saving averaged model to \"%s\"...', export_dir)
    export_model.save(export_dir)
  return export_model


def export_keras_model(train_model: tf.keras.Model,
                       export_model: tf.keras.Model,
                       export_dir: Optional[str] = None) -> tf.keras.Model:
  """Exports a model given as a `tf.keras.Model`.

  Allows for different model architectures, where the model to be exported does
  not have the same set of variables as the training-time model. This will
  assign variables that have matching names between `train_model` and
  `export_model`, then save the export-form model.

  Args:
    train_model: The training-mode model, with variables set to the values to be
      exported.
    export_model: The export/inference-mode model.
    export_dir: Directory in which to write the `SavedModel`. Will skip writing
      out the `SavedModel` if this is None.

  Returns:
    The `export_model` with updated variable values.
  """
  train_var_dict = _var_dict(train_model.weights)
  export_var_dict = _var_dict(export_model.weights)
  _assign_where_present(export_var_dict, train_var_dict)
  if export_dir:
    export_model.save(export_dir)
  return export_model
