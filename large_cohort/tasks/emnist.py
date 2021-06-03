# Copyright 2019, Google LLC.
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
"""Configures federated EMNIST classification tasks."""

import functools

import tensorflow as tf
import tensorflow_federated as tff

from large_cohort import simulation_specs
from utils.datasets import emnist_dataset
from utils.models import emnist_models

EMNIST_MODEL = 'cnn'
ONLY_DIGITS = False
EMNIST_TASK = 'digit_recognition'


def get_model_spec(seed: int = 0) -> simulation_specs.ModelSpec:
  """Configures a model for EMNIST classification."""
  keras_model_builder = functools.partial(
      emnist_models.create_conv_dropout_model,
      only_digits=ONLY_DIGITS,
      seed=seed)
  loss_builder = tf.keras.losses.SparseCategoricalCrossentropy
  metrics_builder = lambda: [tf.keras.metrics.SparseCategoricalAccuracy()]
  return simulation_specs.ModelSpec(
      keras_model_builder=keras_model_builder,
      loss_builder=loss_builder,
      metrics_builder=metrics_builder)


def get_data_spec(
    train_client_spec: simulation_specs.ClientSpec,
    eval_client_spec: simulation_specs.ClientSpec,
    use_synthetic_data: bool = False) -> simulation_specs.DataSpec:
  """Configures data for EMNIST classification.

  Args:
    train_client_spec: A `simulation_specs.ClientSpec` used to configure
      training clients.
    eval_client_spec: A `simulation_specs.ClientSpec` used to configure
      evaluation clients.
    use_synthetic_data: A boolean indicating whether to use synthetic data.
      Suitable for testing purposes.

  Returns:
    A `simulation_specs.DataSpec`.
  """
  if use_synthetic_data:
    synthetic_data = tff.simulation.datasets.emnist.get_synthetic()
    train_data = synthetic_data
    validation_data = synthetic_data
    test_data = synthetic_data
  else:
    train_data, test_data = tff.simulation.datasets.emnist.load_data(
        only_digits=ONLY_DIGITS)
    validation_data = None

  train_preprocess_fn = emnist_dataset.create_preprocess_fn(
      num_epochs=train_client_spec.num_epochs,
      batch_size=train_client_spec.batch_size,
      emnist_task=EMNIST_TASK)

  eval_preprocess_fn = emnist_dataset.create_preprocess_fn(
      num_epochs=eval_client_spec.num_epochs,
      batch_size=eval_client_spec.batch_size,
      shuffle_buffer_size=1,
      emnist_task=EMNIST_TASK)

  return simulation_specs.DataSpec(
      train_data=train_data,
      validation_data=validation_data,
      test_data=test_data,
      train_preprocess_fn=train_preprocess_fn,
      eval_preprocess_fn=eval_preprocess_fn)
