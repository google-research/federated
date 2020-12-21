# Copyright 2020, Google LLC.
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
"""Centralized experiments on the Google Landmark datasets."""

from typing import Any, Mapping, Optional

import tensorflow as tf

from fedopt_guide.gld23k_mobilenet import dataset
from fedopt_guide.gld23k_mobilenet import mobilenet_v2
from utils import centralized_training_loop


def run_centralized(
    optimizer: tf.keras.optimizers.Optimizer,
    image_size: int,
    num_epochs: int,
    batch_size: int,
    num_groups: int = 8,
    dataset_type: dataset.DatasetType = dataset.DatasetType.GLD23K,
    experiment_name: str = 'centralized_gld23k',
    root_output_dir: str = '/tmp/fedopt_guide',
    dropout_prob: Optional[float] = None,
    hparams_dict: Optional[Mapping[str, Any]] = None,
    max_batches: Optional[int] = None):
  """Trains a MobileNetV2 on the Google Landmark datasets.

  Args:
    optimizer: A `tf.keras.optimizers.Optimizer` used to perform training.
    image_size: The height and width of images after preprocessing.
    num_epochs: The number of training epochs.
    batch_size: The batch size, used for train and test.
    num_groups: The number of groups in the GroupNorm layers of MobilenetV2.
    dataset_type: A `dataset.DatasetType` specifying which dataset is used for
      experiments.
    experiment_name: The name of the experiment. Part of the output directory.
    root_output_dir: The top-level output directory for experiment runs. The
      `experiment_name` argument will be appended, and the directory will
      contain tensorboard logs, metrics written as CSVs, and a CSV of
      hyperparameter choices (if `hparams_dict` is used).
    dropout_prob: Probability of setting a weight to zero in the dropout layer
      of MobilenetV2. Must be in the range [0, 1). Setting it to None (default)
      or zero means no dropout.
    hparams_dict: A mapping with string keys representing the hyperparameters
      and their values. If not None, this is written to CSV.
    max_batches: If set to a positive integer, datasets are capped to at most
      that many batches. If set to None or a nonpositive integer, the full
      datasets are used.
  """

  train_data, test_data = dataset.get_centralized_datasets(
      image_size=image_size, batch_size=batch_size, dataset_type=dataset_type)
  num_classes, _ = dataset.get_dataset_stats(dataset_type)

  if max_batches and max_batches >= 1:
    train_data = train_data.take(max_batches)
    test_data = test_data.take(max_batches)

  if dropout_prob and (dropout_prob < 0 or dropout_prob >= 1):
    raise ValueError(
        f'Expected a value in [0, 1) for `dropout_prob`, found {dropout_prob}.')

  model = mobilenet_v2.create_mobilenet_v2(
      input_shape=(image_size, image_size, 3),
      num_groups=num_groups,
      num_classes=num_classes,
      dropout_prob=dropout_prob)

  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
      optimizer=optimizer,
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

  centralized_training_loop.run(
      keras_model=model,
      train_dataset=train_data,
      validation_dataset=test_data,
      experiment_name=experiment_name,
      root_output_dir=root_output_dir,
      num_epochs=num_epochs,
      hparams_dict=hparams_dict)
