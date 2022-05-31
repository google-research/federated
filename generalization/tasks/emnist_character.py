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
"""Federated and Centralized EMNIST character classification library using TFF."""

import collections
import functools
from typing import Callable, Optional

from absl import flags
import tensorflow as tf
import tensorflow_federated as tff

from generalization.tasks import training_specs
from generalization.utils import client_data_utils
from generalization.utils import eval_metric_distribution
from generalization.utils import resnet_models
from generalization.utils import sql_client_data_utils
from generalization.utils import trainer_utils
from utils import utils_impl
from utils.datasets import emnist_dataset

_EMNIST_MODELS = [
    'cnn', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
]

_ELEMENT_SPEC = collections.OrderedDict([
    ('label', tf.TensorSpec(shape=(), dtype=tf.int32)),
    ('pixels', tf.TensorSpec(shape=(28, 28), dtype=tf.float32))
])

_INTRA_CLIENT_SHUFFLE_BUFFER_SIZE = 1000

FLAG_PREFIX = 'emnist_character_'

ClientData = tff.simulation.datasets.ClientData

with utils_impl.record_hparam_flags() as emnist_character_flags:
  flags.DEFINE_enum(
      FLAG_PREFIX + 'model', 'resnet18', _EMNIST_MODELS,
      'Which model to use. This can be a shallow convolutional model (cnn) with'
      'around 1M parameters, resnet18, resnet34, resnet50, resnet101, resnet152'
  )
  flags.DEFINE_boolean(
      FLAG_PREFIX + 'only_digits', False,
      'Whether to use digit-only version of EMNIST (with only 10 classes). '
      'If false, use the full dataset.')
  flags.DEFINE_boolean(
      FLAG_PREFIX + 'merge_case', False,
      'Whether to merge the 15 upper and lower case letters as proposed by '
      'NIST/EMNIST. Can be true only if only_digits is False.')


def create_conv_dropout_model(num_classes: int, seed: Optional[int] = None):
  """Convolutional model with dropout for EMNIST experiments.

  Args:
    num_classes: An integer representing the number of classes.
    seed: A random seed governing the model initialization and layer randomness.
      If not `None`, then the global random seed will be set before constructing
      the tensor initializer, in order to guarantee the same model is produced.

  Returns:
    A `tf.keras.Model`.
  """
  data_format = 'channels_last'
  if seed is not None:
    tf.random.set_seed(seed)

  model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(
          32,
          kernel_size=(3, 3),
          activation='relu',
          data_format=data_format,
          input_shape=(28, 28, 1),
          kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed)),
      tf.keras.layers.Conv2D(
          64,
          kernel_size=(3, 3),
          activation='relu',
          data_format=data_format,
          kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed)),
      tf.keras.layers.MaxPool2D(pool_size=(2, 2), data_format=data_format),
      tf.keras.layers.Dropout(0.25, seed=seed),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(
          128,
          activation='relu',
          kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed)),
      tf.keras.layers.Dropout(0.5, seed=seed),
      tf.keras.layers.Dense(
          num_classes,
          activation=tf.nn.softmax,
          kernel_initializer=tf.keras.initializers.GlorotNormal(seed=seed)),
  ])

  return model


def _parse_model(model: str, num_classes: int) -> Callable[[], tf.keras.Model]:
  """Parse the model description string to a keras model builder."""
  if model == 'cnn':
    keras_model_builder = functools.partial(
        create_conv_dropout_model, num_classes=num_classes)
  elif model in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
    keras_model_builder = functools.partial(
        getattr(resnet_models, f'create_{model}'),
        input_shape=(28, 28, 1),
        num_classes=num_classes)
  else:
    raise ValueError(
        'Cannot handle model flag [{!s}], must be one of {!s}.'.format(
            model, _EMNIST_MODELS))
  return keras_model_builder


def load_custom_emnist_client_data(sql_database: str) -> ClientData:
  """Load (un-splitted) EMNIST(-like) clientdata from sql database."""

  if sql_database is None:
    raise ValueError('sql_database cannot be None.')

  return sql_client_data_utils.load_parsed_sql_client_data(
      sql_database, element_spec=_ELEMENT_SPEC)


def _create_preprocess_fn(
    num_epochs: int,
    batch_size: int,
    merge_case: bool,
    shuffle_buffer_size: int = emnist_dataset.MAX_CLIENT_DATASET_SIZE,
    use_cache: bool = True,
    use_prefetch: bool = True,
) -> Callable[[tf.data.Dataset], tf.data.Dataset]:
  """Create a preprocessing function for EMNIST client datasets."""
  @tf.function
  def merge_mapping(elem):
    original_label_to_merged_label = tf.constant([
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
        12, 38, 39, 40, 41, 42, 18, 19, 20, 21, 22, 43, 24, 25, 44, 45, 28, 46,
        30, 31, 32, 33, 34, 35
    ])
    return collections.OrderedDict(
        label=original_label_to_merged_label[elem['label']],
        pixels=elem['pixels'])

  base_preprocess_fn = emnist_dataset.create_preprocess_fn(
      num_epochs=num_epochs,
      batch_size=batch_size,
      shuffle_buffer_size=shuffle_buffer_size)

  def preprocess_fn(dataset: tf.data.Dataset):
    if merge_case:
      dataset = dataset.map(merge_mapping)
    if use_cache:
      dataset = dataset.cache()
    dataset = base_preprocess_fn(dataset)
    if use_prefetch:
      dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

  return preprocess_fn  # pytype: disable=bad-return-type


def _metrics_builder_generic(tff_training=True):
  metrics_list = [tf.keras.metrics.SparseCategoricalAccuracy(name='acc')]
  if not tff_training:
    # Append loss to metrics unless using TFF training,
    # (in which case loss will be appended to metrics list by keras_utils).
    # This includes centralized training/evaluation and federated evaluation.
    metrics_list.append(
        tf.keras.metrics.SparseCategoricalCrossentropy(name='loss'))
  return metrics_list


def _loss_builder():
  return tf.keras.losses.SparseCategoricalCrossentropy()


class _EmnistCharacterTask():
  """Backend class for configuring centralized and federated emnist-character task."""

  def __init__(
      self,
      task_spec,
      *,  # Caller passes below args by name.
      model: str,
      only_digits: bool,
      merge_case: Optional[bool]):

    self._task_spec = task_spec
    self._only_digits = only_digits
    self._merge_case = merge_case

    if only_digits:
      if merge_case:
        raise ValueError('merge_case cannot be True if only_digits is True')
      num_classes = 10
    elif merge_case:
      num_classes = 47
    else:
      num_classes = 62

    self._keras_model_builder = _parse_model(model, num_classes)

    if task_spec.sql_database is not None:
      # The backend client_data_utils.canonical_three_way_partition_client_data
      # will run validity check.
      total_cd_orig = load_custom_emnist_client_data(task_spec.sql_database)
      (part_train_cd_raw, part_val_cd_raw, unpart_cd_raw
      ) = client_data_utils.canonical_three_way_partition_client_data(
          total_cd_orig,
          unpart_clients_proportion=task_spec.unpart_clients_proportion,
          train_val_ratio_intra_client=task_spec.train_val_ratio_intra_client,
          part_clients_subsampling_rate=task_spec.part_clients_subsampling_rate,
          include_unpart_train_for_val=task_spec.include_unpart_train_for_val,
          max_elements_per_client=task_spec.max_elements_per_client,
          shuffle_buffer_size=_INTRA_CLIENT_SHUFFLE_BUFFER_SIZE,
          seed=task_spec.shared_random_seed)

    else:
      if task_spec.train_val_ratio_intra_client is not None:
        raise ValueError(
            'train_val_ratio_intra_client must be None since TFF original '
            'EMNIST has provided a horizontal (intra-client) train/val split. '
            f'got {task_spec.train_val_ratio_intra_client}.')

      train_cd_orig, val_cd_orig = tff.simulation.datasets.emnist.load_data(
          only_digits=only_digits)

      (part_train_cd_raw, part_val_cd_raw, unpart_cd_raw
      ) = client_data_utils.construct_three_way_split_from_predefined_horizontal_split(
          train_cd_orig,
          val_cd_orig,
          unpart_clients_proportion=task_spec.unpart_clients_proportion,
          part_clients_subsampling_rate=task_spec.part_clients_subsampling_rate,
          include_unpart_train_for_val=task_spec.include_unpart_train_for_val,
          max_elements_per_client=task_spec.max_elements_per_client,
          seed=task_spec.shared_random_seed)

    eval_preprocess_fn = _create_preprocess_fn(
        num_epochs=1,
        batch_size=task_spec.eval_client_batch_size,
        merge_case=merge_case,
        # Set buffer to 1 to disable shuffling since is not necessary for eval.
        shuffle_buffer_size=1)

    self._part_train_cd_raw = part_train_cd_raw
    self._part_train_eval_cd = part_train_cd_raw.preprocess(eval_preprocess_fn)
    self._part_val_cd = part_val_cd_raw.preprocess(eval_preprocess_fn)
    self._unpart_cd = unpart_cd_raw.preprocess(eval_preprocess_fn)

  def _tff_model_builder(self) -> tff.learning.Model:
    return tff.learning.from_keras_model(
        keras_model=self._keras_model_builder(),
        input_spec=self._unpart_cd.element_type_structure,
        loss=_loss_builder(),
        metrics=_metrics_builder_generic(tff_training=True))

  def build_federated_runner_spec(self) -> training_specs.RunnerSpecFederated:
    """Configuring federated runner spec."""

    task_spec = self._task_spec

    train_preprocess_fn = _create_preprocess_fn(
        num_epochs=task_spec.client_epochs_per_round,
        batch_size=task_spec.client_batch_size,
        merge_case=self._merge_case,
        use_cache=True,
        use_prefetch=True)
    part_train_cd = self._part_train_cd_raw.preprocess(train_preprocess_fn)
    iterative_process = task_spec.iterative_process_builder(
        self._tff_model_builder)
    training_process = tff.simulation.compose_dataset_computation_with_iterative_process(
        part_train_cd.dataset_computation, iterative_process)
    client_ids_fn = functools.partial(
        tff.simulation.build_uniform_sampling_fn(
            part_train_cd.client_ids,
            replace=False,
            random_seed=task_spec.shared_random_seed),
        size=task_spec.train_clients_per_round)

    # We convert the output to a list (instead of an np.ndarray) so that it can
    # be used as input to the iterative process.
    client_sampling_fn = lambda x: list(client_ids_fn(x))
    training_process.get_model_weights = iterative_process.get_model_weights

    (part_train_eval_fn, part_val_fn, unpart_fn,
     _) = trainer_utils.create_federated_eval_fns(
         tff_model_builder=self._tff_model_builder,
         metrics_builder=functools.partial(
             _metrics_builder_generic, tff_training=False),
         part_train_eval_cd=self._part_train_eval_cd,
         part_val_cd=self._part_val_cd,
         unpart_cd=self._unpart_cd,
         test_cd=None,
         stat_fns=eval_metric_distribution.ALL_STAT_FNS,
         rounds_per_eval=task_spec.rounds_per_eval,
         part_clients_per_eval=task_spec.part_clients_per_eval,
         unpart_clients_per_eval=task_spec.unpart_clients_per_eval,
         test_clients_for_eval=task_spec.test_clients_for_eval,
         resample_eval_clients=task_spec.resample_eval_clients,
         eval_clients_random_seed=task_spec.shared_random_seed)

    return training_specs.RunnerSpecFederated(
        iterative_process=training_process,
        client_datasets_fn=client_sampling_fn,
        part_train_eval_fn=part_train_eval_fn,
        part_val_fn=part_val_fn,
        unpart_fn=unpart_fn,
        test_fn=None)

  def build_centralized_runner_spec(
      self) -> training_specs.RunnerSpecCentralized:
    """Configuring centralized runner spec."""

    task_spec = self._task_spec

    train_preprocess_fn = _create_preprocess_fn(
        num_epochs=1,
        batch_size=task_spec.batch_size,
        merge_case=self._merge_case,
        shuffle_buffer_size=task_spec.centralized_shuffle_buffer_size)

    train_dataset = train_preprocess_fn(
        client_data_utils.interleave_create_tf_dataset_from_all_clients(
            self._part_train_cd_raw, seed=task_spec.shared_random_seed))

    (part_train_eval_fn, part_val_fn, unpart_fn,
     _) = trainer_utils.create_centralized_eval_fns(
         tff_model_builder=self._tff_model_builder,
         metrics_builder=functools.partial(
             _metrics_builder_generic, tff_training=False),
         part_train_eval_cd=self._part_train_eval_cd,
         part_val_cd=self._part_val_cd,
         unpart_cd=self._unpart_cd,
         test_cd=None,
         stat_fns=eval_metric_distribution.ALL_STAT_FNS,
         part_clients_per_eval=task_spec.part_clients_per_eval,
         unpart_clients_per_eval=task_spec.unpart_clients_per_eval,
         test_clients_for_eval=task_spec.test_clients_for_eval,
         resample_eval_clients=task_spec.resample_eval_clients,
         eval_clients_random_seed=task_spec.shared_random_seed)

    keras_model = self._keras_model_builder()
    keras_model.compile(
        loss=_loss_builder(),
        optimizer=task_spec.optimizer,
        metrics=_metrics_builder_generic(tff_training=False))

    return training_specs.RunnerSpecCentralized(
        keras_model=keras_model,
        train_dataset=train_dataset,
        part_train_eval_fn=part_train_eval_fn,
        part_val_fn=part_val_fn,
        unpart_fn=unpart_fn,
        test_fn=None)


def configure_training_federated(
    task_spec: training_specs.TaskSpecFederated,
    *,  # Caller passes below args by name.
    model: str = 'resnet18',
    only_digits: bool = False,
    merge_case: bool = False,
) -> training_specs.RunnerSpecFederated:
  """Configures federated training for the EMNIST character recognition task.

  This method will load and pre-process datasets and construct a model used for
  the task. It then uses `iterative_process_builder` to create an iterative
  process compatible with `tff.simulation.run_training_process`.

  Args:
    task_spec: A `TaskSpecFederated` instance for creating federated training
      tasks.
    model: Which model to use. This can be a shallow convent (cnn) with around
      1M parameters, resnet18, resnet34, resnet50, resnet101, or resnet152.
    only_digits: Whether to use digit-only version of EMNIST (with 10 classes).
      If false, use the full dataset.
    merge_case: Whether to merge the 15 upper and lower case letters as proposed
      by NIST/EMNIST. Can be true only if only_digits is False.

  Returns:
    A `RunnerSpecFederated` instance containing attributes used for running the
      newly created federated task.
  """
  return _EmnistCharacterTask(
      task_spec,
      model=model,
      only_digits=only_digits,
      merge_case=merge_case,
  ).build_federated_runner_spec()


def configure_training_centralized(
    task_spec: training_specs.TaskSpecCentralized,
    *,  # Caller passes below args by name.
    model: str = 'resnet18',
    only_digits: bool = False,
    merge_case: bool = False,
) -> training_specs.RunnerSpecCentralized:
  """Configures centralized training for the EMNIST character recognition task.

  Args:
    task_spec: A `TaskSpecCentralized` instance for creating federated training
      tasks.
    model: Which model to use. This can be a shallow convent (cnn) with around
      1M parameters, resnet18, resnet34, resnet50, resnet101, or resnet152.
    only_digits: Whether to use digit-only version of EMNIST (with 10 classes).
      If false, use the full dataset.
    merge_case: Whether to merge the 15 upper and lower case letters as proposed
      by NIST/EMNIST. Can be true only if only_digits is False.

  Returns:
    A `RunnerSpecCentralized` instance containing attributes used for running
    the newly created centralized task.
  """
  return _EmnistCharacterTask(
      task_spec,
      model=model,
      only_digits=only_digits,
      merge_case=merge_case,
  ).build_centralized_runner_spec()
