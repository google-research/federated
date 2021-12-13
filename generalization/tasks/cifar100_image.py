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
"""Federated and Centralized CIFAR-100 image classification library using TFF."""

import collections
import functools
from typing import Callable, List, Mapping, Optional, Tuple

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
from utils.datasets import cifar100_dataset

_NUM_EXAMPLES_PER_CLIENT = 500
_CROP_SHAPE = (24, 24, 3)  # The shape of images after preprocessing.
_DISTORT_TRAIN_IMAGES = True  # Images will be randomly cropped.

_ELEMENT_SPEC = collections.OrderedDict([
    ('image', tf.TensorSpec(shape=(32, 32, 3), dtype=tf.uint8)),
    ('label', tf.TensorSpec(shape=(), dtype=tf.int64)),
])

_BATCH_SPEC = collections.OrderedDict([
    ('image', tf.TensorSpec(shape=(None, 32, 32, 3), dtype=tf.uint8)),
    ('label', tf.TensorSpec(shape=(None,), dtype=tf.int64)),
])

ClientData = tff.simulation.datasets.ClientData

with utils_impl.record_hparam_flags() as cifar100_image_flags:
  # CIFAR 100 flags.
  FLAG_PREFIX = 'cifar100_image_'
  flags.DEFINE_enum(
      FLAG_PREFIX + 'resnet_layers', '18', ['18', '34', '50', '101', '152'],
      'Which versions of resnet model to use. This can be 18, 34, 50, 101, or 152.'
  )
  flags.DEFINE_integer(
      FLAG_PREFIX + 'num_classes', 100,
      'An integer representing the number of classes (since we also allow '
      'CIFAR-like dataset such as CIFAR-10).')
  flags.DEFINE_float(
      FLAG_PREFIX + 'l2_weight_decay', 1e-4,
      'A floating number representing the strength of l2 weight decay.')


def load_partitioned_tff_original_cifar100_client_data(
    *,  # Caller passes below args by name.
    part_clients_subsampling_rate: float,
    train_val_ratio_intra_client: int,
    include_unpart_train_for_val: bool,
    max_elements_per_client: Optional[int],
    seed: Optional[int] = None,
) -> Tuple[ClientData, ClientData, ClientData]:
  """Construct partitions from the TFF original CIFAR100 federated dataset."""
  part_cd, unpart_cd = tff.simulation.datasets.cifar100.load_data()

  def elem_map(elem: Mapping[str, tf.Tensor]) -> Mapping[str, tf.Tensor]:
    """Remove coarse label from element to keep consistency."""
    return collections.OrderedDict(image=elem['image'], label=elem['label'])

  ds_map = lambda ds: ds.map(elem_map)
  part_cd = part_cd.preprocess(ds_map)
  unpart_cd = unpart_cd.preprocess(ds_map)

  # Subsampling participating clients.
  part_client_ids = client_data_utils.subsample_list_by_proportion(
      part_cd.client_ids, part_clients_subsampling_rate, seed=seed)
  part_cd = client_data_utils.vertical_subset_client_data(
      part_cd, part_client_ids)

  # Horizontal split participating to train and val.
  part_train_cd, part_val_cd = client_data_utils.horizontal_split_client_data(
      part_cd,
      first_second_ratio_for_each_client=train_val_ratio_intra_client,
      shuffle_before_split=True,
      shuffle_seed=seed)

  # If not include unpart train for val, horizontal split unpart_cd.
  if not include_unpart_train_for_val:
    _, unpart_cd = client_data_utils.horizontal_split_client_data(
        unpart_cd,
        first_second_ratio_for_each_client=train_val_ratio_intra_client,
        shuffle_before_split=True,
        shuffle_seed=seed)

  # Truncate client dataset if max_elements_per_client is not None.
  if max_elements_per_client is not None:
    truncate_fn = lambda ds: ds.take(max_elements_per_client)
    part_train_cd = part_train_cd.preprocess(truncate_fn)
    part_val_cd = part_val_cd.preprocess(truncate_fn)
    unpart_cd = unpart_cd.preprocess(truncate_fn)

  return part_train_cd, part_val_cd, unpart_cd


def load_custom_cifar_client_data(sql_database: str) -> ClientData:
  """Load (un-splitted) CIFAR clientdata from sql database."""

  if sql_database is None:
    raise ValueError('sql_database cannot be None.')

  return sql_client_data_utils.load_parsed_sql_client_data(
      sql_database, element_spec=_ELEMENT_SPEC)


def load_partitioned_custom_cifar_client_data(
    sql_database: str,
    *,  # Caller passes below args by name.
    unpart_clients_proportion: float,
    train_val_ratio_intra_client: int,
    part_clients_subsampling_rate: float,
    include_unpart_train_for_val: bool,
    max_elements_per_client: int,
    seed: Optional[int] = None,
) -> Tuple[ClientData, ClientData, ClientData]:
  """Construct three-way partition from SQL-based custom CIFAR-like ClientData."""

  total_cd = load_custom_cifar_client_data(sql_database)

  (part_train_cd, part_val_cd,
   unpart_cd) = client_data_utils.canonical_three_way_partition_client_data(
       total_cd,
       unpart_clients_proportion=unpart_clients_proportion,
       train_val_ratio_intra_client=train_val_ratio_intra_client,
       part_clients_subsampling_rate=part_clients_subsampling_rate,
       include_unpart_train_for_val=include_unpart_train_for_val,
       max_elements_per_client=max_elements_per_client,
       shuffle_buffer_size=100,
       seed=seed)

  return part_train_cd, part_val_cd, unpart_cd


def _create_preprocess_fn(
    num_epochs: int,
    batch_size: int,
    shuffle_buffer_size: int = _NUM_EXAMPLES_PER_CLIENT,
    use_cache: bool = True,
    use_prefetch: bool = True,
) -> Callable[[tf.data.Dataset], tf.data.Dataset]:
  """Create a preprocessing function for EMNIST client datasets."""
  base_preprocess_fn = cifar100_dataset.create_preprocess_fn(
      num_epochs=num_epochs,
      batch_size=batch_size,
      crop_shape=_CROP_SHAPE,
      distort_image=_DISTORT_TRAIN_IMAGES,
      # Set buffer to 1 to disable shuffling since is not necessary for eval.
      shuffle_buffer_size=shuffle_buffer_size)

  def preprocess_fn(dataset: tf.data.Dataset):
    if use_cache:
      dataset = dataset.cache()
    dataset = base_preprocess_fn(dataset)
    if use_prefetch:
      dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

  return preprocess_fn


def _loss_builder() -> tf.keras.losses.Loss:
  return tf.keras.losses.SparseCategoricalCrossentropy()


def _metrics_builder_generic(
    tff_training: bool = True) -> List[tf.keras.metrics.Metric]:
  metrics_list = [tf.keras.metrics.SparseCategoricalAccuracy(name='acc')]
  if not tff_training:
    # Append loss to metrics unless using TFF training,
    # (in which case loss will be appended to metrics list by keras_utils).
    # This includes centralized training/evaluation and federated evaluation.
    metrics_list.append(
        tf.keras.metrics.SparseCategoricalCrossentropy(name='loss'))
  return metrics_list


class _Cifar100ImageTask():
  """Backend class for configuring centralized and federated CIFAR100 Image task."""

  def __init__(
      self,
      task_spec,
      *,  # Caller passes below args by name.
      resnet_layers: int,
      num_classes: int,
      l2_weight_decay: float):
    self._task_spec = task_spec
    self._keras_model_builder = functools.partial(
        getattr(resnet_models, f'create_resnet{resnet_layers}'),
        input_shape=_CROP_SHAPE,
        num_classes=num_classes,
        l2_weight_decay=l2_weight_decay)

    if task_spec.sql_database is not None:
      # The backend client_data_utils.canonical_three_way_partition_client_data
      # will run validity check.
      (part_train_cd_raw, part_val_cd_raw, unpart_cd_raw
      ) = load_partitioned_custom_cifar_client_data(
          task_spec.sql_database,
          unpart_clients_proportion=task_spec.unpart_clients_proportion,
          train_val_ratio_intra_client=task_spec.train_val_ratio_intra_client,
          part_clients_subsampling_rate=task_spec.part_clients_subsampling_rate,
          include_unpart_train_for_val=task_spec.include_unpart_train_for_val,
          max_elements_per_client=task_spec.max_elements_per_client,
          seed=task_spec.shared_random_seed)
    else:
      if num_classes != 100:
        raise ValueError('num_classes must be 100 for TFF original CIFAR100.')
      if task_spec.unpart_clients_proportion is not None:
        raise ValueError(
            'unpart_clients_proportion must be None since TFF original CIFAR100 '
            'has provided a vertical (inter-client) part/unpart split. '
            f'got {task_spec.unpart_clients_proportion}.')
      if task_spec.train_val_ratio_intra_client is None:
        raise ValueError(
            'train_val_ratio_intra_client cannot be None since TFF original '
            'CIFAR100 does not provide a horizontal split.')
      (part_train_cd_raw, part_val_cd_raw, unpart_cd_raw
      ) = load_partitioned_tff_original_cifar100_client_data(
          train_val_ratio_intra_client=task_spec.train_val_ratio_intra_client,
          part_clients_subsampling_rate=task_spec.part_clients_subsampling_rate,
          include_unpart_train_for_val=task_spec.include_unpart_train_for_val,
          max_elements_per_client=task_spec.max_elements_per_client,
          seed=task_spec.shared_random_seed)

    eval_preprocess_fn = _create_preprocess_fn(
        num_epochs=1,
        batch_size=task_spec.eval_client_batch_size,
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
    resnet_layers: int = 18,
    num_classes: int = 100,
    l2_weight_decay: float = 1e-4,
) -> training_specs.RunnerSpecFederated:
  """Configures federated training for the CIFAR-100 classification task.

  This method will load and pre-process datasets and construct a model used for
  the task. It then uses `iterative_process_builder` to create an iterative
  process compatible with `tff.simulation.run_simulation`.

  Args:
    task_spec: A `TaskSpecFederated` instance for creating federated training
      tasks.
    resnet_layers: Which versions of resnet model to use. This can be 18, 34,
      50, 101, or 152.
    num_classes: An integer representing the number of classes (since we also
      allow other CIFAR-like dataset such as CIFAR-10).
    l2_weight_decay: A floating number representing the strength of l2 weight
      decay.

  Returns:
    A `RunnerSpecFederated` instance containing attributes used for running the
      newly created federated task.
  """

  return _Cifar100ImageTask(
      task_spec,
      resnet_layers=resnet_layers,
      num_classes=num_classes,
      l2_weight_decay=l2_weight_decay).build_federated_runner_spec()


def configure_training_centralized(
    task_spec: training_specs.TaskSpecCentralized,
    *,  # Caller passes below args by name.
    resnet_layers: int = 18,
    num_classes: int = 100,
    l2_weight_decay: float = 1e-4,
) -> training_specs.RunnerSpecCentralized:
  """Configures centralized training for the CIFAR-100 image classification task.

  This method will load and pre-process datasets and construct a model used for
  the task. It then uses `iterative_process_builder` to create an iterative
  process compatible with `tff.simulation.run_simulation`.

  Args:
    task_spec: A `TaskSpecFederated` instance for creating federated training
      tasks.
    resnet_layers: Which versions of resnet model to use. This can be 18, 34,
      50, 101, or 152.
    num_classes: An integer representing the number of classes (since we also
      allow other CIFAR-like dataset such as CIFAR-10).
    l2_weight_decay: A floating number representing the strength of l2 weight
      decay.

  Returns:
    A `RunnerSpecCentralized` instance containing attributes used for running
    the
      newly created centralized task.
  """

  return _Cifar100ImageTask(
      task_spec,
      resnet_layers=resnet_layers,
      num_classes=num_classes,
      l2_weight_decay=l2_weight_decay).build_centralized_runner_spec()
