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
"""Federated and Centralized Character task library using TFF."""

import functools
from typing import Callable, List

import tensorflow as tf
import tensorflow_federated as tff

from generalization.tasks import training_specs
from generalization.utils import client_data_utils
from generalization.utils import eval_metric_distribution
from generalization.utils import trainer_utils
from utils import keras_metrics
from utils import utils_impl
from utils.datasets import shakespeare_dataset
from utils.models import shakespeare_models

ClientData = tff.simulation.datasets.ClientData

with utils_impl.record_hparam_flags() as shakespeare_character_flags:
  FLAG_PREFIX = 'shakespeare_character'

# Vocabulary with OOV ID, zero for the padding, and BOS, EOS IDs.
_VOCAB_SIZE = len(shakespeare_dataset.CHAR_VOCAB) + 4

# The length of the character sequences used for prediction.
_SEQUENCE_LENGTH = 80


def _create_preprocess_fn(
    batch_size: int,
    num_epochs: int,
    max_shuffle_buffer_size: int = 10000,
    use_cache: bool = True,
    use_prefetch: bool = True,
) -> Callable[[tf.data.Dataset], tf.data.Dataset]:
  """Create preprocess function based on batch_size and num_epochs."""

  base_preprocess_fn = shakespeare_dataset.create_preprocess_fn(
      batch_size=batch_size,
      num_epochs=num_epochs,
      shuffle_buffer_size=max_shuffle_buffer_size,
      sequence_length=_SEQUENCE_LENGTH)

  def preprocess_fn(dataset: tf.data.Dataset) -> tf.data.Dataset:
    if use_cache:
      dataset = dataset.cache()
    dataset = base_preprocess_fn(dataset)
    if use_prefetch:
      dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

  return preprocess_fn


def _keras_model_builder() -> tf.keras.Model:
  return shakespeare_models.create_recurrent_model(
      vocab_size=_VOCAB_SIZE, sequence_length=_SEQUENCE_LENGTH)


def _loss_builder() -> tf.keras.losses.Loss:
  return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def _metrics_builder_generic(
    tff_training: bool = True) -> List[tf.keras.metrics.Metric]:
  """Returns a list of `tf.keras.metrics`."""
  pad_token, _, _, _ = shakespeare_dataset.get_special_tokens()

  metrics_list = [
      keras_metrics.MaskedCategoricalAccuracy(
          masked_tokens=[pad_token], name='acc'),
  ]
  if not tff_training:
    # Append loss to metrics unless using TFF training,
    # (in which case loss will be appended to metrics list by keras_utils).
    # This includes centralized training/evaluation and federated evaluation.
    metrics_list.append(
        tf.keras.metrics.SparseCategoricalCrossentropy(
            name='loss', from_logits=True))
  return metrics_list


class _ShakeSpeareCharacterTask():
  """Backend class for configuring centralized and federated shakespeare character task."""

  def __init__(self, task_spec):
    self._task_spec = task_spec

    if task_spec.sql_database is not None:
      raise NotImplementedError(
          'Shakespeare custom SQL ClientData not implemented yet.')
    else:
      if task_spec.train_val_ratio_intra_client is not None:
        raise ValueError(
            'train_val_ratio_intra_client must be None since TFF original '
            'EMNIST has provided a horizontal (intra-client) train/val split. '
            f'got {task_spec.train_val_ratio_intra_client}.')
      (train_cd_orig,
       val_cd_orig) = tff.simulation.datasets.shakespeare.load_data()

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
        batch_size=task_spec.eval_client_batch_size,
        num_epochs=1,
        # Set buffer to 1 to disable shuffling since is not necessary for eval.
        max_shuffle_buffer_size=1)

    self._part_train_cd_raw = part_train_cd_raw
    self._part_train_eval_cd = part_train_cd_raw.preprocess(eval_preprocess_fn)
    self._part_val_cd = part_val_cd_raw.preprocess(eval_preprocess_fn)
    self._unpart_cd = unpart_cd_raw.preprocess(eval_preprocess_fn)
    self._test_cd = None

    self._keras_model_builder = _keras_model_builder
    self._loss_builder = _loss_builder

  def _tff_model_builder(self) -> tff.learning.Model:
    return tff.learning.from_keras_model(
        keras_model=self._keras_model_builder(),
        input_spec=self._unpart_cd.element_type_structure,
        loss=self._loss_builder(),
        metrics=_metrics_builder_generic(tff_training=True))

  def build_federated_runner_spec(self) -> training_specs.RunnerSpecFederated:
    """Configuring federated runner spec."""
    task_spec = self._task_spec

    train_preprocess_fn = _create_preprocess_fn(
        batch_size=task_spec.client_batch_size,
        num_epochs=task_spec.client_epochs_per_round,
        max_shuffle_buffer_size=10000)
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
     test_fn) = trainer_utils.create_federated_eval_fns(
         tff_model_builder=self._tff_model_builder,
         metrics_builder=functools.partial(
             _metrics_builder_generic, tff_training=False),
         part_train_eval_cd=self._part_train_eval_cd,
         part_val_cd=self._part_val_cd,
         unpart_cd=self._unpart_cd,
         test_cd=self._test_cd,
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
        test_fn=test_fn)

  def build_centralized_runner_spec(
      self) -> training_specs.RunnerSpecCentralized:
    """Configuring centralized runner spec."""

    task_spec = self._task_spec

    train_preprocess_fn = _create_preprocess_fn(
        batch_size=task_spec.batch_size,
        num_epochs=1,
        max_shuffle_buffer_size=task_spec.centralized_shuffle_buffer_size)

    train_dataset = train_preprocess_fn(
        client_data_utils.interleave_create_tf_dataset_from_all_clients(
            self._part_train_cd_raw, seed=task_spec.shared_random_seed))

    (part_train_eval_fn, part_val_fn, unpart_fn,
     test_fn) = trainer_utils.create_centralized_eval_fns(
         tff_model_builder=self._tff_model_builder,
         metrics_builder=functools.partial(
             _metrics_builder_generic, tff_training=False),
         part_train_eval_cd=self._part_train_eval_cd,
         part_val_cd=self._part_val_cd,
         unpart_cd=self._unpart_cd,
         test_cd=self._test_cd,
         stat_fns=eval_metric_distribution.ALL_STAT_FNS,
         part_clients_per_eval=task_spec.part_clients_per_eval,
         unpart_clients_per_eval=task_spec.unpart_clients_per_eval,
         test_clients_for_eval=task_spec.test_clients_for_eval,
         resample_eval_clients=task_spec.resample_eval_clients,
         eval_clients_random_seed=task_spec.shared_random_seed)

    model = self._keras_model_builder()
    model.compile(
        loss=self._loss_builder(),
        optimizer=task_spec.optimizer,
        metrics=_metrics_builder_generic(tff_training=False))

    return training_specs.RunnerSpecCentralized(
        keras_model=model,
        train_dataset=train_dataset,
        part_train_eval_fn=part_train_eval_fn,
        part_val_fn=part_val_fn,
        unpart_fn=unpart_fn,
        test_fn=test_fn)


def configure_training_federated(
    task_spec: training_specs.TaskSpecFederated,
) -> training_specs.RunnerSpecFederated:
  """Configures federated training for Shakespeare character task.

  This method will load and pre-process datasets and construct a model used for
  the task. It then uses `iterative_process_builder` to create an iterative
  process compatible with `tff.simulation.run_training_process`.

  Args:
    task_spec: A `TaskSpecFederated` instance for creating federated training
      tasks.

  Returns:
    A `RunnerSpecFederated` instance containing attributes used for running the
      newly created federated task.
  """
  return _ShakeSpeareCharacterTask(task_spec).build_federated_runner_spec()


def configure_training_centralized(
    task_spec: training_specs.TaskSpecCentralized,
) -> training_specs.RunnerSpecCentralized:
  """Configures centralized training for Shakespeare character task.

  This method will load and pre-process datasets and construct a model used for
  the task.

  Args:
    task_spec: A `TaskSpecCentralized` instance for creating centralized
      training tasks.

  Returns:
    A `RunnerSpecCentralized` instance containing attributes used for running
      the newly created centralized task.
  """

  return _ShakeSpeareCharacterTask(task_spec).build_centralized_runner_spec()
