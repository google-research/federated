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
"""Federated and Centralized StackOverflow next word prediction library using TFF."""

import functools
from typing import Callable, List, Optional, Tuple

import tensorflow as tf
import tensorflow_federated as tff

from generalization.tasks import training_specs
from generalization.utils import client_data_utils
from generalization.utils import eval_metric_distribution
from generalization.utils import trainer_utils
from utils import keras_metrics
from utils import utils_impl
from utils.datasets import stackoverflow_word_prediction
from utils.models import stackoverflow_models

ClientData = tff.simulation.datasets.ClientData

with utils_impl.record_hparam_flags() as stackoverflow_word_flags:
  FLAG_PREFIX = 'stackoverflow_word_'

_VOCAB_SIZE = 10000  # Number of most frequent words to use in the vocabulary.
_NUM_OOV_BUCKETS = 1  # Number of out-of-vocabulary buckets to use.
_SEQUENCE_LENGTH = 20  # The maximum number of words to take for each sequence.
_EMBEDDING_SIZE = 96  # The dimension of the word embedding layer.
_LATENT_SIZE = 670  # The dimension of the latent units in the recurrent layers.
_NUM_LAYERS = 1  # The number of stacked recurrent layers to use.
_SHARED_EMBEDDING = False  # Whether to tie input and output embeddings.

# Stackoverflow has 101 participating clients with only one sample.
# These clients will be removed from training set to guarantee non-empty split
# for both training and validation dataset on client.
_TRAIN_CLIENT_IDS_WITH_SINGLE_ELEMENT = [
    '01088827', '01293749', '01521579', '01573395', '01631967', '01926477',
    '02055393', '02376317', '02530536', '02735311', '03072286', '03114765',
    '03205995', '03469273', '03565132', '03596335', '03611224', '03656785',
    '03670607', '03807226', '03944091', '03950039', '04035036', '04044951',
    '04083636', '04142702', '04279414', '04350517', '04397306', '04426149',
    '04469425', '04511702', '04561332', '04670639', '04868978', '04932316',
    '04971999', '05201594', '05250973', '05310544', '05317459', '05405453',
    '05432156', '05433896', '05437487', '05449976', '05557556', '05676318',
    '05797823', '05971071', '05979052', '06050113', '06086865', '06121568',
    '06172826', '06243352', '06477644', '06599635', '06664628', '06682154',
    '06715227', '06786713', '06824968', '06920195', '07015378', '07103882',
    '07148391', '07310077', '07341563', '07352806', '07445289', '07500028',
    '07529386', '07534341', '07550592', '07647827', '07700503', '07913031',
    '08037585', '08153765', '08160881', '08177207', '08179245', '08194089',
    '08479303', '08494762', '08601435', '08619959', '08706234', '08725816',
    '08737036', '08747108', '08808047', '08884381', '08916482', '09018535',
    '09065705', '09088454', '09110142', '09131933', '09131967'
]


def load_partitioned_tff_original_stackoverflow_client_data(
    *,  # Caller passes below args by name.
    part_clients_subsampling_rate: float,
    train_val_ratio_intra_client: int,
    include_unpart_train_for_val: bool,
    max_elements_per_client: Optional[int],
    seed: Optional[int] = None,
) -> Tuple[ClientData, ClientData, ClientData, ClientData]:
  """Construct partitions from the TFF original Stackoverflow federated dataset."""
  (part_cd, unpart_cd,
   test_cd) = tff.simulation.datasets.stackoverflow.load_data()

  # Stackoverflow has ~100 training clients with only one sample.
  # These clients are removed from training set to guarantee non-empty split
  # for both training and validation dataset on client.
  part_client_ids_with_at_least_two_samples = list(
      set(part_cd.client_ids) - set(_TRAIN_CLIENT_IDS_WITH_SINGLE_ELEMENT))

  # Subsampling participating clients.
  part_client_ids = client_data_utils.subsample_list_by_proportion(
      part_client_ids_with_at_least_two_samples,
      part_clients_subsampling_rate,
      seed=seed)
  part_cd = client_data_utils.vertical_subset_client_data(
      part_cd, part_client_ids)

  # Horizontal split participating to train and val.
  (part_train_cd, part_val_cd) = client_data_utils.horizontal_split_client_data(
      part_cd,
      first_second_ratio_for_each_client=train_val_ratio_intra_client,
      shuffle_before_split=True,
      shuffle_seed=seed)

  # If not include unpart train for val,  horizontal split unpart and test cd.
  if not include_unpart_train_for_val:
    _, unpart_cd = client_data_utils.horizontal_split_client_data(
        unpart_cd,
        first_second_ratio_for_each_client=train_val_ratio_intra_client,
        shuffle_before_split=True,
        shuffle_seed=seed)
    _, test_cd = client_data_utils.horizontal_split_client_data(
        test_cd,
        first_second_ratio_for_each_client=train_val_ratio_intra_client,
        shuffle_before_split=True,
        shuffle_seed=seed)

  # Truncate client dataset if task_spec.max_elements_per_client is not None.
  if max_elements_per_client is not None:
    truncate_fn = lambda ds: ds.take(max_elements_per_client)
    part_train_cd = part_train_cd.preprocess(truncate_fn)
    part_val_cd = part_val_cd.preprocess(truncate_fn)
    unpart_cd = unpart_cd.preprocess(truncate_fn)
    test_cd = test_cd.preprocess(truncate_fn)

  return part_train_cd, part_val_cd, unpart_cd, test_cd


@functools.lru_cache()  # Avoid duplicate I/O by caching the result.
def _load_vocab() -> List[str]:
  return stackoverflow_word_prediction.create_vocab(_VOCAB_SIZE)


def _create_preprocess_fn(
    batch_size: int,
    num_epochs: int,
    max_shuffle_buffer_size: int = 10000,
    use_cache: bool = True,
    use_prefetch: bool = True,
) -> Callable[[tf.data.Dataset], tf.data.Dataset]:
  """Create preprocess function based on batch_size and num_epochs."""
  vocab = _load_vocab()
  base_preprocess_fn = stackoverflow_word_prediction.create_preprocess_fn(
      vocab=vocab,
      client_batch_size=batch_size,
      client_epochs_per_round=num_epochs,
      num_oov_buckets=_NUM_OOV_BUCKETS,
      max_sequence_length=_SEQUENCE_LENGTH,
      max_shuffle_buffer_size=max_shuffle_buffer_size,
      max_elements_per_client=-1
  )  # Client dataset truncation is handled separately.

  def preprocess_fn(dataset: tf.data.Dataset) -> tf.data.Dataset:
    if use_cache:
      dataset = dataset.cache()
    dataset = base_preprocess_fn(dataset)
    if use_prefetch:
      dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

  return preprocess_fn


def _keras_model_builder():
  return stackoverflow_models.create_recurrent_model(
      vocab_size=_VOCAB_SIZE,
      num_oov_buckets=_NUM_OOV_BUCKETS,
      embedding_size=_EMBEDDING_SIZE,
      latent_size=_LATENT_SIZE,
      num_layers=_NUM_LAYERS,
      shared_embedding=_SHARED_EMBEDDING)


def _loss_builder() -> tf.keras.losses.Loss:
  return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def _metrics_builder_generic(
    tff_training: bool = True) -> List[tf.keras.metrics.Metric]:
  """Returns a list of `tf.keras.metrics`."""
  special_tokens = stackoverflow_word_prediction.get_special_tokens(
      vocab_size=_VOCAB_SIZE, num_oov_buckets=_NUM_OOV_BUCKETS)
  pad_token = special_tokens.pad
  oov_tokens = special_tokens.oov
  eos_token = special_tokens.eos

  metrics_list = [
      # keras_metrics.MaskedCategoricalAccuracy(
      #     name='acc_w_oov', masked_tokens=[pad_token]),
      # keras_metrics.MaskedCategoricalAccuracy(
      #     name='acc_no_oov', masked_tokens=[pad_token] + oov_tokens),
      # Notice BOS never appears in ground truth.
      keras_metrics.MaskedCategoricalAccuracy(
          name='acc_no_oov_or_eos',
          masked_tokens=[pad_token, eos_token] + oov_tokens)
  ]
  if tff_training:
    # Notice num_tokens are necessary for weighting.
    metrics_list.append(
        keras_metrics.NumTokensCounter(masked_tokens=[pad_token]))
  else:
    # Append loss to metrics unless using TFF training,
    # (in which case loss will be appended to metrics list by keras_utils).
    # This includes centralized training/evaluation and federated evaluation.
    metrics_list.append(
        tf.keras.metrics.SparseCategoricalCrossentropy(
            name='loss', from_logits=True))
  return metrics_list


class _StackoverflowWordTask():
  """Backend class for configuring centralized and federated Stackoverflow Word task."""

  def __init__(self, task_spec):
    self._task_spec = task_spec

    if task_spec.sql_database is not None:
      raise NotImplementedError(
          'Stackoverflow custom SQL ClientData not implemented yet.')
    else:
      if task_spec.unpart_clients_proportion is not None:
        raise ValueError(
            'unpart_clients_proportion must be None since TFF original stackoverflow '
            'has provided a vertical (inter-client) part/unpart split. '
            f'got {task_spec.unpart_clients_proportion}.')
      if task_spec.train_val_ratio_intra_client is None:
        raise ValueError(
            'train_val_ratio_intra_client cannot be None since TFF original '
            'stackoverflow does not provide a vertical split.')
      (part_train_cd_raw, part_val_cd_raw, unpart_cd_raw, test_cd_raw
      ) = load_partitioned_tff_original_stackoverflow_client_data(
          train_val_ratio_intra_client=task_spec.train_val_ratio_intra_client,
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
    self._test_cd = test_cd_raw.preprocess(eval_preprocess_fn)

  def _tff_model_builder(self) -> tff.learning.Model:
    return tff.learning.from_keras_model(
        keras_model=_keras_model_builder(),
        input_spec=self._unpart_cd.element_type_structure,
        loss=_loss_builder(),
        metrics=_metrics_builder_generic(tff_training=True))

  def build_federated_runner_spec(self) -> training_specs.RunnerSpecFederated:
    """Configuring federated runner spec."""
    task_spec = self._task_spec

    if task_spec.max_elements_per_client is None:
      intra_client_shuffle_buffer_size = 10000
    else:
      intra_client_shuffle_buffer_size = min(10000,
                                             task_spec.max_elements_per_client)

    train_preprocess_fn = _create_preprocess_fn(
        batch_size=task_spec.client_batch_size,
        num_epochs=task_spec.client_epochs_per_round,
        max_shuffle_buffer_size=intra_client_shuffle_buffer_size)
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

    model = _keras_model_builder()
    model.compile(
        loss=_loss_builder(),
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
  """Configures federated training for Stack Overflow next-word prediction.

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
  return _StackoverflowWordTask(task_spec).build_federated_runner_spec()


def configure_training_centralized(
    task_spec: training_specs.TaskSpecCentralized,
) -> training_specs.RunnerSpecCentralized:
  """Configures centralized training for Stack Overflow next-word prediction.

  This method will load and pre-process datasets and construct a model used for
  the task.

  Args:
    task_spec: A `TaskSpecCentralized` instance for creating centralized
      training tasks.

  Returns:
    A `RunnerSpecCentralized` instance containing attributes used for running
      the newly created centralized task.
  """

  return _StackoverflowWordTask(task_spec).build_centralized_runner_spec()
