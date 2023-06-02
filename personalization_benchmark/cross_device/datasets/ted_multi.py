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
"""Data preprocessing functions for Ted Multi dataset."""

import collections
import os
import re
import string
from typing import Callable, List, Mapping, OrderedDict, Tuple

from absl import logging
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from personalization_benchmark.cross_device import constants
from personalization_benchmark.cross_device.datasets import emnist
from personalization_benchmark.cross_device.datasets import transformer_models
from utils import keras_metrics
from utils.datasets import stackoverflow_word_prediction as stackoverflow

# Must replace it with the directory that saves the federated version of the
# TedMultiTranslate dataset created by `write_ted_multi_to_sql_client_data.py`.
_TED_MULTI_DATA_DIRECTORY = None

# Path to load the vocabulary and client information from Google Cloud Storage.
_VOCAB_PATH = 'https://storage.googleapis.com/tff-experiments-public/personalization_benchmark/ted_multi_en_es_vocab_with_client_cnt'
_CLIENT_ID_PATH = 'https://storage.googleapis.com/tff-experiments-public/personalization_benchmark/ted_multi_en_es_client_data_size'

# Words that appear in "<_MIN_CLIENTS_PER_WORD" clients will be filtered.
_MIN_CLIENTS_PER_WORD = 20

# Clients with "<_MIN_ELEMENTS_PER_CLIENT" samples will be filtered.
_MIN_ELEMENTS_PER_CLIENT = 20

# Most of the clients in EN & ES have < 500 samples.
_MAX_ELEMENTS_PER_CLIENT = 512
_ACCURACY_NAME = 'accuracy_no_oov_or_eos'
_NUM_OOV_BUCKETS = 1
_MAX_SEQ_LEN = 20

# DEFAULT TRANSFORMER PARAMETERS
_DIM_EMBED = 96
_DIM_MODEL = 512
_DIM_HIDDEN = 2048
_NUM_HEADS = 8
_NUM_LAYER = 1
_MAX_POS_CODE = 1000
_DROPOUT = 0.1


def _get_metrics(vocab_size: int,
                 num_oov_buckets: int) -> List[tf.keras.metrics.Metric]:
  """Returns language model metrics."""
  special_tokens = stackoverflow.get_special_tokens(vocab_size, num_oov_buckets)
  pad_token = special_tokens.pad
  oov_tokens = special_tokens.oov
  eos_token = special_tokens.eos

  return [
      keras_metrics.MaskedCategoricalAccuracy(
          name='accuracy_with_oov', masked_tokens=[pad_token]),
      keras_metrics.MaskedCategoricalAccuracy(
          name='accuracy_no_oov', masked_tokens=[pad_token] + oov_tokens),
      # Notice BOS never appears in ground truth.
      keras_metrics.MaskedCategoricalAccuracy(
          name='accuracy_no_oov_or_eos',
          masked_tokens=[pad_token, eos_token] + oov_tokens),
      tff.learning.metrics.NumBatchesCounter(),
      keras_metrics.NumTokensCounter(masked_tokens=[pad_token])
  ]


def _get_keras_model(vocab_size: int,
                     num_oov_buckets: int,
                     dim_embed: int = _DIM_EMBED,
                     dim_model: int = _DIM_MODEL,
                     dim_hidden: int = _DIM_HIDDEN,
                     num_heads: int = _NUM_HEADS,
                     num_layers: int = _NUM_LAYER,
                     max_position_encoding: int = _MAX_POS_CODE,
                     dropout: float = _DROPOUT) -> tf.keras.Model:
  return transformer_models.create_transformer_lm(
      vocab_size=vocab_size,
      num_oov_buckets=num_oov_buckets,
      dim_embed=dim_embed,
      dim_model=dim_model,
      dim_hidden=dim_hidden,
      num_heads=num_heads,
      num_layers=num_layers,
      max_position_encoding=max_position_encoding,
      dropout=dropout,
      name='ted_multi-transformer')


def _load_vocab(min_clients: int = _MIN_CLIENTS_PER_WORD) -> List[str]:
  """Returns saved vocabulary with a filter on minimum number of clients."""
  vocab_filepath = tf.keras.utils.get_file(
      fname='ted_multi_en_es_vocab_with_client_cnt', origin=_VOCAB_PATH)
  with tf.io.gfile.GFile(vocab_filepath, mode='r') as f:
    vocab2cnt_strs = f.read().split('\n')
  vocab2cnt = [word2cnt.split(',') for word2cnt in vocab2cnt_strs]
  return [word for word, cnt_str in vocab2cnt if int(cnt_str) >= min_clients]


def _load_filtered_client_ids_by_min_elements(
    min_elements: int = _MIN_ELEMENTS_PER_CLIENT) -> List[str]:
  """Returns saved client ids with a filter on minimum number of elements."""
  client_id_file = tf.keras.utils.get_file(
      fname='ted_multi_en_es_client_data_size', origin=_CLIENT_ID_PATH)
  with tf.io.gfile.GFile(client_id_file, mode='r') as f:
    cid2cnt_strs = f.read().split('\n')
  cid2cnt_list = [cid2cnt.split(',') for cid2cnt in cid2cnt_strs]
  return [cid for cid, cnt_str in cid2cnt_list if int(cnt_str) >= min_elements]


def _load_filtered_data(filtered_cids: List[str]):
  """Returns a tuple of (train, validation, test) dataset with filtered clients."""
  cids_set = set(filtered_cids)
  if _TED_MULTI_DATA_DIRECTORY is None:
    raise ValueError(
        'You must run `write_ted_multi_to_sql_client_data.py` first to create '
        'the federated version of the TedMultiTranslate dataset, and then '
        'update the value of `_TED_MULTI_DATA_DIRECTORY` in file '
        '`cross_device/datasets/ted_multi.py` with the directory that saves '
        'the created federated dataset.')
  element_spec = collections.OrderedDict(
      # Features are intentionally sorted lexicographically by key for
      # consistency across datasets.
      sorted([
          ('language', tf.TensorSpec(shape=[], dtype=tf.string)),
          ('talk_name', tf.TensorSpec(shape=[], dtype=tf.string)),
          ('translation', tf.TensorSpec(shape=[], dtype=tf.string)),
      ]))
  try:
    train_set = tff.simulation.datasets.load_and_parse_sql_client_data(
        database_filepath=os.path.join(_TED_MULTI_DATA_DIRECTORY, 'train'),
        element_spec=element_spec)
    val_set = tff.simulation.datasets.load_and_parse_sql_client_data(
        database_filepath=os.path.join(_TED_MULTI_DATA_DIRECTORY, 'validation'),
        element_spec=element_spec)
    test_set = tff.simulation.datasets.load_and_parse_sql_client_data(
        database_filepath=os.path.join(_TED_MULTI_DATA_DIRECTORY, 'test'),
        element_spec=element_spec)
  except Exception as e:
    raise ValueError(
        'Cannot load and parse the sql client data from the provided '
        '`_TED_MULTI_DATA_DIRECTORY`. Please make sure you run '
        '`write_ted_multi_to_sql_client_data.py` first to create the federated '
        'version of the TedMultiTranslate dataset, and then update the value '
        'of `_TED_MULTI_DATA_DIRECTORY` in `cross_device/datasets/ted_multi.py`'
        ' with the correct directory that saves the created federated dataset.'
    ) from e

  def _filter_cid_for_clientdata(
      client_data: tff.simulation.datasets.ClientData
  ) -> tff.simulation.datasets.ClientData:
    filtered_cids = [cid for cid in client_data.client_ids if cid in cids_set]
    return tff.simulation.datasets.ClientData.from_clients_and_tf_fn(
        filtered_cids, client_data.serializable_dataset_fn)

  return (_filter_cid_for_clientdata(train_set),
          _filter_cid_for_clientdata(val_set),
          _filter_cid_for_clientdata(test_set))


def _filter_client_ids_by_language(client_ids: List[str],
                                   language: str = 'en') -> List[str]:
  return [cid for cid in client_ids if cid[:2] == language]


def _print_clients_number(clients: tff.simulation.datasets.ClientData,
                          prefix: str = 'train'):
  logging.info('%s clients total: %d, en: %d, es: %d', prefix,
               len(clients.client_ids),
               len(_filter_client_ids_by_language(clients.client_ids, 'en')),
               len(_filter_client_ids_by_language(clients.client_ids, 'es')))


def _get_example_dataset(clients: tff.simulation.datasets.ClientData,
                         preprocess_fn: Callable[[tf.data.Dataset],
                                                 tf.data.Dataset],
                         language: str = 'en') -> tf.data.Dataset:
  cids = _filter_client_ids_by_language(clients.client_ids, language)
  example_dataset = preprocess_fn(clients.create_tf_dataset_for_client(cids[0]))
  logging.info('%s example data batch %s', language,
               next(iter(example_dataset)))
  return example_dataset


def _text_standardization(input_data: tf.Tensor) -> tf.Tensor:
  """Standardizes input string Tensor by `lower_and_strip_punctuation` and removing html."""
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation), '')


def _text_tokenize(input_data: Mapping[str, tf.Tensor]) -> tf.Tensor:
  """Standardizes and tokenizes input string Tensor."""
  input_str = tf.reshape(input_data['translation'], shape=[1])
  standard_text = _text_standardization(input_str)
  return tf.strings.split(standard_text)


def _build_to_ids_fn(
    vocab: List[str],
    max_sequence_length: int,
    num_oov_buckets: int = 1) -> Callable[[Mapping[str, tf.Tensor]], tf.Tensor]:
  """Constructs function mapping examples to sequences of token indices."""
  special_tokens = stackoverflow.get_special_tokens(len(vocab), num_oov_buckets)
  bos = special_tokens.bos
  eos = special_tokens.eos

  table_values = np.arange(len(vocab), dtype=np.int64)
  table = tf.lookup.StaticVocabularyTable(
      tf.lookup.KeyValueTensorInitializer(vocab, table_values),
      num_oov_buckets=num_oov_buckets)

  def to_ids(example: Mapping[str, tf.Tensor]) -> tf.Tensor:
    words = _text_tokenize(example).values
    truncated_words = words[:max_sequence_length]
    tokens = table.lookup(truncated_words) + 1
    tokens = tf.cond(
        tf.less(tf.size(tokens), max_sequence_length),
        lambda: tf.concat([tokens, [eos]], 0), lambda: tokens)

    return tf.concat([[bos], tokens], 0)

  return to_ids


def _create_preprocess_fn(
    vocab: List[str],
    num_oov_buckets: int,
    client_batch_size: int,
    client_epochs_per_round: int,
    max_sequence_length: int,
    max_elements_per_client: int,
    max_shuffle_buffer_size: int = 10000
) -> Callable[[tf.data.Dataset], tf.data.Dataset]:
  """Creates a preprocessing functions for Ted Multi next-word-prediction.

  This function returns a function which takes a dataset and returns a
  dataset, suitable for mapping over a set of unprocessed client datasets.

  Args:
    vocab: Vocabulary which defines the embedding.
    num_oov_buckets: The number of out of vocabulary buckets. Tokens that are
      not present in the `vocab` are hashed into one of these buckets.
    client_batch_size: Integer representing batch size to use on the clients.
    client_epochs_per_round: Number of epochs for which to repeat train client
      dataset. Must be a positive integer.
    max_sequence_length: Integer determining shape of padded batches. Sequences
      will be padded up to this length, and sentences longer than this will be
      truncated to this length.
    max_elements_per_client: Integer controlling the maximum number of elements
      to take per client. If -1, keeps all elements for each client. This is
      applied before repeating `client_epochs_per_round`, and is intended
      primarily to contend with the small set of clients with tens of thousands
      of examples.
    max_shuffle_buffer_size: Maximum shuffle buffer size.

  Returns:
    A callable taking as input a `tf.data.Dataset`, and returning a
    `tf.data.Dataset` formed by preprocessing according to the input arguments.
  """
  if client_batch_size <= 0:
    raise ValueError('client_batch_size must be a positive integer. You have '
                     'passed {}.'.format(client_batch_size))
  elif client_epochs_per_round <= 0:
    raise ValueError('client_epochs_per_round must be a positive integer. '
                     'You have passed {}.'.format(client_epochs_per_round))
  elif max_sequence_length <= 0:
    raise ValueError('max_sequence_length must be a positive integer. You have '
                     'passed {}.'.format(max_sequence_length))
  elif max_elements_per_client == 0 or max_elements_per_client < -1:
    raise ValueError(
        'max_elements_per_client must be a positive integer or -1. You have '
        'passed {}.'.format(max_elements_per_client))
  if num_oov_buckets <= 0:
    raise ValueError('num_oov_buckets must be a positive integer. You have '
                     'passed {}.'.format(num_oov_buckets))

  if (max_elements_per_client == -1 or
      max_elements_per_client > max_shuffle_buffer_size):
    shuffle_buffer_size = max_shuffle_buffer_size
  else:
    shuffle_buffer_size = max_elements_per_client

  def preprocess_fn(dataset):
    to_ids = _build_to_ids_fn(
        vocab=vocab,
        max_sequence_length=max_sequence_length,
        num_oov_buckets=num_oov_buckets)
    dataset = dataset.take(max_elements_per_client).shuffle(
        shuffle_buffer_size).repeat(client_epochs_per_round).map(
            to_ids, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return stackoverflow.batch_and_split(dataset, max_sequence_length,
                                         client_batch_size)

  return preprocess_fn


def create_model_and_data(
    num_local_epochs: int, train_batch_size: int, use_synthetic_data: bool
) -> Tuple[constants.ModelFnType, constants.FederatedDatasetsType,
           constants.ProcessFnType, constants.SplitDataFnType, str]:
  """Creates model, datasets, and processing functions for StackOverflow.

  Args:
    num_local_epochs: Number of local epochs performed by a client in a round of
      FedAvg training. It is used in the returned `train_preprocess_fn`, which
      contains an operation `train_dataset.repeat(num_local_epochs)`.
    train_batch_size: Batch size used in FedAvg training. It is also used in the
      returned `train_preprocess_fn`, which contains an operation
      `train_dataset.batch(train_batch_size)`.
    use_synthetic_data: Whether to use synthetic data. This should only be set
      to True for debugging and testing purposes.

  Returns:
    1. A no-argument function that returns a `tff.learning.Model`.
    2. An `OrderedDict` of three federated datasets with keys being
      `constants.TRAIN_CLIENTS_KEY`, `constants.VALID_CLIENTS_KEY` and
      `constants.TEST_CLIENT_KEY`.
    3. A dataset preprocess function for per-client dataset in FedAvg training.
      Batching and repeat operations are contained.
    4. A dataset split function for per-client dataset in evaluation (including
      both validation clients and test clients). Specifically, each client's
      examples are sorted by `date` and split into two equal-sized unbatched
      datasets (a personalization dataset and a test dataset). The
      personalization dataset is used for finetuning the model in
      `finetuning_trainer` or choosing the best model in `hypcluster_trainer`.
    5. The accuracy name key stored in the evaluation metrics. It will be used
      in postprocessing the validation clients and test clients metrics in the
      `finetuning_trainer`.
  """
  if use_synthetic_data:
    logging.info('Ted Multi synthetic problem.')
    vocab = _synethetic_vocab()
    train_clients = tff.simulation.datasets.TestClientData(
        _create_sample_data_dictionary())
    val_clients = train_clients
    test_clients = train_clients
  else:
    min_clients = _MIN_CLIENTS_PER_WORD
    vocab = _load_vocab(min_clients=min_clients)
    vocab_size = len(vocab)
    logging.info(
        'Filtering vocabulary for all word to appearing in >= %d train clients',
        min_clients)
    min_elements = _MIN_ELEMENTS_PER_CLIENT
    filtered_cids = _load_filtered_client_ids_by_min_elements(
        min_elements=min_elements)
    logging.info('Ted Multi clients %d with each client having >= %d samples',
                 len(filtered_cids), min_elements)
    logging.info('Example client ids %s, %s', filtered_cids[:3],
                 filtered_cids[-3:])
    train_clients, val_clients, test_clients = _load_filtered_data(
        filtered_cids)
  vocab_size = len(vocab)
  logging.info('Ted Multi vocab size %d', vocab_size)
  logging.info('Example words in vocab: %s, %s', vocab[:3], vocab[-3:])
  _print_clients_number(train_clients, 'train')
  _print_clients_number(val_clients, 'validation')
  _print_clients_number(test_clients, 'test')
  datasets = collections.OrderedDict([
      (constants.TRAIN_CLIENTS_KEY, train_clients),
      (constants.VALID_CLIENTS_KEY, val_clients),
      (constants.TEST_CLIENTS_KEY, test_clients)
  ])

  num_oov_buckets, max_seq_len, max_elements = (_NUM_OOV_BUCKETS, _MAX_SEQ_LEN,
                                                _MAX_ELEMENTS_PER_CLIENT)
  train_preprocess_fn = _create_preprocess_fn(
      vocab,
      num_oov_buckets,
      client_batch_size=train_batch_size,
      client_epochs_per_round=num_local_epochs,
      max_sequence_length=max_seq_len,
      max_elements_per_client=max_elements,
      max_shuffle_buffer_size=max_elements)

  eval_preprocess_fn = _create_preprocess_fn(
      vocab,
      num_oov_buckets,
      client_batch_size=train_batch_size,
      client_epochs_per_round=1,
      max_sequence_length=max_seq_len,
      max_elements_per_client=-1,
      max_shuffle_buffer_size=1)

  @tf.function
  def split_data_fn(
      raw_data: tf.data.Dataset) -> OrderedDict[str, tf.data.Dataset]:
    """Process the raw data and split it equally into two unbatched datasets."""
    # `tff.learning.build_personalization_eval_computation` expects *unbatched*
    # client-side datasets. Batching is part of user-supplied personalization
    # function.
    processed_data = eval_preprocess_fn(raw_data).unbatch()
    personalization_data, test_data = emnist.split_half(processed_data)
    return collections.OrderedDict([(constants.PERSONALIZATION_DATA_KEY,
                                     personalization_data),
                                    (constants.TEST_DATA_KEY, test_data)])

  en_example_dataset = _get_example_dataset(train_clients, train_preprocess_fn,
                                            'en')
  es_example_dataset = _get_example_dataset(train_clients, train_preprocess_fn,
                                            'es')
  assert en_example_dataset.element_spec == es_example_dataset.element_spec
  input_spec = en_example_dataset.element_spec

  def model_fn() -> tff.learning.models.VariableModel:
    return tff.learning.models.from_keras_model(
        keras_model=_get_keras_model(vocab_size, num_oov_buckets),
        input_spec=input_spec,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=_get_metrics(vocab_size, num_oov_buckets=num_oov_buckets),
    )

  return model_fn, datasets, train_preprocess_fn, split_data_fn, _ACCURACY_NAME


def _create_sample_data_dictionary():
  """Returns small number of data samples for a synthetic problem."""
  return {
      'en-9_11_healing_the_mothers_who_found_forgiveness_friendship':
          collections.OrderedDict(
              language=[b'en'] * 3,
              talk_name=[
                  b'9_11_healing_the_mothers_who_found_forgiveness_friendship'
              ] * 3,
              translation=[
                  b'And it is .', b'( Applause )', b'You must be tolerant .'
              ]),
      'es-9_11_healing_the_mothers_who_found_forgiveness_friendship':
          collections.OrderedDict(
              language=[b'es'] * 3,
              talk_name=[
                  b'9_11_healing_the_mothers_who_found_forgiveness_friendship'
              ] * 3,
              translation=[
                  b'Y lo es .', b'( Aplausos )', b'Ella estaba nerviosa .'
              ]),
  }


def _synethetic_vocab() -> List[str]:
  return [
      'the',
      'and',
      'to',
      'of',
      'a',
      'that',
      'i',
      'in',
      'it',
      'you',
      'de',
      'que',
      'y',
      'la',
      'en',
      'el',
      'lo',
      'ella',
      'es',
      'quot',
  ]
