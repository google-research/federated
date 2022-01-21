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
"""Client sampling for periodic distribution shift."""
import math
from typing import Callable, Optional, List
from absl import flags

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from periodic_distribution_shift.datasets import cifar_classification_preprocessing
from periodic_distribution_shift.datasets import emnist_preprocessing
from periodic_distribution_shift.datasets import stackoverflow_nwp_preprocessing as word_prediction_preprocessing
from utils.datasets import cifar10_dataset

#  Settings for a multiplicative linear congruential generator (aka Lehmer
#  generator) suggested in 'Random Number Generators: Good
#  Ones are Hard to Find' by Park and Miller.
MLCG_MODULUS = 2**31 - 1
MLCG_MULTIPLIER = 16807


class ClientDatasetSampler(object):
  """Sampling clients according to periodically shifting distribution.

  We simulate the daytime-nighttime shifts, so the period represents the 24-hour
  cycle. The simulation always starts from the peak where all clients are from
  `dataset1`.

  Attributes:
    clients_per_round: Number of clients participating training in each round.
    period: Period of the distribution shift.
    shift_fn: Function type of distribution shift.
    shift_p: The exponent used to control the bias of the distribution.
    dataset1: The dataset for clients from the daytime mode.
    dataset2: The dataset for clients from the nighttime mode.
    random_seed: Optional random seed for data sampling.

  """

  def __init__(
      self,
      clients_per_round: int,
      period: int,
      shift_fn: str,
      shift_p: float,
      dataset1: tff.simulation.datasets.ClientData,
      dataset2: tff.simulation.datasets.ClientData,
      random_seed=None):
    self._clients_per_round = clients_per_round
    self._period = period
    self._shift_fn = shift_fn
    self._shift_p = shift_p

    self._dataset1 = dataset1
    self._dataset2 = dataset2
    self._g1_cids = list(dataset1.client_ids)
    self._g2_cids = list(dataset2.client_ids)

    self.random_seed = random_seed
    if isinstance(random_seed, int):
      self._mlcg_start = np.random.RandomState(random_seed).randint(
          1, MLCG_MODULUS - 1)

  def get_pseudo_random_int(self, round_num):
    return pow(MLCG_MULTIPLIER, round_num,
               MLCG_MODULUS) * self._mlcg_start % MLCG_MODULUS

  def sample_client_datasets(self, round_num: int) -> List[tf.data.Dataset]:
    """Returns sampled client datasets.

    This function can be used as `client_datasets_fn` in `training_loop.run`.

    Args:
      round_num: the current round index.

    Returns:
      A list of daytime and nighttime client datasets.
    """
    if isinstance(self.random_seed, int):
      random_state = np.random.RandomState(
          self.get_pseudo_random_int(round_num))
    else:
      random_state = np.random.RandomState()

    proc = round_num % self._period / self._period
    if self._shift_fn == 'linear':
      ratio = abs(proc - 0.5) * 2
    elif self._shift_fn == 'cosine':
      ratio = 0.5 * (math.cos(2 * math.pi * proc) + 1)
    else:
      raise ValueError(f'Shift function type {self._shift_fn} not defined. '
                       f'Expecting `linear` or `cosine`.')

    prob = ratio ** self._shift_p

    g1_num = random_state.binomial(self._clients_per_round, prob)
    g2_num = self._clients_per_round - g1_num

    g1_sampled_ids = random_state.choice(
        self._g1_cids, g1_num, replace=False).tolist()
    g2_sampled_ids = random_state.choice(
        self._g2_cids, g2_num, replace=False).tolist()

    daytime_list = [self._dataset1.create_tf_dataset_for_client(cid)
                    for cid in g1_sampled_ids]
    nighttime_list = [self._dataset2.create_tf_dataset_for_client(cid)
                      for cid in g2_sampled_ids]

    return daytime_list + nighttime_list


def create_emnist_shifting_dataset(
    train_client_spec,
    clients_per_round: int,
    period: int,
    shift_fn: str,
    shift_p: float,
    random_seed: Optional[int] = None) -> ClientDatasetSampler:
  """Creates the client sampler with distribution shift for EMNIST.

  Args:
    train_client_spec: Specifying the batch size, epochs and max elements.
    clients_per_round: Number of clients participating training in each round.
    period: Period of the distribution shift.
    shift_fn: Function type of distribution shift.
    shift_p: The exponent used to control the bias of the distribution.
    random_seed: Optional random seed for data sampling.

  Returns:
    A list of client datasets for federated training.
  """
  digits_prepro_fn = emnist_preprocessing.create_preprocess_fn(
      train_client_spec,
      emnist_task='character_recognition',
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
      filter_fn=None)
  digits_set, _ = tff.simulation.datasets.emnist.load_data(
      only_digits=True)
  digits_set = digits_set.preprocess(digits_prepro_fn)

  chars_prepro_fn = emnist_preprocessing.create_preprocess_fn(
      train_client_spec,
      emnist_task='character_recognition',
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
      filter_fn=lambda x: x['label'] >= 10)
  chars_set, _ = tff.simulation.datasets.emnist.load_data(
      only_digits=False)
  chars_set = chars_set.preprocess(chars_prepro_fn)
  return ClientDatasetSampler(
      clients_per_round,
      period,
      shift_fn,
      shift_p,
      dataset1=digits_set,
      dataset2=chars_set,
      random_seed=random_seed)


def create_stackoverflow_shifting_dataset(
    train_client_spec,
    clients_per_round: int,
    period: int,
    shift_fn: str,
    shift_p: float,
    random_seed: Optional[int] = None) -> ClientDatasetSampler:
  """Creates the client sampler with distribution shift for Stack Overflow.

  Args:
    train_client_spec: Specifying the batch size, epochs and max elements.
    clients_per_round: Number of clients participating training in each round.
    period: Period of the distribution shift.
    shift_fn: Function type of distribution shift.
    shift_p: The exponent used to control the bias of the distribution.
    random_seed: Optional random seed for data sampling.

  Returns:
    A list of client datasets for federated training.
  """
  vocab = list(tff.simulation.datasets.stackoverflow.load_word_counts(
      vocab_size=flags.FLAGS.stackoverflow_word_vocab_size).keys())

  question_prepro_fn = word_prediction_preprocessing.create_preprocess_fn(
      preprocess_spec=train_client_spec,
      vocab=vocab,
      sequence_length=flags.FLAGS.stackoverflow_word_sequence_length,
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
      filter_fn=lambda x: x['type'] == 'question')
  train_set, _, _ = tff.simulation.datasets.stackoverflow.load_data()
  question_set = train_set.preprocess(question_prepro_fn)

  answer_prepro_fn = word_prediction_preprocessing.create_preprocess_fn(
      preprocess_spec=train_client_spec,
      vocab=vocab,
      sequence_length=flags.FLAGS.stackoverflow_word_sequence_length,
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
      filter_fn=lambda x: x['type'] == 'answer')
  answer_set = train_set.preprocess(answer_prepro_fn)

  return ClientDatasetSampler(
      clients_per_round,
      period,
      shift_fn,
      shift_p,
      dataset1=question_set,
      dataset2=answer_set,
      random_seed=random_seed)


def create_cifar_shifting_dataset(
    train_client_spec,
    clients_per_round: int,
    period: int,
    shift_fn: str,
    shift_p: float,
    random_seed: Optional[int] = None) -> ClientDatasetSampler:
  """Creates the client sampler with distribution shift for CIFAR.

  Args:
    train_client_spec: Specifying the batch size, epochs and max elements.
    clients_per_round: Number of clients participating training in each round.
    period: Period of the distribution shift.
    shift_fn: Function type of distribution shift.
    shift_p: The exponent used to control the bias of the distribution.
    random_seed: Optional random seed for data sampling.

  Returns:
    A list of client datasets for federated training.
  """
  train_cifar100, _ = tff.simulation.datasets.cifar100.load_data()
  train_cifar10, _ = cifar10_dataset.load_cifar10_federated(
      num_clients=500)

  crop_shape = (24, 24, 3)
  train_cifar10_prep = cifar_classification_preprocessing.create_preprocess_fn(
      train_client_spec, crop_shape=crop_shape, is_cifar10=True)
  train_cifar100_prep = cifar_classification_preprocessing.create_preprocess_fn(
      train_client_spec, crop_shape=crop_shape, is_cifar10=False)

  train_cifar10 = train_cifar10.preprocess(train_cifar10_prep)
  train_cifar100 = train_cifar100.preprocess(train_cifar100_prep)

  return ClientDatasetSampler(
      clients_per_round,
      period,
      shift_fn,
      shift_p,
      dataset1=train_cifar10,
      dataset2=train_cifar100,
      random_seed=random_seed)


def build_time_varying_dataset_fn(
    train_client_spec,
    clients_per_round: int,
    period: int = 32,
    shift_fn: str = 'linear',
    shift_p: float = 1.,
    task_name: str = 'emnist_character',
    random_seed: Optional[int] = None,
) -> Callable[[int], List[tf.data.Dataset]]:
  """Creates the client sampler with periodic distribution shift.

  Args:
    train_client_spec: Specifying the batch size, epochs and max elements.
    clients_per_round: Number of clients participating training in each round.
    period: Period of the distribution shift.
    shift_fn: Function type of distribution shift.
    shift_p: The exponent used to control the bias of the distribution.
    task_name: The task name, corresponding to different datasets.
    random_seed: Optional random seed for data sampling.

  Returns:
    A list of datasets for sampled clients.

  """
  if task_name == 'emnist_character':
    dataset_fn = create_emnist_shifting_dataset
  elif task_name == 'stackoverflow_word':
    dataset_fn = create_stackoverflow_shifting_dataset
  elif task_name == 'cifar100_10':
    dataset_fn = create_cifar_shifting_dataset
  else:
    raise ValueError(f'Distribution shift dataset function has not been '
                     f'implemented for task {task_name}.')

  dataset_sampler = dataset_fn(
      train_client_spec=train_client_spec,
      clients_per_round=clients_per_round,
      period=period,
      shift_fn=shift_fn,
      shift_p=shift_p,
      random_seed=random_seed)

  return dataset_sampler.sample_client_datasets
