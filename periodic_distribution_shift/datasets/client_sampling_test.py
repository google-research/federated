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

from absl import flags
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from periodic_distribution_shift.datasets import client_sampling

flags.DEFINE_integer(
    'stackoverflow_word_vocab_size', 10000, 'Vocabulary size.')
flags.DEFINE_integer(
    'stackoverflow_word_sequence_length', 20, 'Sequence length.')


class SamplingTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': '_emnist_seeded',
          'shift_fn': 'linear',
          'seed': range(3),
      }, {
          'testcase_name': '_emnist_no_seed',
          'shift_fn': 'linear',
          'seed': None,
      },
      {
          'testcase_name': '_emnist_cosine_seeded',
          'shift_fn': 'cosine',
          'seed': range(3),
      }, {
          'testcase_name': '_emnist_cosine_no_seed',
          'shift_fn': 'cosine',
          'seed': None,
      },)
  def test_build_emnist_with_random_seed(self, seed, shift_fn):
    random_seed = seed
    train_client_spec = tff.simulation.baselines.ClientSpec(
        num_epochs=1,
        batch_size=4,
        max_elements=4)

    sample_fn_1 = client_sampling.build_time_varying_dataset_fn(
        train_client_spec,
        clients_per_round=4,
        period=8,
        shift_fn=shift_fn,
        shift_p=1.,
        task_name='emnist_character',
        random_seed=random_seed)
    g1_datasets = sample_fn_1(round_num=0)
    g2_datasets = sample_fn_1(round_num=4)
    g2_labels_in_g1, g1_labels_in_g2 = 0, 0
    for dataset in g1_datasets:
      for sample in dataset:
        # g1 should only be digits, with label < 10
        g2_labels_in_g1 += np.sum(sample[1].numpy() >= 10)

    for dataset in g2_datasets:
      for sample in dataset:
        # g2 should only be non-digits, with label >= 10
        g1_labels_in_g2 += np.sum(sample[1].numpy() < 10)

    self.assertEqual(g2_labels_in_g1, 0)
    self.assertEqual(g1_labels_in_g2, 0)

  @parameterized.named_parameters(
      {
          'testcase_name': '_cifar_seeded',
          'shift_fn': 'linear',
          'seed': range(3),
      }, {
          'testcase_name': '_cifar_no_seed',
          'shift_fn': 'linear',
          'seed': None,
      },
      {
          'testcase_name': '_cifar_cosine_seeded',
          'shift_fn': 'cosine',
          'seed': range(3),
      },
      {
          'testcase_name': '_cifar_cosine_no_seed',
          'shift_fn': 'cosine',
          'seed': None,
      },
  )
  def test_build_cifar_with_random_seed(self, seed, shift_fn):
    random_seed = seed
    train_client_spec = tff.simulation.baselines.ClientSpec(
        num_epochs=1,
        batch_size=4,
        max_elements=4)

    sample_fn_1 = client_sampling.build_time_varying_dataset_fn(
        train_client_spec,
        clients_per_round=4,
        period=8,
        shift_fn=shift_fn,
        shift_p=1.,
        task_name='cifar100_10',
        random_seed=random_seed)
    g1_datasets = sample_fn_1(round_num=0)
    g2_datasets = sample_fn_1(round_num=4)
    g2_labels_in_g1, g1_labels_in_g2 = 0, 0
    for dataset in g1_datasets:
      for sample in dataset:
        # g1 should only be cifar10, with label >= 100
        g2_labels_in_g1 += np.sum(sample[1].numpy() < 100)

    for dataset in g2_datasets:
      for sample in dataset:
        # g2 should only be cifar100, with label < 100
        g1_labels_in_g2 += np.sum(sample[1].numpy() >= 100)

    self.assertEqual(g2_labels_in_g1, 0)
    self.assertEqual(g1_labels_in_g2, 0)

  @parameterized.named_parameters(
      {
          'testcase_name': '_so_seeded',
          'shift_fn': 'linear',
          'seed': range(3),
      }, {
          'testcase_name': '_so_no_seed',
          'shift_fn': 'linear',
          'seed': None,
      },
      {
          'testcase_name': '_so_cosine_seeded',
          'shift_fn': 'cosine',
          'seed': range(3),
      }, {
          'testcase_name': '_so_cosine_no_seed',
          'shift_fn': 'cosine',
          'seed': None,
      },)
  def test_build_so_with_random_seed(self, seed, shift_fn):
    random_seed = seed
    train_client_spec = tff.simulation.baselines.ClientSpec(
        num_epochs=1,
        batch_size=4,
        max_elements=4)

    sample_fn_1 = client_sampling.build_time_varying_dataset_fn(
        train_client_spec,
        clients_per_round=4,
        period=8,
        shift_fn=shift_fn,
        shift_p=1.,
        task_name='stackoverflow_word',
        random_seed=random_seed)
    g1_datasets = sample_fn_1(round_num=0)
    g2_datasets = sample_fn_1(round_num=4)
    g2_labels_in_g1, g1_labels_in_g2 = 0, 0
    for dataset in g1_datasets:
      for sample in dataset:
        # g1 should only be group 0 (questions)
        g2_labels_in_g1 += np.sum(sample[0][1].numpy() == 1)

    for dataset in g2_datasets:
      for sample in dataset:
        # g2 should only be group 1 (answers)
        g1_labels_in_g2 += np.sum(sample[0][1].numpy() == 0)

    self.assertEqual(g2_labels_in_g1, 0)
    self.assertEqual(g1_labels_in_g2, 0)

if __name__ == '__main__':
  tf.test.main()
