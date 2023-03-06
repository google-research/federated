# Copyright 2023, Google LLC.
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
"""Tests for bandits_process."""
import collections
import functools
from absl.testing import parameterized

import tensorflow as tf
import tensorflow_federated as tff

from bandits import bandits_process
from bandits import bandits_utils
from bandits.algs import epsilon_greedy
from bandits.tasks import emnist
from bandits.tasks import task_utils

_IMAGE_SIZE = 28
_SUPERVISED_DATA_SPEC = (
    tf.TensorSpec(shape=(None, _IMAGE_SIZE, _IMAGE_SIZE, 1), dtype=tf.float32),
    tf.TensorSpec(shape=(None,), dtype=tf.int32),
)
_BANDITS_DATA_SPEC = collections.OrderedDict(
    x=tf.TensorSpec(
        shape=(None, _IMAGE_SIZE, _IMAGE_SIZE, 1), dtype=tf.float32
    ),
    y=collections.OrderedDict(
        label=tf.TensorSpec(shape=(None,), dtype=tf.int32),
        action=tf.TensorSpec(shape=(None,), dtype=tf.int32),
        reward=tf.TensorSpec(shape=(None,), dtype=tf.float32),
        prob=tf.TensorSpec(shape=(None,), dtype=tf.float32),
        weight_scale=tf.TensorSpec(shape=(), dtype=tf.float32),
    ),
)


def _get_synthetic_emnist_task():
  model_fn = emnist.create_emnist_bandits_model_fn(
      _BANDITS_DATA_SPEC, loss=task_utils.SupervisedCELoss()
  )
  datasets = emnist.create_emnist_preprocessed_datasets(
      train_client_batch_size=16,
      test_client_batch_size=32,
      train_shuffle_buffer_size=1,
      train_client_max_elements=None,
      only_digits=True,
      use_synthetic_data=True,
  )
  return model_fn, datasets


def _bandit_identity_data_fn(model, model_weights, dataset, zero_reward=True):
  if zero_reward:
    del model, model_weights  # unused
    return bandits_utils.dataset_format_map(dataset)
  else:
    bandit_fn, _ = epsilon_greedy.build_epsilon_greedy_bandit_data_fn(
        _SUPERVISED_DATA_SPEC, epsilon=0, reward_fn=None
    )
    return bandit_fn(model, model_weights, dataset)


class BanditsProcessTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('zero_reward', True), ('greedy_reward', False)
  )
  def test_supervised_training(self, zero_reward):
    tf.random.set_seed(42)
    train2infer_frequency = 2
    model_fn, datasets = _get_synthetic_emnist_task()
    fedavg_process = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn=model_fn,
        client_optimizer_fn=functools.partial(
            tf.keras.optimizers.SGD, learning_rate=0.1
        ),
    )
    # The bandits inference returns the original dataset, and hence the train
    # state and metrics are similar to a vanilla FedAvg process.
    bandit_fn = functools.partial(
        _bandit_identity_data_fn, zero_reward=zero_reward
    )
    bandit_process = bandits_process.build_bandits_iterative_process(
        model_fn=model_fn,
        training_process=fedavg_process,
        train2infer_frequency=train2infer_frequency,
        data_element_spec=_SUPERVISED_DATA_SPEC,
        bandit_data_fn=bandit_fn,
    )
    bandit_state = bandit_process.initialize()
    fedavg_state = bandit_state.train_state
    with tf.device('/device:cpu:0'):
      model = model_fn()
    round_reward_list = []
    for round_num in range(5):
      clients = datasets.sample_train_clients(num_clients=2, replace=True)
      mapped_clients = [
          bandit_fn(model, bandit_state.delayed_inference_model, ds)
          for ds in clients
      ]
      fedavg_output = fedavg_process.next(fedavg_state, mapped_clients)
      fedavg_state = fedavg_output.state
      fedavg_metrics = fedavg_output.metrics
      bandit_state, bandit_metrics = bandit_process.next(bandit_state, clients)
      self.assertAllClose(
          tf.nest.flatten(fedavg_state),
          tf.nest.flatten(bandit_state.train_state),
          rtol=1e-5,
      )
      self.assertAllClose(
          fedavg_metrics, bandit_metrics['optimization'], rtol=1e-5
      )
      round_reward = bandit_metrics['bandits']['round_rewards']
      round_prob = bandit_metrics['bandits']['round_prob']
      if zero_reward:
        self.assertEqual(0, round_reward)
        self.assertEqual(1, round_prob)
      else:
        if round_num % train2infer_frequency == 0:
          acc = fedavg_metrics['client_work']['train'][
              'sparse_categorical_accuracy'
          ]
          estimated_reward = (
              bandits_utils.REWARD_WRONG * (1 - acc)
              + bandits_utils.REWARD_RIGHT * acc
          )
        self.assertNear(round_reward, estimated_reward, err=1e-5)

        self.assertGreaterEqual(round_prob, 0)
        self.assertLessEqual(round_prob, 1)
      round_reward_list.append(round_reward)
      average_reward = bandit_metrics['bandits']['average_rewards']
      self.assertNear(
          sum(round_reward_list) / len(round_reward_list),
          average_reward,
          err=1e-5,
      )

  def test_deploy_model(self):
    tf.random.set_seed(42)
    train2infer_frequency = 2
    model_fn, datasets = _get_synthetic_emnist_task()
    fedavg_process = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn=model_fn, client_optimizer_fn=tf.keras.optimizers.SGD
    )
    bandit_process = bandits_process.build_bandits_iterative_process(
        model_fn=model_fn,
        training_process=fedavg_process,
        train2infer_frequency=train2infer_frequency,
        data_element_spec=_SUPERVISED_DATA_SPEC,
        bandit_data_fn=_bandit_identity_data_fn,
    )
    bandit_state = bandit_process.initialize()
    for round_num in range(1, 6):
      clients = datasets.sample_train_clients(num_clients=2, replace=True)
      bandit_state, _ = bandit_process.next(bandit_state, clients)
      self.assertEqual(bandit_state.round_num, round_num)
      if round_num % train2infer_frequency == 0:
        self.assertAllClose(
            tf.nest.flatten(bandit_state.delayed_inference_model),
            tf.nest.flatten(bandit_state.train_state.global_model_weights),
        )
      else:
        self.assertNotAllClose(
            tf.nest.flatten(bandit_state.delayed_inference_model),
            tf.nest.flatten(bandit_state.train_state.global_model_weights),
        )


if __name__ == '__main__':
  tf.test.main()
