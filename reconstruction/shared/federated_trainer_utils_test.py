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
"""Tests for federated_trainer_utils.py."""

import tensorflow as tf

from reconstruction.shared import federated_trainer_utils


class FederatedTrainerUtilsTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    if tf.config.list_logical_devices('GPU'):
      self.skipTest('skip GPU test')

  def test_build_dataset_split_fn_none(self):
    # 3 batches.
    client_dataset = tf.data.Dataset.range(6).batch(2)

    split_dataset_fn = federated_trainer_utils.build_dataset_split_fn(
        recon_epochs_max=2,
        recon_epochs_constant=True,
        recon_steps_max=None,
        post_recon_epochs=1,
        post_recon_steps_max=None,
        split_dataset=False,
        split_dataset_strategy=None,
        split_dataset_proportion=None)
    # Round number shouldn't matter.
    recon_dataset, post_recon_dataset = split_dataset_fn(client_dataset, 3)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list,
                        [[0, 1], [2, 3], [4, 5], [0, 1], [2, 3], [4, 5]])
    self.assertAllEqual(post_recon_list, [[0, 1], [2, 3], [4, 5]])

  def test_build_dataset_split_fn_skip(self):
    # 3 batches.
    client_dataset = tf.data.Dataset.range(6).batch(2)

    split_dataset_fn = federated_trainer_utils.build_dataset_split_fn(
        recon_epochs_max=2,
        recon_epochs_constant=True,
        recon_steps_max=None,
        post_recon_epochs=1,
        post_recon_steps_max=None,
        split_dataset=True,
        split_dataset_strategy=federated_trainer_utils.SPLIT_STRATEGY_SKIP,
        split_dataset_proportion=2)
    # Round number shouldn't matter.
    recon_dataset, post_recon_dataset = split_dataset_fn(client_dataset, 3)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0, 1], [4, 5], [0, 1], [4, 5]])
    self.assertAllEqual(post_recon_list, [[2, 3]])

  def test_build_dataset_split_fn_aggregated(self):
    # 3 batches.
    client_dataset = tf.data.Dataset.range(6).batch(2)

    split_dataset_fn = federated_trainer_utils.build_dataset_split_fn(
        recon_epochs_max=2,
        recon_epochs_constant=True,
        recon_steps_max=None,
        post_recon_epochs=1,
        post_recon_steps_max=None,
        split_dataset=True,
        split_dataset_strategy=federated_trainer_utils
        .SPLIT_STRATEGY_AGGREGATED,
        split_dataset_proportion=2)
    # Round number shouldn't matter.
    recon_dataset, post_recon_dataset = split_dataset_fn(client_dataset, 3)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0, 1], [2, 3], [0, 1], [2, 3]])
    self.assertAllEqual(post_recon_list, [[4, 5]])

  def test_build_dataset_split_fn_none_recon_epochs_variable(self):
    # 3 batches.
    client_dataset = tf.data.Dataset.range(6).batch(2)

    split_dataset_fn = federated_trainer_utils.build_dataset_split_fn(
        recon_epochs_max=8,
        recon_epochs_constant=False,
        recon_steps_max=None,
        post_recon_epochs=1,
        post_recon_steps_max=None,
        split_dataset=False,
        split_dataset_strategy=None,
        split_dataset_proportion=None)

    round_num = tf.constant(1, dtype=tf.int64)
    recon_dataset, post_recon_dataset = split_dataset_fn(
        client_dataset, round_num)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0, 1], [2, 3], [4, 5]])
    self.assertAllEqual(post_recon_list, [[0, 1], [2, 3], [4, 5]])

    round_num = tf.constant(2, dtype=tf.int64)
    recon_dataset, post_recon_dataset = split_dataset_fn(
        client_dataset, round_num)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list,
                        [[0, 1], [2, 3], [4, 5], [0, 1], [2, 3], [4, 5]])
    self.assertAllEqual(post_recon_list, [[0, 1], [2, 3], [4, 5]])

  def test_build_dataset_split_fn_skip_recon_epochs_variable(self):
    # 3 batches.
    client_dataset = tf.data.Dataset.range(6).batch(2)

    split_dataset_fn = federated_trainer_utils.build_dataset_split_fn(
        recon_epochs_max=8,
        recon_epochs_constant=False,
        recon_steps_max=None,
        post_recon_epochs=1,
        post_recon_steps_max=None,
        split_dataset=True,
        split_dataset_strategy=federated_trainer_utils.SPLIT_STRATEGY_SKIP,
        split_dataset_proportion=2)

    round_num = tf.constant(1, dtype=tf.int64)
    recon_dataset, post_recon_dataset = split_dataset_fn(
        client_dataset, round_num)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0, 1], [4, 5]])
    self.assertAllEqual(post_recon_list, [[2, 3]])

    round_num = tf.constant(2, dtype=tf.int64)
    recon_dataset, post_recon_dataset = split_dataset_fn(
        client_dataset, round_num)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0, 1], [4, 5], [0, 1], [4, 5]])
    self.assertAllEqual(post_recon_list, [[2, 3]])

  def test_build_dataset_split_fn_aggregated_recon_epochs_variable(self):
    # 3 batches.
    client_dataset = tf.data.Dataset.range(6).batch(2)

    split_dataset_fn = federated_trainer_utils.build_dataset_split_fn(
        recon_epochs_max=8,
        recon_epochs_constant=False,
        recon_steps_max=None,
        post_recon_epochs=1,
        post_recon_steps_max=None,
        split_dataset=True,
        split_dataset_strategy=federated_trainer_utils
        .SPLIT_STRATEGY_AGGREGATED,
        split_dataset_proportion=3)

    round_num = tf.constant(1, dtype=tf.int64)
    recon_dataset, post_recon_dataset = split_dataset_fn(
        client_dataset, round_num)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0, 1], [2, 3]])
    self.assertAllEqual(post_recon_list, [[4, 5]])

    round_num = tf.constant(2, dtype=tf.int64)
    recon_dataset, post_recon_dataset = split_dataset_fn(
        client_dataset, round_num)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0, 1], [2, 3], [0, 1], [2, 3]])
    self.assertAllEqual(post_recon_list, [[4, 5]])

  def test_build_dataset_split_fn_none_recon_max_steps(self):
    # 3 batches.
    client_dataset = tf.data.Dataset.range(6).batch(2)

    split_dataset_fn = federated_trainer_utils.build_dataset_split_fn(
        recon_epochs_max=2,
        recon_epochs_constant=True,
        recon_steps_max=4,
        post_recon_epochs=1,
        post_recon_steps_max=None,
        split_dataset=False,
        split_dataset_strategy=None,
        split_dataset_proportion=None)
    # Round number shouldn't matter.
    recon_dataset, post_recon_dataset = split_dataset_fn(client_dataset, 3)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0, 1], [2, 3], [4, 5], [0, 1]])
    self.assertAllEqual(post_recon_list, [[0, 1], [2, 3], [4, 5]])

    # Adding more steps than the number of actual steps has no effect.
    split_dataset_fn = federated_trainer_utils.build_dataset_split_fn(
        recon_epochs_max=2,
        recon_epochs_constant=True,
        recon_steps_max=7,
        post_recon_epochs=1,
        post_recon_steps_max=None,
        split_dataset=False,
        split_dataset_strategy=None,
        split_dataset_proportion=None)
    # Round number shouldn't matter.
    recon_dataset, post_recon_dataset = split_dataset_fn(client_dataset, 3)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list,
                        [[0, 1], [2, 3], [4, 5], [0, 1], [2, 3], [4, 5]])
    self.assertAllEqual(post_recon_list, [[0, 1], [2, 3], [4, 5]])

  def test_build_dataset_split_fn_skip_recon_max_steps(self):
    # 3 batches.
    client_dataset = tf.data.Dataset.range(6).batch(2)

    split_dataset_fn = federated_trainer_utils.build_dataset_split_fn(
        recon_epochs_max=2,
        recon_epochs_constant=True,
        recon_steps_max=4,
        post_recon_epochs=1,
        post_recon_steps_max=None,
        split_dataset=True,
        split_dataset_strategy=federated_trainer_utils.SPLIT_STRATEGY_SKIP,
        split_dataset_proportion=3)
    # Round number shouldn't matter.
    recon_dataset, post_recon_dataset = split_dataset_fn(client_dataset, 3)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0, 1], [0, 1]])
    self.assertAllEqual(post_recon_list, [[2, 3], [4, 5]])

    # Adding more steps than the number of actual steps has no effect.
    split_dataset_fn = federated_trainer_utils.build_dataset_split_fn(
        recon_epochs_max=2,
        recon_epochs_constant=True,
        recon_steps_max=7,
        post_recon_epochs=1,
        post_recon_steps_max=None,
        split_dataset=True,
        split_dataset_strategy=federated_trainer_utils.SPLIT_STRATEGY_SKIP,
        split_dataset_proportion=3)
    # Round number shouldn't matter.
    recon_dataset, post_recon_dataset = split_dataset_fn(client_dataset, 3)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0, 1], [0, 1]])
    self.assertAllEqual(post_recon_list, [[2, 3], [4, 5]])

  def test_build_dataset_split_fn_aggregated_recon_max_steps(self):
    # 3 batches.
    client_dataset = tf.data.Dataset.range(6).batch(2)

    split_dataset_fn = federated_trainer_utils.build_dataset_split_fn(
        recon_epochs_max=2,
        recon_epochs_constant=True,
        recon_steps_max=4,
        post_recon_epochs=1,
        post_recon_steps_max=None,
        split_dataset=True,
        split_dataset_strategy=federated_trainer_utils
        .SPLIT_STRATEGY_AGGREGATED,
        split_dataset_proportion=2)
    # Round number shouldn't matter.
    recon_dataset, post_recon_dataset = split_dataset_fn(client_dataset, 3)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0, 1], [2, 3], [0, 1], [2, 3]])
    self.assertAllEqual(post_recon_list, [[4, 5]])

    # Adding more steps than the number of actual steps has no effect.
    split_dataset_fn = federated_trainer_utils.build_dataset_split_fn(
        recon_epochs_max=2,
        recon_epochs_constant=True,
        recon_steps_max=7,
        post_recon_epochs=1,
        post_recon_steps_max=None,
        split_dataset=True,
        split_dataset_strategy=federated_trainer_utils
        .SPLIT_STRATEGY_AGGREGATED,
        split_dataset_proportion=2)
    # Round number shouldn't matter.
    recon_dataset, post_recon_dataset = split_dataset_fn(client_dataset, 3)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0, 1], [2, 3], [0, 1], [2, 3]])
    self.assertAllEqual(post_recon_list, [[4, 5]])

  def test_build_dataset_split_fn_none_recon_epochs_variable_max_steps(self):
    # 3 batches.
    client_dataset = tf.data.Dataset.range(6).batch(2)

    split_dataset_fn = federated_trainer_utils.build_dataset_split_fn(
        recon_epochs_max=8,
        recon_epochs_constant=False,
        recon_steps_max=4,
        post_recon_epochs=1,
        post_recon_steps_max=None,
        split_dataset=False,
        split_dataset_strategy=None,
        split_dataset_proportion=None)

    round_num = tf.constant(1, dtype=tf.int64)
    recon_dataset, post_recon_dataset = split_dataset_fn(
        client_dataset, round_num)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0, 1], [2, 3], [4, 5]])
    self.assertAllEqual(post_recon_list, [[0, 1], [2, 3], [4, 5]])

    round_num = tf.constant(2, dtype=tf.int64)
    recon_dataset, post_recon_dataset = split_dataset_fn(
        client_dataset, round_num)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0, 1], [2, 3], [4, 5], [0, 1]])
    self.assertAllEqual(post_recon_list, [[0, 1], [2, 3], [4, 5]])

  def test_build_dataset_split_fn_skip_recon_epochs_variable_max_steps(self):
    # 3 batches.
    client_dataset = tf.data.Dataset.range(6).batch(2)

    split_dataset_fn = federated_trainer_utils.build_dataset_split_fn(
        recon_epochs_max=8,
        recon_epochs_constant=False,
        recon_steps_max=4,
        post_recon_epochs=1,
        post_recon_steps_max=None,
        split_dataset=True,
        split_dataset_strategy=federated_trainer_utils.SPLIT_STRATEGY_SKIP,
        split_dataset_proportion=2)

    round_num = tf.constant(1, dtype=tf.int64)
    recon_dataset, post_recon_dataset = split_dataset_fn(
        client_dataset, round_num)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0, 1], [4, 5]])
    self.assertAllEqual(post_recon_list, [[2, 3]])

    round_num = tf.constant(2, dtype=tf.int64)
    recon_dataset, post_recon_dataset = split_dataset_fn(
        client_dataset, round_num)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0, 1], [4, 5], [0, 1], [4, 5]])
    self.assertAllEqual(post_recon_list, [[2, 3]])

  def test_build_dataset_split_fn_aggregated_recon_epochs_variable_max_steps(
      self):
    # 3 batches.
    client_dataset = tf.data.Dataset.range(6).batch(2)

    split_dataset_fn = federated_trainer_utils.build_dataset_split_fn(
        recon_epochs_max=8,
        recon_epochs_constant=False,
        recon_steps_max=4,
        post_recon_epochs=1,
        post_recon_steps_max=None,
        split_dataset=True,
        split_dataset_strategy=federated_trainer_utils
        .SPLIT_STRATEGY_AGGREGATED,
        split_dataset_proportion=3)

    round_num = tf.constant(1, dtype=tf.int64)
    recon_dataset, post_recon_dataset = split_dataset_fn(
        client_dataset, round_num)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0, 1], [2, 3]])
    self.assertAllEqual(post_recon_list, [[4, 5]])

    round_num = tf.constant(2, dtype=tf.int64)
    recon_dataset, post_recon_dataset = split_dataset_fn(
        client_dataset, round_num)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0, 1], [2, 3], [0, 1], [2, 3]])
    self.assertAllEqual(post_recon_list, [[4, 5]])

  def test_build_dataset_split_fn_none_recon_epochs_variable_max_steps_zero_post_epochs(
      self):
    # 3 batches.
    client_dataset = tf.data.Dataset.range(6).batch(2)

    split_dataset_fn = federated_trainer_utils.build_dataset_split_fn(
        recon_epochs_max=8,
        recon_epochs_constant=False,
        recon_steps_max=4,
        post_recon_epochs=0,
        post_recon_steps_max=None,
        split_dataset=False,
        split_dataset_strategy=None,
        split_dataset_proportion=None)

    round_num = tf.constant(1, dtype=tf.int64)
    recon_dataset, post_recon_dataset = split_dataset_fn(
        client_dataset, round_num)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0, 1], [2, 3], [4, 5]])
    self.assertAllEqual(post_recon_list, [])

    round_num = tf.constant(2, dtype=tf.int64)
    recon_dataset, post_recon_dataset = split_dataset_fn(
        client_dataset, round_num)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0, 1], [2, 3], [4, 5], [0, 1]])
    self.assertAllEqual(post_recon_list, [])

  def test_build_dataset_split_fn_skip_recon_epochs_variable_max_steps_zero_post_epochs(
      self):
    # 3 batches.
    client_dataset = tf.data.Dataset.range(6).batch(2)

    split_dataset_fn = federated_trainer_utils.build_dataset_split_fn(
        recon_epochs_max=8,
        recon_epochs_constant=False,
        recon_steps_max=4,
        post_recon_epochs=0,
        post_recon_steps_max=None,
        split_dataset=True,
        split_dataset_strategy=federated_trainer_utils.SPLIT_STRATEGY_SKIP,
        split_dataset_proportion=2)

    round_num = tf.constant(1, dtype=tf.int64)
    recon_dataset, post_recon_dataset = split_dataset_fn(
        client_dataset, round_num)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0, 1], [4, 5]])
    self.assertAllEqual(post_recon_list, [])

    round_num = tf.constant(2, dtype=tf.int64)
    recon_dataset, post_recon_dataset = split_dataset_fn(
        client_dataset, round_num)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0, 1], [4, 5], [0, 1], [4, 5]])
    self.assertAllEqual(post_recon_list, [])

  def test_build_dataset_split_fn_aggregated_recon_epochs_variable_max_steps_zero_post_epochs(
      self):
    # 3 batches.
    client_dataset = tf.data.Dataset.range(6).batch(2)

    split_dataset_fn = federated_trainer_utils.build_dataset_split_fn(
        recon_epochs_max=8,
        recon_epochs_constant=False,
        recon_steps_max=4,
        post_recon_epochs=0,
        post_recon_steps_max=None,
        split_dataset=True,
        split_dataset_strategy=federated_trainer_utils
        .SPLIT_STRATEGY_AGGREGATED,
        split_dataset_proportion=3)

    round_num = tf.constant(1, dtype=tf.int64)
    recon_dataset, post_recon_dataset = split_dataset_fn(
        client_dataset, round_num)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0, 1], [2, 3]])
    self.assertAllEqual(post_recon_list, [])

    round_num = tf.constant(2, dtype=tf.int64)
    recon_dataset, post_recon_dataset = split_dataset_fn(
        client_dataset, round_num)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0, 1], [2, 3], [0, 1], [2, 3]])
    self.assertAllEqual(post_recon_list, [])

  def test_build_dataset_split_fn_none_recon_epochs_variable_max_steps_multiple_post_epochs(
      self):
    # 3 batches.
    client_dataset = tf.data.Dataset.range(6).batch(2)

    split_dataset_fn = federated_trainer_utils.build_dataset_split_fn(
        recon_epochs_max=8,
        recon_epochs_constant=False,
        recon_steps_max=4,
        post_recon_epochs=2,
        post_recon_steps_max=None,
        split_dataset=False,
        split_dataset_strategy=None,
        split_dataset_proportion=None)

    round_num = tf.constant(1, dtype=tf.int64)
    recon_dataset, post_recon_dataset = split_dataset_fn(
        client_dataset, round_num)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0, 1], [2, 3], [4, 5]])
    self.assertAllEqual(post_recon_list,
                        [[0, 1], [2, 3], [4, 5], [0, 1], [2, 3], [4, 5]])

    round_num = tf.constant(2, dtype=tf.int64)
    recon_dataset, post_recon_dataset = split_dataset_fn(
        client_dataset, round_num)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0, 1], [2, 3], [4, 5], [0, 1]])
    self.assertAllEqual(post_recon_list,
                        [[0, 1], [2, 3], [4, 5], [0, 1], [2, 3], [4, 5]])

  def test_build_dataset_split_fn_skip_recon_epochs_variable_max_steps_multiple_post_epochs(
      self):
    # 3 batches.
    client_dataset = tf.data.Dataset.range(6).batch(2)

    split_dataset_fn = federated_trainer_utils.build_dataset_split_fn(
        recon_epochs_max=8,
        recon_epochs_constant=False,
        recon_steps_max=4,
        post_recon_epochs=2,
        post_recon_steps_max=None,
        split_dataset=True,
        split_dataset_strategy=federated_trainer_utils.SPLIT_STRATEGY_SKIP,
        split_dataset_proportion=2)

    round_num = tf.constant(1, dtype=tf.int64)
    recon_dataset, post_recon_dataset = split_dataset_fn(
        client_dataset, round_num)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0, 1], [4, 5]])
    self.assertAllEqual(post_recon_list, [[2, 3], [2, 3]])

    round_num = tf.constant(2, dtype=tf.int64)
    recon_dataset, post_recon_dataset = split_dataset_fn(
        client_dataset, round_num)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0, 1], [4, 5], [0, 1], [4, 5]])
    self.assertAllEqual(post_recon_list, [[2, 3], [2, 3]])

  def test_build_dataset_split_fn_aggregated_recon_epochs_variable_max_steps_multiple_post_epochs(
      self):
    # 3 batches.
    client_dataset = tf.data.Dataset.range(6).batch(2)

    split_dataset_fn = federated_trainer_utils.build_dataset_split_fn(
        recon_epochs_max=8,
        recon_epochs_constant=False,
        recon_steps_max=4,
        post_recon_epochs=2,
        post_recon_steps_max=None,
        split_dataset=True,
        split_dataset_strategy=federated_trainer_utils
        .SPLIT_STRATEGY_AGGREGATED,
        split_dataset_proportion=2)

    round_num = tf.constant(1, dtype=tf.int64)
    recon_dataset, post_recon_dataset = split_dataset_fn(
        client_dataset, round_num)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0, 1], [2, 3]])
    self.assertAllEqual(post_recon_list, [[4, 5], [4, 5]])

    round_num = tf.constant(2, dtype=tf.int64)
    recon_dataset, post_recon_dataset = split_dataset_fn(
        client_dataset, round_num)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0, 1], [2, 3], [0, 1], [2, 3]])
    self.assertAllEqual(post_recon_list, [[4, 5], [4, 5]])

  def test_build_dataset_split_fn_none_post_recon_multiple_epochs_max_steps(
      self):
    # 3 batches.
    client_dataset = tf.data.Dataset.range(6).batch(2)

    split_dataset_fn = federated_trainer_utils.build_dataset_split_fn(
        recon_epochs_max=1,
        recon_epochs_constant=True,
        recon_steps_max=None,
        post_recon_epochs=2,
        post_recon_steps_max=4,
        split_dataset=False,
        split_dataset_strategy=None,
        split_dataset_proportion=None)

    # Round number doesn't matter.
    round_num = tf.constant(1, dtype=tf.int64)
    recon_dataset, post_recon_dataset = split_dataset_fn(
        client_dataset, round_num)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0, 1], [2, 3], [4, 5]])
    self.assertAllEqual(post_recon_list, [[0, 1], [2, 3], [4, 5], [0, 1]])

    # Round number doesn't matter.
    round_num = tf.constant(2, dtype=tf.int64)
    recon_dataset, post_recon_dataset = split_dataset_fn(
        client_dataset, round_num)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0, 1], [2, 3], [4, 5]])
    self.assertAllEqual(post_recon_list, [[0, 1], [2, 3], [4, 5], [0, 1]])

  def test_build_dataset_split_fn_skip_post_recon_multiple_epochs_max_steps(
      self):
    # 3 batches.
    client_dataset = tf.data.Dataset.range(6).batch(2)

    split_dataset_fn = federated_trainer_utils.build_dataset_split_fn(
        recon_epochs_max=1,
        recon_epochs_constant=True,
        recon_steps_max=None,
        post_recon_epochs=2,
        post_recon_steps_max=4,
        split_dataset=True,
        split_dataset_strategy=federated_trainer_utils.SPLIT_STRATEGY_SKIP,
        split_dataset_proportion=2)

    # Round number doesn't matter.
    round_num = tf.constant(1, dtype=tf.int64)
    recon_dataset, post_recon_dataset = split_dataset_fn(
        client_dataset, round_num)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0, 1], [4, 5]])
    self.assertAllEqual(post_recon_list, [[2, 3], [2, 3]])

    # Round number doesn't matter.
    round_num = tf.constant(2, dtype=tf.int64)
    recon_dataset, post_recon_dataset = split_dataset_fn(
        client_dataset, round_num)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0, 1], [4, 5]])
    self.assertAllEqual(post_recon_list, [[2, 3], [2, 3]])

  def test_build_dataset_split_fn_aggregated_post_recon_multiple_epochs_max_steps(
      self):
    # 3 batches.
    client_dataset = tf.data.Dataset.range(6).batch(2)

    split_dataset_fn = federated_trainer_utils.build_dataset_split_fn(
        recon_epochs_max=1,
        recon_epochs_constant=True,
        recon_steps_max=None,
        post_recon_epochs=2,
        post_recon_steps_max=4,
        split_dataset=True,
        split_dataset_strategy=federated_trainer_utils
        .SPLIT_STRATEGY_AGGREGATED,
        split_dataset_proportion=2)

    # Round number doesn't matter.
    round_num = tf.constant(1, dtype=tf.int64)
    recon_dataset, post_recon_dataset = split_dataset_fn(
        client_dataset, round_num)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0, 1], [2, 3]])
    self.assertAllEqual(post_recon_list, [[4, 5], [4, 5]])

    # Round number doesn't matter.
    round_num = tf.constant(2, dtype=tf.int64)
    recon_dataset, post_recon_dataset = split_dataset_fn(
        client_dataset, round_num)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [[0, 1], [2, 3]])
    self.assertAllEqual(post_recon_list, [[4, 5], [4, 5]])

  def test_build_dataset_split_none_fn_split_dataset_zero_batches(self):
    """Ensures clients without any data don't fail."""
    # 0 batches.
    client_dataset = tf.data.Dataset.range(0).batch(2)

    split_dataset_fn = federated_trainer_utils.build_dataset_split_fn(
        recon_epochs_max=1,
        recon_epochs_constant=True,
        recon_steps_max=None,
        post_recon_epochs=1,
        post_recon_steps_max=None,
        split_dataset=False,
        split_dataset_strategy=None,
        split_dataset_proportion=None)

    # Round number doesn't matter.
    round_num = tf.constant(1, dtype=tf.int64)
    recon_dataset, post_recon_dataset = split_dataset_fn(
        client_dataset, round_num)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [])
    self.assertAllEqual(post_recon_list, [])

    # Round number doesn't matter.
    round_num = tf.constant(2, dtype=tf.int64)
    recon_dataset, post_recon_dataset = split_dataset_fn(
        client_dataset, round_num)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [])
    self.assertAllEqual(post_recon_list, [])

  def test_build_dataset_split_skip_fn_split_dataset_zero_batches(self):
    """Ensures clients without any data don't fail."""
    # 0 batches.
    client_dataset = tf.data.Dataset.range(0).batch(2)

    split_dataset_fn = federated_trainer_utils.build_dataset_split_fn(
        recon_epochs_max=1,
        recon_epochs_constant=True,
        recon_steps_max=None,
        post_recon_epochs=1,
        post_recon_steps_max=None,
        split_dataset=True,
        split_dataset_strategy=federated_trainer_utils.SPLIT_STRATEGY_SKIP,
        split_dataset_proportion=10)

    # Round number doesn't matter.
    round_num = tf.constant(1, dtype=tf.int64)
    recon_dataset, post_recon_dataset = split_dataset_fn(
        client_dataset, round_num)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [])
    self.assertAllEqual(post_recon_list, [])

    # Round number doesn't matter.
    round_num = tf.constant(2, dtype=tf.int64)
    recon_dataset, post_recon_dataset = split_dataset_fn(
        client_dataset, round_num)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [])
    self.assertAllEqual(post_recon_list, [])

  def test_build_dataset_split_aggregated_fn_split_dataset_zero_batches(self):
    """Ensures clients without any data don't fail."""
    # 0 batches.
    client_dataset = tf.data.Dataset.range(0).batch(2)

    split_dataset_fn = federated_trainer_utils.build_dataset_split_fn(
        recon_epochs_max=1,
        recon_epochs_constant=True,
        recon_steps_max=None,
        post_recon_epochs=1,
        post_recon_steps_max=None,
        split_dataset=True,
        split_dataset_strategy=federated_trainer_utils
        .SPLIT_STRATEGY_AGGREGATED,
        split_dataset_proportion=10)

    # Round number doesn't matter.
    round_num = tf.constant(1, dtype=tf.int64)
    recon_dataset, post_recon_dataset = split_dataset_fn(
        client_dataset, round_num)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [])
    self.assertAllEqual(post_recon_list, [])

    # Round number doesn't matter.
    round_num = tf.constant(2, dtype=tf.int64)
    recon_dataset, post_recon_dataset = split_dataset_fn(
        client_dataset, round_num)

    recon_list = list(recon_dataset.as_numpy_iterator())
    post_recon_list = list(post_recon_dataset.as_numpy_iterator())

    self.assertAllEqual(recon_list, [])
    self.assertAllEqual(post_recon_list, [])


if __name__ == '__main__':
  tf.test.main()
