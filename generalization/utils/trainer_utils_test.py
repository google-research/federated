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
"""Tests for trainer_utils.py."""

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_federated as tff

from generalization.utils import eval_metric_distribution
from generalization.utils import trainer_utils


def keras_model_builder_with_zeros():
  # Create a simple linear regression model, single output.
  # We initialize all weights to zero.
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(
          1,
          kernel_initializer='zeros',
          bias_initializer='zeros',
          input_shape=(1,))
  ])
  return model


def keras_model_builder_with_ones():
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(
          1,
          kernel_initializer='ones',
          bias_initializer='ones',
          input_shape=(1,))
  ])
  return model


def create_dataset():
  # Create data satisfying y = 2*x + 1
  x = [[1.0], [2.0], [3.0]]
  y = [[3.0], [5.0], [7.0]]
  return tf.data.Dataset.from_tensor_slices((x, y)).batch(1)


def create_federated_cd():
  x1 = [[1.0]]
  y1 = [[3.0]]
  dataset1 = (x1, y1)
  x2 = [[2.0]]
  y2 = [[5.0]]
  dataset2 = (x2, y2)
  x3 = [[3.0]]
  y3 = [[7.0]]
  dataset3 = (x3, y3)
  return tff.simulation.datasets.TestClientData({
      1: dataset1,
      2: dataset2,
      3: dataset3
  }).preprocess(lambda ds: ds.batch(1))


def get_input_spec():
  return create_dataset().element_spec


def metrics_builder():
  return [tf.keras.metrics.MeanSquaredError()]


def tff_model_builder():
  return tff.learning.from_keras_model(
      keras_model=keras_model_builder_with_zeros(),
      input_spec=get_input_spec(),
      loss=tf.keras.losses.MeanSquaredError(),
      metrics=metrics_builder())


class CreateEvalFnsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('with_test_cd', True),
                                  ('without_test_cd', False))
  def test_create_federated_eval_fns(self, use_test_cd):
    """Test for create_federated_eval_fns."""

    (part_train_eval_fn, part_val_fn, unpart_fn,
     test_fn) = trainer_utils.create_federated_eval_fns(
         tff_model_builder=tff_model_builder,
         metrics_builder=metrics_builder,
         part_train_eval_cd=create_federated_cd(),
         part_val_cd=create_federated_cd(),
         unpart_cd=create_federated_cd(),
         test_cd=create_federated_cd() if use_test_cd else None,
         stat_fns=eval_metric_distribution.ALL_STAT_FNS,
         rounds_per_eval=1,
         part_clients_per_eval=2,
         unpart_clients_per_eval=2,
         test_clients_for_eval=3,
         resample_eval_clients=False,
         eval_clients_random_seed=1)
    keras_model = keras_model_builder_with_zeros()
    model_weights = tff.learning.ModelWeights.from_model(keras_model)
    server_state = tff.learning.framework.ServerState(model_weights, [], [], [])

    expected_keys = [
        f'mean_squared_error/{s}' for s in eval_metric_distribution.ALL_STAT_FNS
    ]

    # Federated validation fn requires a positional arg round_num.
    if use_test_cd:
      self.assertIsNotNone(test_fn)
      eval_fns_to_test = (part_train_eval_fn, part_val_fn, unpart_fn, test_fn)
    else:
      self.assertIsNone(test_fn)
      eval_fns_to_test = (part_train_eval_fn, part_val_fn, unpart_fn)

    for eval_fn in eval_fns_to_test:
      metrics_dict = eval_fn(server_state, 0)
      self.assertEqual(list(metrics_dict.keys()), expected_keys)

  @parameterized.named_parameters(('case1', 3, 4), ('case2', 3, 5),
                                  ('case3', 2, 3))
  def test_create_federated_eval_fns_skips_rounds(self, rounds_per_eval,
                                                  round_num):
    """Test that create_federated_eval_fns skips the appropriate rounds."""

    part_train_eval_fn, part_val_fn, unpart_fn, _ = trainer_utils.create_federated_eval_fns(
        tff_model_builder=tff_model_builder,
        metrics_builder=metrics_builder,
        part_train_eval_cd=create_federated_cd(),
        part_val_cd=create_federated_cd(),
        unpart_cd=create_federated_cd(),
        test_cd=create_federated_cd(),
        stat_fns=eval_metric_distribution.ALL_STAT_FNS,
        rounds_per_eval=rounds_per_eval,
        part_clients_per_eval=2,
        unpart_clients_per_eval=2,
        test_clients_for_eval=3,
        resample_eval_clients=False,
        eval_clients_random_seed=1)
    keras_model = keras_model_builder_with_zeros()
    model_weights = tff.learning.ModelWeights.from_model(keras_model)
    server_state = tff.learning.framework.ServerState(model_weights, [], [], [])

    # Federated validation fn requires a positional arg round_num.
    for eval_fn in (part_train_eval_fn, part_val_fn, unpart_fn):
      metrics_dict = eval_fn(server_state, round_num)
      self.assertEmpty(metrics_dict.keys())

  @parameterized.named_parameters(('with_test_cd', True),
                                  ('without_test_cd', False))
  def test_create_centralized_eval_fns(self, use_test_cd):
    """Test for create_centralized_eval_fns."""

    (part_train_eval_fn, part_val_fn, unpart_fn,
     test_fn) = trainer_utils.create_centralized_eval_fns(
         tff_model_builder=tff_model_builder,
         metrics_builder=metrics_builder,
         part_train_eval_cd=create_federated_cd(),
         part_val_cd=create_federated_cd(),
         unpart_cd=create_federated_cd(),
         test_cd=create_federated_cd() if use_test_cd else None,
         stat_fns=eval_metric_distribution.ALL_STAT_FNS,
         part_clients_per_eval=2,
         unpart_clients_per_eval=2,
         test_clients_for_eval=3,
         resample_eval_clients=False,
         eval_clients_random_seed=1)
    keras_model = keras_model_builder_with_zeros()

    expected_keys = [
        f'mean_squared_error/{s}' for s in eval_metric_distribution.ALL_STAT_FNS
    ]

    if use_test_cd:
      self.assertIsNotNone(test_fn)
      eval_fns_to_test = (part_train_eval_fn, part_val_fn, unpart_fn, test_fn)
    else:
      self.assertIsNone(test_fn)
      eval_fns_to_test = (part_train_eval_fn, part_val_fn, unpart_fn)

    for eval_fn in eval_fns_to_test:
      metrics_dict = eval_fn(keras_model)
      self.assertEqual(list(metrics_dict.keys()), expected_keys)


if __name__ == '__main__':
  tf.test.main()
