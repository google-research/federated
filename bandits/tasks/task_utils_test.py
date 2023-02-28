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
"""Tests for task_utils."""
import collections
import math
from typing import Any, Optional

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_federated as tff

from bandits import bandits_utils
from bandits.tasks import task_utils

_TOTAL_ARMS = 3


def _get_synthetic_batch(
    batch_size: int = 4,
    deterministic_action: Optional[int] = None,
    correct_prediction: bool = False,
    multi_label: bool = False,
    weight_scale: Optional[float] = None,
) -> tuple[dict[str, Any], tf.Tensor]:
  # This synthetic batch uses random rewards, probability, and predictions.
  if deterministic_action is None:
    action = tf.random.uniform(
        shape=[batch_size], minval=0, maxval=_TOTAL_ARMS, dtype=tf.int32
    )
  else:
    assert deterministic_action < _TOTAL_ARMS
    action = tf.constant(deterministic_action, shape=[batch_size])
  reward = tf.cast(
      tf.random.uniform(shape=[batch_size], minval=0, maxval=2, dtype=tf.int32),
      dtype=tf.float32,
  )
  prob = tf.constant(1.0 / _TOTAL_ARMS, shape=[batch_size])
  y_pred = tf.random.uniform(
      shape=[batch_size, _TOTAL_ARMS], minval=0, maxval=1, dtype=tf.float32
  )
  if correct_prediction:
    label = tf.argmax(y_pred, axis=1)
    if multi_label:
      label = tf.one_hot(label, depth=_TOTAL_ARMS, on_value=1, off_value=0)
  else:
    if multi_label:
      label = tf.random.uniform(
          shape=[batch_size, _TOTAL_ARMS], minval=0, maxval=2, dtype=tf.int32
      )
    else:
      label = tf.random.uniform(
          shape=[batch_size], minval=0, maxval=_TOTAL_ARMS, dtype=tf.int32
      )
  if weight_scale is None:
    weight_scale = 1.0 / _TOTAL_ARMS
  y_true = collections.OrderedDict(
      label=label,
      action=action,
      reward=reward,
      prob=prob,
      weight_scale=weight_scale,
  )
  return y_true, y_pred


class TaskUtilsTest(tf.test.TestCase, parameterized.TestCase):

  def test_vector_normalize(self):
    norm_layer = task_utils.VecNormLayer()
    batch_size, dim = 5, 32
    vec = tf.random.stateless_uniform(shape=[batch_size, dim], seed=[1, 2])
    normalize_embeddings = norm_layer(vec)
    norms = tf.norm(normalize_embeddings, ord='euclidean', axis=1)
    self.assertAllClose(norms, tf.ones([batch_size]))

  @parameterized.named_parameters(('weight', True), ('unweight', False))
  def test_mse_loss_run(self, weighted):
    bmse = task_utils.BanditsMSELoss(importance_weighting=weighted)
    y_true, y_pred = _get_synthetic_batch()
    # The rewards are 0/1, and the random predictions are [0, 1),
    # probability and weight_scale 1/_TOTAL_ARMS.
    self.assertLessEqual(bmse(y_true, y_pred), 1)

  @parameterized.named_parameters(('weight', True), ('unweight', False))
  def test_bce_loss_run(self, weighted):
    bce = task_utils.BanditsCELoss(importance_weighting=weighted)
    y_true, y_pred = _get_synthetic_batch()
    # The rewards are 0/1, and the random predictions are [0, 1),
    # probability and weight_scale 1/_TOTAL_ARMS.
    self.assertGreaterEqual(bce(y_true, y_pred), math.log(1.0 + math.exp(-1)))
    self.assertLessEqual(bce(y_true, y_pred), math.log(2))

  @parameterized.named_parameters(('action0', 0), ('action2', 2))
  def test_mse_loss_correct(self, deterministic_action):
    bmse = task_utils.BanditsMSELoss()
    y_true, y_pred = _get_synthetic_batch(
        batch_size=16, deterministic_action=deterministic_action
    )
    logits = tf.squeeze(y_pred[:, deterministic_action])
    mse = tf.reduce_mean(
        (y_true[bandits_utils.BanditsKeys.reward] - logits) ** 2
        * y_true['weight_scale']
        / y_true['prob']
    )
    self.assertNear(bmse(y_true, y_pred), mse, err=1e-7)

  @parameterized.named_parameters(('action0', 0), ('action2', 2))
  def test_mse_loss_unweight_correct(self, deterministic_action):
    bmse = task_utils.BanditsMSELoss(importance_weighting=False)
    y_true, y_pred = _get_synthetic_batch(
        batch_size=16, deterministic_action=deterministic_action
    )
    logits = tf.squeeze(y_pred[:, deterministic_action])
    mse = tf.reduce_mean(
        (y_true[bandits_utils.BanditsKeys.reward] - logits) ** 2
    )
    self.assertNear(bmse(y_true, y_pred), mse, err=1e-7)

  @parameterized.named_parameters(('action0', 0), ('action2', 2))
  def test_bce_loss_correct(self, deterministic_action):
    bce = task_utils.BanditsCELoss()
    y_true, y_pred = _get_synthetic_batch(
        batch_size=16, deterministic_action=deterministic_action
    )
    logits = tf.squeeze(y_pred[:, deterministic_action])
    rewards = y_true[bandits_utils.BanditsKeys.reward]
    ce = tf.reduce_mean(
        (
            rewards * tf.math.log(1.0 + tf.math.exp(-logits))
            + (1 - rewards) * tf.math.log(1.0 + tf.math.exp(logits))
        )
        * y_true['weight_scale']
        / y_true['prob']
    )
    self.assertNear(bce(y_true, y_pred), ce, err=1e-5)

  @parameterized.named_parameters(('action0', 0), ('action2', 2))
  def test_bce_loss_unweight_correct(self, deterministic_action):
    bce = task_utils.BanditsCELoss(importance_weighting=False)
    y_true, y_pred = _get_synthetic_batch(
        batch_size=16, deterministic_action=deterministic_action
    )
    logits = tf.squeeze(y_pred[:, deterministic_action])
    rewards = y_true[bandits_utils.BanditsKeys.reward]
    ce = tf.reduce_mean(
        (
            rewards * tf.math.log(1.0 + tf.math.exp(-logits))
            + (1 - rewards) * tf.math.log(1.0 + tf.math.exp(logits))
        )
    )
    self.assertNear(bce(y_true, y_pred), ce, err=1e-5)

  def test_supervised_mse_loss_run(self):
    smse = task_utils.SupervisedMSELoss(
        num_arms=_TOTAL_ARMS, reward_right=1.0, reward_wrong=0.0
    )
    y_true, y_pred = _get_synthetic_batch()
    # The rewards are 0/1, and the random predictions are [0, 1)
    self.assertLessEqual(smse(y_true, y_pred), 1.0)

  def test_supervised_mse_loss_correct(self):
    batch_size = 16
    reward_right, reward_wrong = 1.0, 0.0
    smse = task_utils.SupervisedMSELoss(
        num_arms=_TOTAL_ARMS,
        reward_right=reward_right,
        reward_wrong=reward_wrong,
    )
    y_true, y_pred = _get_synthetic_batch(batch_size=batch_size)
    label = y_true[bandits_utils.BanditsKeys.label]
    right_logits = tf.gather(y_pred, label, axis=1, batch_dims=1)
    sse = (
        tf.math.reduce_sum((y_pred - reward_wrong) ** 2)
        + tf.math.reduce_sum((right_logits - reward_right) ** 2)
        - tf.math.reduce_sum((right_logits - reward_wrong) ** 2)
    )
    self.assertNear(
        smse(y_true, y_pred), sse / float(batch_size * _TOTAL_ARMS), err=1e-7
    )

  def test_supervised_ce_loss_run(self):
    sce = task_utils.SupervisedCELoss()
    y_true, y_pred = _get_synthetic_batch()
    self.assertGreater(sce(y_true, y_pred), 0.0)

  def test_supervised_ce_loss_correct(self):
    sce = task_utils.SupervisedCELoss()
    y_true, y_pred = _get_synthetic_batch(batch_size=16)
    label = y_true[bandits_utils.BanditsKeys.label]
    exp_logits = tf.math.exp(y_pred)
    pred_exp = tf.gather(exp_logits, label, axis=1, batch_dims=1)
    pred_prob = tf.squeeze(pred_exp / tf.reduce_sum(exp_logits, axis=1))
    ce = tf.reduce_mean(-tf.math.log(pred_prob))
    self.assertNear(sce(y_true, y_pred), ce, err=1e-5)

  def test_accuracy_run(self):
    acc = task_utils.WrapCategoricalAccuracy()
    y_true, y_pred = _get_synthetic_batch()
    self.assertLessEqual(acc(y_true, y_pred), 1.0)

  def test_accuracy_correct(self):
    batch_size = 32
    acc = task_utils.WrapCategoricalAccuracy()
    y_true, y_pred = _get_synthetic_batch(
        batch_size=batch_size, correct_prediction=True
    )
    self.assertNear(acc(y_true, y_pred), 1.0, err=1.0 / batch_size)
    # Flip the prediction so that we will have a second batch with wrong
    # prediction
    self.assertNear(acc(y_true, 1.0 - y_pred), 0.5, err=1.0 / batch_size)

  @parameterized.named_parameters(
      ('default', None), ('scale1', 1.0), ('scale3.3', 3.3)
  )
  def test_weight_accuracy_correct(self, weight_scale):
    batch_size = 32
    acc = task_utils.WeightCategoricalAccuracy()
    y_true, y_pred = _get_synthetic_batch(
        batch_size=batch_size,
        correct_prediction=True,
        weight_scale=weight_scale,
    )
    # The sample_weight in metrics also appear in the denominator, so a single
    # scalar change of weight won't affect accuracy.
    self.assertNear(acc(y_true, y_pred), 1.0, err=1.0 / batch_size)
    # Flip the prediction so that we will have a second batch with wrong
    # prediction
    self.assertNear(acc(y_true, 1.0 - y_pred), 0.5, err=1.0 / batch_size)
    y_true, y_pred = _get_synthetic_batch(
        batch_size=batch_size, correct_prediction=True, weight_scale=0
    )
    # A batch of samples with zero sample_weight won't affect the accuracy.
    self.assertNear(acc(y_true, y_pred), 0.5, err=1.0 / batch_size)

  def test_multilabel_mse_loss_run(self):
    smse = task_utils.MultiLabelMSELoss(reward_right=1.0, reward_wrong=0.0)
    y_true, y_pred = _get_synthetic_batch(multi_label=True)
    # The rewards are 0/1, and the random predictions are [0, 1)
    self.assertLessEqual(smse(y_true, y_pred), 1.0)

  def test_multilabel_mse_loss_correct(self):
    batch_size = 16
    reward_right, reward_wrong = 1.0, 0.0
    smse = task_utils.MultiLabelMSELoss(
        reward_right=reward_right, reward_wrong=reward_wrong
    )
    y_true, y_pred = _get_synthetic_batch(
        batch_size=batch_size, multi_label=True
    )
    label = tf.cast(y_true[bandits_utils.BanditsKeys.label], tf.float32)
    sse = tf.math.reduce_sum(
        (y_pred - label * reward_right - (1 - label) * reward_wrong) ** 2
    )
    self.assertNear(
        smse(y_true, y_pred), sse / float(batch_size * _TOTAL_ARMS), err=1e-7
    )

  def test_multilabel_ce_loss_run(self):
    sce = task_utils.MultiLabelCELoss()
    y_true, y_pred = _get_synthetic_batch(multi_label=True)
    self.assertGreater(sce(y_true, y_pred), 0.0)

  def test_multilabel_ce_loss_correct(self):
    sce = task_utils.MultiLabelCELoss()
    y_true, y_pred = _get_synthetic_batch(batch_size=16, multi_label=True)
    label = tf.cast(y_true[bandits_utils.BanditsKeys.label], tf.float32)
    exp_logits = tf.math.exp(y_pred)
    prob = exp_logits / (1.0 + exp_logits)
    pred_prob = prob * label + (1.0 - prob) * (1.0 - label)
    ce = tf.reduce_mean(-tf.math.log(pred_prob))
    self.assertNear(sce(y_true, y_pred), ce, err=1e-5)

  def test_client_selection(self):
    num_clients = 30
    sorted_client_ids = sorted([str(i) for i in range(num_clients)])
    cd = tff.simulation.datasets.TestClientData(
        {str(i): [i] for i in range(num_clients)}
    )
    scd = task_utils.clientdata_for_select_clients(cd, '0-10')
    self.assertCountEqual(scd.client_ids, sorted_client_ids[:10])
    scd = task_utils.clientdata_for_select_clients(cd, '2-17')
    self.assertCountEqual(scd.client_ids, sorted_client_ids[2:17])
    scd = task_utils.clientdata_for_select_clients(cd, '25-30')
    self.assertCountEqual(scd.client_ids, sorted_client_ids[25:30])

  def test_client_selection_raise(self):
    num_clients = 30
    cd = tff.simulation.datasets.TestClientData(
        {str(i): [i] for i in range(num_clients)}
    )
    with self.assertRaises(AssertionError):
      task_utils.clientdata_for_select_clients(cd, '-0-10')
      task_utils.clientdata_for_select_clients(cd, '10-8')
      task_utils.clientdata_for_select_clients(cd, '3-33')


if __name__ == '__main__':
  tf.test.main()
