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
"""Test server optimizers."""

import collections
from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from dp_ftrl import optimizer_utils

ModelVariables = collections.namedtuple('ModelVariables', 'weights bias')


def _create_model_variables():
  return ModelVariables(
      weights=tf.Variable(
          lambda: tf.zeros(dtype=tf.float32, shape=(784, 10)),
          name='weights',
          trainable=True),
      bias=tf.Variable(
          lambda: tf.zeros(dtype=tf.float32, shape=(10)),
          name='bias',
          trainable=True))


class OptimizerTest(tf.test.TestCase, parameterized.TestCase):

  def test_deterministic_sgd(self):
    model_variables = _create_model_variables()
    grad = tf.nest.map_structure(tf.ones_like, model_variables)
    optimizer = optimizer_utils.SGDServerOptimizer(learning_rate=0.1)

    state = optimizer.init_state()
    for i in range(2):
      state = optimizer.model_update(state, model_variables, grad, i)

    self.assertLen(model_variables, 2)
    # variables initialize with all zeros and update with all ones and learning
    # rate 0.1 for several steps.
    flatten_variables = tf.nest.flatten(model_variables)
    self.assertAllClose(flatten_variables,
                        [-0.2 * np.ones_like(v) for v in flatten_variables])

  @parameterized.named_parameters(
      ('ftrl_m0s2', optimizer_utils.DPFTRLMServerOptimizer, 0, 2, 0.2),
      ('ftrl_m0.9s2', optimizer_utils.DPFTRLMServerOptimizer, 0.9, 2, 0.29),
      ('ftrl_m0s3', optimizer_utils.DPFTRLMServerOptimizer, 0, 3, 0.3),
      ('ftrl_m0.9s3', optimizer_utils.DPFTRLMServerOptimizer, 0.9, 3, 0.561),
      ('sgd_m0s2', optimizer_utils.DPSGDMServerOptimizer, 0, 2, 0.2),
      ('sgd_m0.9s2', optimizer_utils.DPSGDMServerOptimizer, 0.9, 2, 0.29),
      ('sgd_m0s3', optimizer_utils.DPSGDMServerOptimizer, 0, 3, 0.3),
      ('sgd_m0.9s3', optimizer_utils.DPSGDMServerOptimizer, 0.9, 3, 0.561))
  def test_deterministic(self, optimizer_fn, momentum, steps, result):
    model_variables = _create_model_variables()
    model_weight_specs = tf.nest.map_structure(
        lambda v: tf.TensorSpec(v.shape, v.dtype), model_variables)
    grad = tf.nest.map_structure(tf.ones_like, model_variables)
    optimizer = optimizer_fn(
        learning_rate=0.1,
        momentum=momentum,
        noise_std=0.0,
        model_weight_specs=model_weight_specs)

    state = optimizer.init_state()
    for i in range(steps):
      state = optimizer.model_update(state, model_variables, grad, i)

    self.assertLen(model_variables, 2)
    # variables initialize with all zeros and update with all ones and learning
    # rate 0.1 for several steps.
    flatten_variables = tf.nest.flatten(model_variables)
    self.assertAllClose(flatten_variables,
                        [-result * np.ones_like(v) for v in flatten_variables])

  @parameterized.named_parameters(
      ('m0s2', 0, 2, False),
      ('m0.9s2', 0.9, 2, False),
      ('m0s3', 0.9, 10, False),
      ('m0s3nes', 0.9, 10, True),
  )
  def test_ftrl_match_keras(self, momentum, steps, nesterov):
    # FTRL is identical to SGD for unconstrained problem when no noise is added;
    # it is identical to Keras SGD without learning rate change.
    lr = 0.1

    def _run_ftrl():
      model_variables = _create_model_variables()
      model_weight_specs = tf.nest.map_structure(
          lambda v: tf.TensorSpec(v.shape, v.dtype), model_variables)
      grad = tf.nest.map_structure(tf.ones_like, model_variables)
      optimizer = optimizer_utils.DPFTRLMServerOptimizer(
          learning_rate=lr,
          momentum=momentum,
          noise_std=0.0,
          model_weight_specs=model_weight_specs,
          use_nesterov=nesterov)

      state = optimizer.init_state()
      for i in range(steps):
        state = optimizer.model_update(state, model_variables, grad, i)

      self.assertLen(model_variables, 2)
      return tf.nest.flatten(model_variables)

    def _run_keras():
      model_variables = tf.nest.flatten(_create_model_variables())
      grad = tf.nest.map_structure(tf.ones_like, model_variables)
      optimizer = tf.keras.optimizers.SGD(
          learning_rate=lr, momentum=momentum, nesterov=nesterov)
      for _ in range(steps):
        optimizer.apply_gradients(zip(grad, model_variables))
      return model_variables

    self.assertAllClose(_run_ftrl(), _run_keras())


if __name__ == '__main__':
  tf.test.main()
