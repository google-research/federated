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
"""Utilities for creating warmup learning rate schedules."""

import tensorflow as tf


class WarmupSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  """A warmup learning rate schedule for `tf.keras.optimizers` classes."""

  def __init__(self, max_learning_rate, warmup_steps):
    if max_learning_rate < 0:
      raise ValueError('The max_learning_rate must be positive.')

    self.max_learning_rate = max_learning_rate
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    multiplier = tf.math.minimum(1.0, (step + 1) / self.warmup_steps)
    return multiplier * self.max_learning_rate
