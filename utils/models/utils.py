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
"""Shared utilities for all the model experiments."""

import random
from typing import Type

import tensorflow as tf


class DeterministicInitializer():
  """Wrapper to produce different deterministic initialization values."""

  def __init__(self, initializer_type: Type[tf.keras.initializers.Initializer],
               base_seed: int):
    self._initializer_type = initializer_type
    if base_seed is None:
      base_seed = random.randint(1, 1e9)
    self._base_seed = base_seed

  def __call__(self):
    self._base_seed += 1
    return self._initializer_type(seed=self._base_seed)
