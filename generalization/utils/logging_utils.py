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
"""Utility functions and classes for logging progress."""

from typing import Optional

from absl import logging


class ProgressLogger:
  """Utility class for logging progress."""

  def __init__(self,
               name: str,
               every: int = 10000,
               total: Optional[int] = None):
    self._name = name
    self._cnt = 0
    self._every = every
    self._current_shard = 0
    self._total = total

    logging.info('Starting %s.', self._name)

  def increment(self, addl_cnt: int = 1):
    """Add the number of new elements processed, export logging if desired."""
    self._cnt += addl_cnt

    new_shard = self._cnt // self._every

    if new_shard > self._current_shard:
      self._current_shard = new_shard

      if self._total is None:
        logging.info('  %s, %d processed.', self._name, self._cnt)
      else:
        logging.info('  %s, %d out of %d processed.', self._name, self._cnt,
                     self._total)

  def __del__(self):
    logging.info('Finished %s.', self._name)
