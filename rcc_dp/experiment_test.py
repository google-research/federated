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
"""Tests for experiment definitions."""

from absl.testing import absltest
from rcc_dp import config as defaults
from rcc_dp import experiment


class ExperimentTest(absltest.TestCase):

  def test_evaluate_does_not_fail(self):
    work_path = self.create_tempdir().full_path
    config = defaults.get_config()
    experiment.evaluate(work_path, config)


if __name__ == "__main__":
  absltest.main()
