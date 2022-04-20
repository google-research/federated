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

import tempfile

from absl.testing import absltest
from absl.testing import flagsaver

from data_poor_fl import global_trainer


class GlobalTrainerTest(absltest.TestCase):

  @flagsaver.flagsaver(
      root_output_dir=tempfile.mkdtemp(),
      experiment_name='test_experiment',
      clients_per_train_round=1,
      total_rounds=2,
      client_optimizer='sgd',
      client_learning_rate=0.01,
      server_optimizer='sgd',
      server_learning_rate=1.0,
      use_synthetic_data=True)
  def test_executes(self):
    global_trainer.main([])


if __name__ == '__main__':
  absltest.main()
