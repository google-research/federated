# Copyright 2018, Google LLC.
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

import collections
import contextlib

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf
import tensorflow_federated as tff

from data_poor_fl import optimizer_flag_utils

FLAGS = flags.FLAGS
TEST_CLIENT_FLAG_PREFIX = 'test_client'
TEST_SERVER_FLAG_PREFIX = 'test_server'


@contextlib.contextmanager
def flag_sandbox(flag_value_dict):

  def _set_flags(flag_dict):
    for name, value in flag_dict.items():
      FLAGS[name].value = value

  # Store the current values and override with the new.
  preserved_value_dict = {
      name: FLAGS[name].value for name in flag_value_dict.keys()
  }
  _set_flags(flag_value_dict)
  yield

  # Restore the saved values.
  for name in preserved_value_dict.keys():
    FLAGS[name].unparse()
  _set_flags(preserved_value_dict)


def setUpModule():
  # Create flags here to ensure duplicate flags are not created.
  optimizer_flag_utils.define_optimizer_flags(TEST_SERVER_FLAG_PREFIX)
  optimizer_flag_utils.define_optimizer_flags(TEST_CLIENT_FLAG_PREFIX)

# Create a list of `(test name, optimizer name flag value, optimizer class)`
# for parameterized tests.
_OPTIMIZERS_TO_TEST = [
    (name, name, builder)
    for name, builder in optimizer_flag_utils._OPTIMIZER_BUILDERS.items()
]


class CreateOptimizerFromFlagsTest(parameterized.TestCase):

  def test_create_optimizer_from_flags_invalid_optimizer(self):
    FLAGS['{}_optimizer'.format(TEST_CLIENT_FLAG_PREFIX)].value = 'foo'
    with self.assertRaisesRegex(ValueError, 'not a valid optimizer'):
      optimizer_flag_utils.create_optimizer_from_flags(TEST_CLIENT_FLAG_PREFIX)

  def test_create_optimizer_fn_with_no_learning_rate(self):
    with flag_sandbox({
        '{}_optimizer'.format(TEST_CLIENT_FLAG_PREFIX): 'sgd',
        '{}_learning_rate'.format(TEST_CLIENT_FLAG_PREFIX): None
    }):
      with self.assertRaisesRegex(ValueError, 'Learning rate'):
        optimizer_flag_utils.create_optimizer_from_flags(
            TEST_CLIENT_FLAG_PREFIX)

  def test_create_optimizer_from_flags_flags_set_not_for_optimizer(self):
    with flag_sandbox({'{}_optimizer'.format(TEST_CLIENT_FLAG_PREFIX): 'sgd'}):
      # Set an Adam flag that isn't used in SGD.
      # We need to use `_parse_args` because that is the only way FLAGS is
      # notified that a non-default value is being used.
      bad_adam_flag = '{}_adam_beta_1'.format(TEST_CLIENT_FLAG_PREFIX)
      FLAGS._parse_args(
          args=['--{}=0.5'.format(bad_adam_flag)], known_only=True)
      with self.assertRaisesRegex(
          ValueError,
          r'Commandline flags for .*\[sgd\].*\'test_client_adam_beta_1\'.*'):
        optimizer_flag_utils.create_optimizer_from_flags(
            TEST_CLIENT_FLAG_PREFIX)
      FLAGS[bad_adam_flag].unparse()

  @parameterized.named_parameters(_OPTIMIZERS_TO_TEST)
  def test_create_client_optimizer_from_flags(self, optimizer_name,
                                              optimizer_builder):
    commandline_set_learning_rate = 100.0
    with flag_sandbox({
        '{}_optimizer'.format(TEST_CLIENT_FLAG_PREFIX):
            optimizer_name,
        '{}_learning_rate'.format(TEST_CLIENT_FLAG_PREFIX):
            commandline_set_learning_rate
    }):

      custom_optimizer = optimizer_flag_utils.create_optimizer_from_flags(
          TEST_CLIENT_FLAG_PREFIX)
      expected_optimizer = optimizer_builder(learning_rate=0.01)
      optimizer_state = custom_optimizer.initialize(
          tf.TensorSpec(shape=(1,), dtype=tf.float32))
      self.assertIsInstance(custom_optimizer, tff.learning.optimizers.Optimizer)
      self.assertIsInstance(custom_optimizer, type(expected_optimizer))
      self.assertEqual(optimizer_state['learning_rate'],
                       commandline_set_learning_rate)

  @parameterized.named_parameters(_OPTIMIZERS_TO_TEST)
  def test_create_server_optimizer_from_flags(self, optimizer_name,
                                              optimizer_builder):
    commandline_set_learning_rate = 100.0
    with flag_sandbox({
        '{}_optimizer'.format(TEST_SERVER_FLAG_PREFIX):
            optimizer_name,
        '{}_learning_rate'.format(TEST_SERVER_FLAG_PREFIX):
            commandline_set_learning_rate
    }):
      custom_optimizer = optimizer_flag_utils.create_optimizer_from_flags(
          TEST_SERVER_FLAG_PREFIX)
      expected_optimizer = optimizer_builder(learning_rate=0.01)
      optimizer_state = custom_optimizer.initialize(
          tf.TensorSpec(shape=(1,), dtype=tf.float32))
      self.assertIsInstance(custom_optimizer, tff.learning.optimizers.Optimizer)
      self.assertIsInstance(custom_optimizer, type(expected_optimizer))
      self.assertEqual(optimizer_state['learning_rate'],
                       commandline_set_learning_rate)


class RemoveUnusedFlagsTest(absltest.TestCase):

  def test_remove_unused_flags_without_optimizer_flag(self):
    hparam_dict = collections.OrderedDict([('client_opt_fn', 'sgd'),
                                           ('client_sgd_momentum', 0.3)])
    with self.assertRaisesRegex(ValueError,
                                'The flag client_optimizer was not defined.'):
      _ = optimizer_flag_utils.remove_unused_flags('client', hparam_dict)

  def test_remove_unused_flags_with_empty_optimizer(self):
    hparam_dict = collections.OrderedDict([('optimizer', '')])

    with self.assertRaisesRegex(
        ValueError, 'The flag optimizer was not set. '
        'Unable to determine the relevant optimizer.'):
      _ = optimizer_flag_utils.remove_unused_flags(
          prefix=None, hparam_dict=hparam_dict)

  def test_remove_unused_flags_with_prefix(self):
    hparam_dict = collections.OrderedDict([('client_optimizer', 'sgd'),
                                           ('non_client_value', 0.1),
                                           ('client_sgd_momentum', 0.3),
                                           ('client_adam_momentum', 0.5)])

    relevant_hparam_dict = optimizer_flag_utils.remove_unused_flags(
        'client', hparam_dict)
    expected_flag_names = [
        'client_optimizer', 'non_client_value', 'client_sgd_momentum'
    ]
    self.assertCountEqual(relevant_hparam_dict.keys(), expected_flag_names)
    self.assertEqual(relevant_hparam_dict['client_optimizer'], 'sgd')
    self.assertEqual(relevant_hparam_dict['non_client_value'], 0.1)
    self.assertEqual(relevant_hparam_dict['client_sgd_momentum'], 0.3)

  def test_remove_unused_flags_without_prefix(self):
    hparam_dict = collections.OrderedDict([('optimizer', 'sgd'), ('value', 0.1),
                                           ('sgd_momentum', 0.3),
                                           ('adam_momentum', 0.5)])
    relevant_hparam_dict = optimizer_flag_utils.remove_unused_flags(
        prefix=None, hparam_dict=hparam_dict)
    expected_flag_names = ['optimizer', 'value', 'sgd_momentum']
    self.assertCountEqual(relevant_hparam_dict.keys(), expected_flag_names)
    self.assertEqual(relevant_hparam_dict['optimizer'], 'sgd')
    self.assertEqual(relevant_hparam_dict['value'], 0.1)
    self.assertEqual(relevant_hparam_dict['sgd_momentum'], 0.3)

  def test_removal_with_standard_default_values(self):
    hparam_dict = collections.OrderedDict([('client_optimizer', 'adam'),
                                           ('non_client_value', 0),
                                           ('client_sgd_momentum', 0),
                                           ('client_adam_param1', None),
                                           ('client_adam_param2', False)])

    relevant_hparam_dict = optimizer_flag_utils.remove_unused_flags(
        'client', hparam_dict)
    expected_flag_names = [
        'client_optimizer', 'non_client_value', 'client_adam_param1',
        'client_adam_param2'
    ]
    self.assertCountEqual(relevant_hparam_dict.keys(), expected_flag_names)
    self.assertEqual(relevant_hparam_dict['client_optimizer'], 'adam')
    self.assertEqual(relevant_hparam_dict['non_client_value'], 0)
    self.assertIsNone(relevant_hparam_dict['client_adam_param1'])
    self.assertEqual(relevant_hparam_dict['client_adam_param2'], False)


if __name__ == '__main__':
  absltest.main()
