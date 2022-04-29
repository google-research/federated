# Copyright 2019, Google LLC.
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
"""Utilities for creating and removing optimizer-specific flags."""

import collections
import contextlib
import inspect
from typing import Any, Callable, Dict, Iterable, List, Optional

from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_addons.optimizers as tfao

from utils.optimizers import lars
from utils.optimizers import shampoo
from utils.optimizers import yogi


_all_hparam_flags = []


@contextlib.contextmanager
def record_hparam_flags():
  """A context manager that yields all flags created in its scope.

  This is useful for defining hyperparameter flags of an experiment, especially
  when the flags are partitioned across a number of modules. The total list of
  flags defined across modules can then be accessed via get_hparam_flags().

  Example usage:
  ```python
  with record_hparam_flags() as optimizer_hparam_flags:
      flags.DEFINE_string('optimizer', 'sgd', 'Optimizer for training.')
  with record_hparam_flags() as evaluation_hparam_flags:
      flags.DEFINE_string('eval_metric', 'accuracy', 'Metric for evaluation.')
  experiment_hparam_flags = get_hparam_flags().
  ```

  Check `research/optimization/emnist/run_emnist.py` for more usage details.

  Yields:
    A list of all newly created flags.
  """
  old_flags = set(iter(flags.FLAGS))
  new_flags = []
  yield new_flags
  new_flags.extend([f for f in flags.FLAGS if f not in old_flags])
  _all_hparam_flags.extend(new_flags)


def lookup_flag_values(flag_list: Iterable[str]) -> collections.OrderedDict:
  """Returns a dictionary of (flag_name, flag_value) for a list of flags."""
  flag_odict = collections.OrderedDict()
  for flag_name in flag_list:
    if not isinstance(flag_name, str):
      raise ValueError(
          'All flag names must be strings. Flag {} was of type {}.'.format(
              flag_name, type(flag_name)))

    if flag_name not in flags.FLAGS:
      raise ValueError('"{}" is not a defined flag.'.format(flag_name))
    flag_odict[flag_name] = flags.FLAGS[flag_name].value

  return flag_odict


def _optimizer_canonical_name(optimizer_cls):
  """Return a short, canonical name for an optimizer for us in flags."""
  return optimizer_cls.__name__.lower()


# List of optimizers currently supported.
_SUPPORTED_OPTIMIZERS_CLS = [
    tf.keras.optimizers.SGD,
    tf.keras.optimizers.Adagrad,
    tf.keras.optimizers.Adam,
    yogi.Yogi,
    lars.LARS,
    tfao.lamb.LAMB,
    shampoo.Shampoo,
]

_SUPPORTED_OPTIMIZERS = {
    _optimizer_canonical_name(cls): cls for cls in _SUPPORTED_OPTIMIZERS_CLS
}


def define_optimizer_flags(prefix: str) -> None:
  """Defines flags with `prefix` to configure an optimizer.

  This method is inteded to be paired with `create_optimizer_from_flags` using
  the same `prefix`, to allow Python binaries to constructed TensorFlow
  optimizers parameterized by commandline flags.

  This creates two new flags:
    * `--<prefix>_optimizer=<optimizer name>`
    * `--<prefix>_learning_rate`

  In addition to a suite of flags for each optimizer:
    * `--<prefix>_<optimizer name>_<constructor_argument>`

  For example, given the prefix "client" this will create flags (non-exhaustive
  list):

    *  `--client_optimizer`
    *  `--client_learning_rate`
    *  `--client_sgd_momentum`
    *  `--client_sgd_nesterov`
    *  `--client_adam_beta_1`
    *  `--client_adam_beta_2`
    *  `--client_adam_epsilon`

  Then calls to `create_optimizer_from_flags('client')` will construct an
  optimizer of the type named in `--client_optimizer`, parameterized by the
  flags prefixed with the matching optimizer name. For example,  if
  `--client_optimizer=sgd`, `--client_sgd_*` flags will be used.

  IMPORTANT: For flags to be correctly parsed from the commandline, this method
  must be called before `absl.app.run(main)`, and is recommened to be called
  next to other flag definitions at the top of a py_binary.

  Args:
    prefix: A string (possibly empty) indicating which optimizer is being
      configured.
  """
  # Create top-level, non-optimizer specific flags for picking the optimizer
  # type and the learning rate.
  flags.DEFINE_enum(
      name='{!s}_optimizer'.format(prefix),
      default=None,
      enum_values=list(_SUPPORTED_OPTIMIZERS.keys()),
      help='The type of optimizer to construct for `{!s}`'.format(prefix))
  logging.info('Defined new flag: [%s]', '{!s}_optimizer'.format(prefix))
  flags.DEFINE_float(
      name='{!s}_learning_rate'.format(prefix),
      default=None,
      help='Base learning rate for optimizer `{!s}`'.format(prefix))
  logging.info('Defined new flag: [%s]', '{!s}_learning_rate'.format(prefix))

  for optimizer_name, optimizer_cls in _SUPPORTED_OPTIMIZERS.items():
    # Pull out the constructor parameters except for `self`.
    constructor_signature = inspect.signature(optimizer_cls.__init__)
    constructor_params = list(constructor_signature.parameters.values())[1:]

    def prefixed(basename, optimizer_name=optimizer_name):
      if prefix:
        return '{!s}_{!s}_{!s}'.format(prefix, optimizer_name, basename)
      else:
        return '{!s}_{!s}'.format(optimizer_name, basename)

    def is_param_of_type(param, typ):
      return (param.default is None and param.annotation == Optional[typ] or
              isinstance(param.default, typ))

    for param in constructor_params:
      if param.name in ['kwargs', 'args', 'learning_rate']:
        continue

      if is_param_of_type(param, bool):
        define_flag_fn = flags.DEFINE_bool
      elif is_param_of_type(param, float):
        define_flag_fn = flags.DEFINE_float
      elif is_param_of_type(param, int):
        define_flag_fn = flags.DEFINE_integer
      elif is_param_of_type(param, str):
        define_flag_fn = flags.DEFINE_string
      elif is_param_of_type(param, List[str]):
        define_flag_fn = flags.DEFINE_multi_string
      else:
        raise NotImplementedError('Cannot define flag [{!s}] '
                                  'for parameter [{!s}] of type [{!s}] '
                                  '(default value type [{!s}]) '
                                  'on optimizer [{!s}]'.format(
                                      prefixed(param.name),
                                      param.name, param.annotation,
                                      type(param.default), optimizer_name))
      define_flag_fn(
          name=prefixed(param.name),
          default=param.default,
          help='{!s} argument for the {!s} optimizer.'.format(
              param.name, optimizer_name))
      logging.info('Defined new flag: [%s]', prefixed(param.name))


def create_optimizer_fn_from_flags(
    prefix: str) -> Callable[..., tf.keras.optimizers.Optimizer]:
  """Returns an optimizer function based on prefixed flags.

  This method is inteded to be paired with `define_optimizer_flags` using the
  same `prefix`, to allow Python binaries to constructed TensorFlow optimizers
  parameterized by commandline flags.

  This method expects at least two flags to have been defined and set:
    * `--<prefix>_optimizer=<optimizer name>`
    * `--<prefix>_learning_rate`

  In addition to suites of flags for each optimizer:
    * `--<prefix>_<optimizer name>_<constructor_argument>`

  For example, if `prefix='client'` this method first reads the flags:
    * `--client_optimizer`
    * `--client_learning_rate`

  If the optimizer flag is `'sgd'`, then a `tf.keras.optimizer.SGD` optimizer is
  constructed using the values in the flags prefixed with  `--client_sgd_`.

  Args:
    prefix: The same string prefix passed to `define_optimizer_flags`.

  Returns:
    A 1-arg function that accepts a learning rate and returns a
      `tf.keras.optimizers.Optimizer`.
  """
  def prefixed(basename):
    return '{}_{}'.format(prefix, basename) if prefix else basename

  optimizer_flag_name = prefixed('optimizer')
  if flags.FLAGS[optimizer_flag_name] is None:
    raise ValueError('Must specify flag --{!s}'.format(optimizer_flag_name))
  optimizer_name = flags.FLAGS[optimizer_flag_name].value
  optimizer_cls = _SUPPORTED_OPTIMIZERS.get(optimizer_name)
  if optimizer_cls is None:
    # To support additional optimizers, implement it as a
    # `tf.keras.optimizers.Optimizer` and add to the `_SUPPORTED_OPTIMIZERS`
    # dict.
    logging.error(
        'Unknown optimizer [%s], known optimziers are [%s]. To add '
        'support for an optimizer, add the optimzier class to the '
        'flag_utils._SUPPORTED_OPTIMIZERS list.', optimizer_name,
        list(_SUPPORTED_OPTIMIZERS.keys()))
    raise ValueError('`{!s}` is not a valid optimizer for flag --{!s}, must be '
                     'one of {!s}. See error log for details.'.format(
                         optimizer_name, optimizer_flag_name,
                         list(_SUPPORTED_OPTIMIZERS.keys())))

  def _has_user_value(flag):
    """Check if a commandline flag has a user set value."""
    return flag.present or flag.value != flag.default

  # Validate that the optimizers that weren't picked don't have flag values set.
  # Settings that won't be used likely means there is an expectation gap between
  # the user and the system and we should notify them.
  unused_flag_prefixes = [
      prefixed(k) for k in _SUPPORTED_OPTIMIZERS.keys() if k != optimizer_name
  ]
  mistakenly_set_flags = []
  for flag_name in flags.FLAGS:
    if not _has_user_value(flags.FLAGS[flag_name]):
      # Flag was not set by the user, skip it.
      continue
    # Otherwise the flag has a value set by the user.
    for unused_prefix in unused_flag_prefixes:
      if flag_name.startswith(unused_prefix):
        mistakenly_set_flags.append(flag_name)
        break
  if mistakenly_set_flags:
    raise ValueError('Commandline flags for optimizers other than [{!s}] '
                     '(value of --{!s}) are set. These would be ignored, '
                     'were the flags set by mistake? Flags: {!s}'.format(
                         optimizer_name, optimizer_flag_name,
                         mistakenly_set_flags))

  lr_flag_name = prefixed('learning_rate')
  lr_flag = flags.FLAGS[lr_flag_name]
  if _has_user_value(lr_flag):
    default_lr = lr_flag.value
  else:
    raise ValueError(
        'Learning rate for {!s} must be set by the flag --{!s} .'.format(
            prefix, lr_flag_name))

  flag_prefix = prefixed(optimizer_name)
  prefix_len = len(flag_prefix) + 1
  kwargs = {}
  for flag_name in flags.FLAGS:
    if not flag_name.startswith(flag_prefix):
      continue
    arg_name = flag_name[prefix_len:]
    kwargs[arg_name] = flags.FLAGS[flag_name].value

  if 'learning_rate' in kwargs:
    kwargs.pop('learning_rate')

  def optimizer_fn(learning_rate=default_lr):
    return optimizer_cls(learning_rate=learning_rate, **kwargs)
  return optimizer_fn


def remove_unused_optimizer_flags(
    prefix: str, hparam_dict: Dict[str, Any]) -> collections.OrderedDict:
  """Removes unused optimizer flags with a given prefix.

  This method is intended to be used with `define_optimizer_flags`, and is used
  to remove elements of hparam_dict associated with unused optimizer flags.

  For example, given the prefix "client", define_optimizer_flags will create
  flags including:
    *  `--client_optimizer`
    *  `--client_learning_rate`
    *  `--client_sgd_momentum`
    *  `--client_sgd_nesterov`
    *  `--client_adam_beta_1`
    *  `--client_adam_beta_2`
    *  `--client_adam_epsilon`
  and other such flags.

  However, for purposes of recording hyperparameters, we would like to only keep
  those that correspond to the optimizer selected in the flag
  --client_optimizer. This method is intended to remove the unused flags.

  For example, if `--client_optimizer=sgd` was set, then calling this method
  with the prefix `client` will remove all pairs in hparam_dict except those
  associated with the flags:
    *  `--client_optimizer`
    *  `--client_learning_rate`
    *  `--client_sgd_momentum`
    *  `--client_sgd_nesterov`

  Args:
    prefix: The prefix used to define optimizer flags, such as via
      `flag_utils.define_optimizer_flags(prefix)`. Standard examples include
      `prefix=client` and `prefix=server`.
    hparam_dict: A dictionary of (string, value) pairs corresponding to
      experiment hyperparameters.

  Returns:
    An ordered dictionary of (string, value) pairs from hparam_dict that omits
    any pairs where string = "<prefix>_<optimizer>*" but <optimizer> is not the
    one set via the flag --<prefix>_optimizer=...
  """

  def prefixed(basename):
    return '{}_{}'.format(prefix, basename) if prefix else basename

  if prefixed('optimizer') not in hparam_dict.keys():
    raise ValueError('The flag {!s} was not defined.'.format(
        prefixed('optimizer')))

  optimizer_name = hparam_dict[prefixed('optimizer')]
  if not optimizer_name:
    raise ValueError('The flag {!s} was not set. Unable to determine the '
                     'relevant optimizer.'.format(prefixed('optimizer')))

  unused_optimizer_flag_prefixes = [
      prefixed(k) for k in _SUPPORTED_OPTIMIZERS.keys() if k != optimizer_name
  ]

  def _is_used_flag(flag_name):
    # We filter by whether the flag contains an unused optimizer prefix.
    # This automatically retains any flag not of the form <prefix>_<optimizer>*.
    for unused_flag_prefix in unused_optimizer_flag_prefixes:
      if flag_name.startswith(unused_flag_prefix):
        return False
    return True

  used_flags = collections.OrderedDict()
  for (flag_name, flag_value) in hparam_dict.items():
    if _is_used_flag(flag_name):
      used_flags[flag_name] = flag_value

  return used_flags
