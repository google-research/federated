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
"""Utilities for configuring tasks for TFF simulations via flags."""

import inspect
from typing import Optional

from absl import flags
from absl import logging
import tensorflow_federated as tff

TASK_CONSTRUCTORS = {
    'cifar100_image':
        tff.simulation.baselines.cifar100.create_image_classification_task,
    'emnist_autoencoder':
        tff.simulation.baselines.emnist.create_autoencoder_task,
    'emnist_character':
        tff.simulation.baselines.emnist.create_character_recognition_task,
    'shakespeare_character':
        tff.simulation.baselines.shakespeare.create_character_prediction_task,
    'stackoverflow_tag':
        tff.simulation.baselines.stackoverflow.create_tag_prediction_task,
    'stackoverflow_word':
        tff.simulation.baselines.stackoverflow.create_word_prediction_task
}
SUPPORTED_TASKS = TASK_CONSTRUCTORS.keys()
TASK_FLAG_NAME = 'task'


def define_task_flags() -> None:
  """Defines flags to configure a `tff.simulation.baselines.BaselineTask`.

  This creates flags of the form `--<task_name>_<argument>` where `task_name` is
  one of `SUPPORTED_TASKS` and `argument` is an input argument to the associated
  task configured via `tff.simulation.baselines`.
  """
  # Create top-level flags for selecting the task.
  flags.DEFINE_enum(
      name=TASK_FLAG_NAME,
      default=None,
      enum_values=list(SUPPORTED_TASKS),
      help='Which task to configure.')
  logging.info('Defined new flag: [%s]', 'task')

  def create_flag_name(task_name, arg_name):
    return '{!s}_{!s}'.format(task_name, arg_name)

  for task_name in SUPPORTED_TASKS:
    constructor = TASK_CONSTRUCTORS[task_name]
    # Pull out the constructor parameters except for `self`.
    constructor_signature = inspect.signature(constructor)
    constructor_params = list(constructor_signature.parameters.values())

    def is_param_of_type(param, typ):
      return (param.default is None and param.annotation == Optional[typ] or
              isinstance(param.default, typ))

    for param in constructor_params:
      if param.name in [
          'train_client_spec', 'eval_client_spec', 'use_synthetic_data',
          'cache_dir'
      ]:
        continue

      if is_param_of_type(param, bool):
        define_flag_fn = flags.DEFINE_bool
      elif is_param_of_type(param, float):
        define_flag_fn = flags.DEFINE_float
      elif is_param_of_type(param, int):
        define_flag_fn = flags.DEFINE_integer
      elif is_param_of_type(param, str):
        define_flag_fn = flags.DEFINE_string
      else:
        raise NotImplementedError(
            'Cannot define flag for argument [{!s}] of type [{!s}] (default '
            'value type [{!s}]) for task [{!s}]'.format(param.name,
                                                        param.annotation,
                                                        type(param.default),
                                                        task_name))
      flag_name = create_flag_name(task_name, param.name)
      define_flag_fn(
          name=flag_name,
          default=param.default,
          help='{!s} argument for the {!s} task.'.format(param.name, task_name))
      logging.info('Defined new flag: [%s]', flag_name)


def create_task_from_flags(
    train_client_spec: tff.simulation.baselines.ClientSpec,
    eval_client_spec: Optional[tff.simulation.baselines.ClientSpec] = None,
    cache_dir: Optional[str] = None,
    use_synthetic_data: bool = False) -> tff.simulation.baselines.BaselineTask:
  """Returns a `tff.simulation.baselines.BaselineTask` from flags.

  Args:
    train_client_spec: A `tff.simulation.baselines.ClientSpec` specifying how to
      preprocess train client data.
    eval_client_spec: An optional `tff.simulation.baselines.ClientSpec`
      specifying how to preprocess evaluation client data. If set to `None`, the
      evaluation datasets will use a batch size of 64 with no extra
      preprocessing.
    cache_dir: An optional directory to cache the downloadeded datasets. If
      `None`, they will be cached to `~/.tff/`.
    use_synthetic_data: A boolean indicating whether to use synthetic data. This
      option should only be used for testing purposes, in order to avoid
      downloading the entire dataset.

  Returns:
    A `tff.simulation.baselines.BaselineTask`.
  """
  if flags.FLAGS[TASK_FLAG_NAME] is None or flags.FLAGS[
      TASK_FLAG_NAME].value is None:
    raise ValueError('Must specify flag --{!s}'.format(TASK_FLAG_NAME))
  task_name = flags.FLAGS[TASK_FLAG_NAME].value

  def _has_user_value(flag):
    """Check if a commandline flag has a user set value."""
    return flag.present or flag.value != flag.default

  unused_task_names = [x for x in SUPPORTED_TASKS if x != task_name]
  mistakenly_set_flags = []
  for flag_name in flags.FLAGS:
    if not _has_user_value(flags.FLAGS[flag_name]):
      # Flag was not set by the user, skip it.
      continue
    # Otherwise the flag has a value set by the user.
    for unused_task_name in unused_task_names:
      if flag_name.startswith(f'{unused_task_name}_'):
        mistakenly_set_flags.append(flag_name)
        break
  if mistakenly_set_flags:
    raise ValueError('Commandline flags for task other than [{!s}] are set. '
                     'Were the flags set by mistake? Flags: {!s}'.format(
                         task_name, mistakenly_set_flags))

  kwargs = {}
  prefix_len = len(task_name) + 1
  for flag_name in flags.FLAGS:
    if not flag_name.startswith(f'{task_name}_'):
      continue
    arg_name = flag_name[prefix_len:]
    kwargs[arg_name] = flags.FLAGS[flag_name].value

  task_constructor = TASK_CONSTRUCTORS[task_name]
  return task_constructor(
      train_client_spec,
      eval_client_spec=eval_client_spec,
      cache_dir=cache_dir,
      use_synthetic_data=use_synthetic_data,
      **kwargs)
