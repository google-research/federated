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
"""Runs centralized training on CIFAR-10."""

import collections

from absl import app
from absl import flags

from fedopt_guide.cifar10_resnet import centralized_cifar10
from optimization.shared import optimizer_utils

from utils import utils_impl

_SUPPORTED_TASKS = ['cifar10']

with utils_impl.record_new_flags() as hparam_flags:
  flags.DEFINE_enum('task', None, _SUPPORTED_TASKS,
                    'Which task to perform federated training on.')

  # Generic centralized training flags
  optimizer_utils.define_optimizer_flags('centralized')
  flags.DEFINE_string(
      'experiment_name', None,
      'Name of the experiment. Part of the name of the output directory.')
  flags.mark_flag_as_required('experiment_name')
  flags.DEFINE_string(
      'root_output_dir', '/tmp/centralized_opt',
      'The top-level output directory experiment runs. --experiment_name will '
      'be appended, and the directory will contain tensorboard logs, metrics '
      'written as CSVs, and a CSV of hyperparameter choices.')
  flags.DEFINE_integer('num_epochs', 50, 'Number of epochs to train.')
  flags.DEFINE_integer('batch_size', 32,
                       'Size of batches for training and eval.')
  flags.DEFINE_integer('decay_epochs', None, 'Number of epochs before decaying '
                       'the learning rate.')
  flags.DEFINE_float('lr_decay', None, 'How much to decay the learning rate by'
                     ' at each stage.')

  # CIFAR-100 flags
  flags.DEFINE_integer('cifar10_crop_size', 24, 'The height and width of '
                       'images after preprocessing.')

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  optimizer = optimizer_utils.create_optimizer_fn_from_flags('centralized')()
  hparams_dict = collections.OrderedDict([
      (name, FLAGS[name].value) for name in hparam_flags
  ])

  common_args = collections.OrderedDict([
      ('optimizer', optimizer),
      ('experiment_name', FLAGS.experiment_name),
      ('root_output_dir', FLAGS.root_output_dir),
      ('num_epochs', FLAGS.num_epochs),
      ('batch_size', FLAGS.batch_size),
      ('decay_epochs', FLAGS.decay_epochs),
      ('lr_decay', FLAGS.lr_decay),
      ('hparams_dict', hparams_dict),
  ])

  if FLAGS.task == 'cifar10':
    centralized_cifar10.run_centralized(
        **common_args, crop_size=FLAGS.cifar100_crop_size)

  else:
    raise ValueError(
        '--task flag {} is not supported, must be one of {}.'.format(
            FLAGS.task, _SUPPORTED_TASKS))


if __name__ == '__main__':
  app.run(main)
