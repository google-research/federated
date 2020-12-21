# Copyright 2020, Google LLC.
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
"""Runs centralized training on Google Landmark dataset."""

from absl import app
from absl import flags

from fedopt_guide.gld23k_mobilenet import centralized_main
from fedopt_guide.gld23k_mobilenet import dataset
from optimization.shared import optimizer_utils
from utils import utils_impl

with utils_impl.record_new_flags() as hparam_flags:
  # Generic centralized training flags
  optimizer_utils.define_optimizer_flags('centralized')
  flags.DEFINE_string(
      'experiment_name', None,
      'Name of the experiment. Part of the name of the output directory.')
  flags.mark_flag_as_required('experiment_name')
  flags.DEFINE_string(
      'root_output_dir', '/tmp/fedopt_guide',
      'The top-level output directory experiment runs. --experiment_name will '
      'be appended, and the directory will contain tensorboard logs, metrics '
      'written as CSVs, and a CSV of hyperparameter choices.')
  flags.DEFINE_integer('num_epochs', 50, 'Number of epochs to train.')
  flags.DEFINE_integer('batch_size', 128,
                       'Size of batches for training and eval.')

  # GLD flags
  flags.DEFINE_enum('dataset_type', 'gld23k', ['gld23k', 'gld160k'],
                    'Whether to run on gld23k or gld160k.')
  flags.DEFINE_integer('image_size', 224,
                       'The height and width of images after preprocessing.')
  flags.DEFINE_integer(
      'num_groups', 8, 'The number of groups to use in the GroupNorm layers of '
      'MobilenetV2.')
  flags.DEFINE_float(
      'dropout_prob', None,
      'Probability of setting a weight to zero in the dropout layer of '
      'MobilenetV2. Must be in the range [0, 1). Setting it to None (default) '
      'or zero means no dropout.')

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  hparams_dict = utils_impl.lookup_flag_values(hparam_flags)
  hparams_dict = optimizer_utils.remove_unused_flags('centralized',
                                                     hparams_dict)

  dataset_type = dataset.DatasetType.GLD23K
  if FLAGS.dataset_type == 'gld160k':
    dataset_type = dataset.DatasetType.GLD160K

  centralized_main.run_centralized(
      optimizer=optimizer_utils.create_optimizer_fn_from_flags('centralized')(),
      image_size=FLAGS.image_size,
      num_epochs=FLAGS.num_epochs,
      batch_size=FLAGS.batch_size,
      num_groups=FLAGS.num_groups,
      dataset_type=dataset_type,
      experiment_name=FLAGS.experiment_name,
      root_output_dir=FLAGS.root_output_dir,
      dropout_prob=FLAGS.dropout_prob,
      hparams_dict=hparams_dict)


if __name__ == '__main__':
  app.run(main)
