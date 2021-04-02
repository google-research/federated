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
"""Runs centralized training on Stackoverflow dataset."""

from absl import app
from absl import flags

from fedopt_guide.stackoverflow_transformer import centralized_main
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
  flags.DEFINE_integer('max_batches', -1, 'Maximum number of batches.')

  # Transformer flags
  flags.DEFINE_integer('vocab_size', 10000,
                       'Vocab size for normal tokens.')
  flags.DEFINE_integer('dim_embed', 96, 'Dimension of the token embeddings.')
  flags.DEFINE_integer('dim_model', 512,
                       'Dimension of features of MultiHeadAttention layers.')
  flags.DEFINE_integer('dim_hidden', 2048,
                       'Dimension of hidden layers of the FFN.')
  flags.DEFINE_integer('num_heads', 8,
                       'Number of attention heads.')
  flags.DEFINE_integer('num_layers', 1,
                       'Number of Transformer blocks.')
  flags.DEFINE_integer('sequence_length', 20,
                       'The maximum number of words to take for each sequence.')
  flags.DEFINE_float('dropout', 0.1, 'Dropout rate.')

FLAGS = flags.FLAGS


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  hparams_dict = utils_impl.lookup_flag_values(hparam_flags)
  hparams_dict = optimizer_utils.remove_unused_flags('centralized',
                                                     hparams_dict)

  centralized_main.run_centralized(
      optimizer_utils.create_optimizer_fn_from_flags('centralized')(),
      FLAGS.num_epochs,
      FLAGS.batch_size,
      vocab_size=FLAGS.vocab_size,
      dim_embed=FLAGS.dim_embed,
      dim_model=FLAGS.dim_model,
      dim_hidden=FLAGS.dim_hidden,
      num_heads=FLAGS.num_heads,
      num_layers=FLAGS.num_layers,
      dropout=FLAGS.dropout,
      experiment_name=FLAGS.experiment_name,
      root_output_dir=FLAGS.root_output_dir,
      max_batches=FLAGS.max_batches,
      hparams_dict=hparams_dict)


if __name__ == '__main__':
  app.run(main)
