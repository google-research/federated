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
"""Trains an EMNIST classification model with FLARS using TFF."""

import collections
import functools

from absl import app
from absl import flags
import tensorflow_federated as tff

from flars import flars_fedavg
from flars import flars_optimizer
from utils import training_utils
from utils import utils_impl
from utils.optimizers import optimizer_utils

with utils_impl.record_new_flags() as hparam_flags:
  # Metadata
  flags.DEFINE_string(
      'exp_name', 'emnist', 'Unique name for the experiment, suitable for use '
      'in filenames.')

  # Training hyperparameters
  flags.DEFINE_boolean(
      'digit_only_emnist', True,
      'Whether to train on the digits only (10 classes) data '
      'or the full data (62 classes).')
  flags.DEFINE_integer('total_rounds', 500, 'Number of total training rounds.')
  flags.DEFINE_integer('rounds_per_eval', 1, 'How often to evaluate')
  flags.DEFINE_integer(
      'rounds_per_checkpoint', 25,
      'How often to emit a state checkpoint. Higher numbers '
      'mean more lost work in case of failure, lower numbers '
      'mean more overhead per round.')
  flags.DEFINE_integer('train_clients_per_round', 2,
                       'How many clients to sample per round.')
  flags.DEFINE_integer('client_epochs_per_round', 1,
                       'Number of epochs in the client to take per round.')
  flags.DEFINE_integer('batch_size', 20, 'Batch size used on the client.')

  # Client optimizer configuration (it defines one or more flags per optimizer).
  optimizer_utils.define_optimizer_flags('client')

  # Server optimizer configuration (it defines one or more flags per optimizer).
  flags.DEFINE_float('server_learning_rate', 1., 'Server learning rate.')
  flags.DEFINE_float(
      'server_momentum', 0.9,
      'Server momentum. This is also the `beta1` parameter for '
      'the Yogi optimizer.')

  # Parameter for FLARS.
  flags.DEFINE_float('max_ratio', 0.1, 'max_ratio for optimizer FLARS.')

# End of hyperparameter flags.

# Output directory flags.
flags.DEFINE_string(
    'root_output_dir', '/tmp/emnist_fedavg/',
    'Root directory for writing experiment output. This will '
    'be the destination for metrics CSV files, Tensorboard log '
    'directory, and checkpoint files.')
flags.DEFINE_string(
    'experiment_name', None, 'The name of this experiment. Will be append to '
    '--root_output_dir to separate experiment results.')

FLAGS = flags.FLAGS


def _run_experiment():
  """Data preprocessing and experiment execution."""
  train_client_spec = tff.simulation.baselines.ClientSpec(
      num_epochs=FLAGS.client_epochs_per_round, batch_size=FLAGS.batch_size)
  emnist_task = tff.simulation.baselines.emnist.create_character_recognition_task(
      train_client_spec,
      model_id=tff.simulation.baselines.emnist.CharacterRecognitionModel.CNN,
      only_digits=FLAGS.digit_only_emnist)

  client_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('client')
  server_optimizer_fn = functools.partial(
      flars_optimizer.FLARSOptimizer,
      learning_rate=FLAGS.server_learning_rate,
      momentum=FLAGS.server_momentum,
      max_ratio=FLAGS.max_ratio)

  iterative_process = flars_fedavg.build_federated_averaging_process(
      emnist_task.model_fn,
      client_optimizer_fn=client_optimizer_fn,
      server_optimizer_fn=server_optimizer_fn)
  emnist_train = emnist_task.datasets.train_data.preprocess(
      emnist_task.datasets.train_preprocess_fn)
  training_process = (
      tff.simulation.compose_dataset_computation_with_iterative_process(
          emnist_train.dataset_computation, iterative_process))

  client_selection_fn = functools.partial(
      tff.simulation.build_uniform_sampling_fn(emnist_train.client_ids),
      size=FLAGS.train_clients_per_round)

  validation_data = emnist_task.datasets.get_centralized_test_data()
  federated_eval = tff.learning.build_federated_evaluation(emnist_task.model_fn)

  # TODO(b/210890827): Use a polymorphic computation if possible
  @tff.federated_computation(training_process.initialize.type_signature.result,
                             federated_eval.type_signature.parameter[1])
  def evaluation_fn(state, evaluation_data):
    return federated_eval(state.model, evaluation_data)

  evaluation_selection_fn = lambda _: [validation_data]

  hparam_dict = collections.OrderedDict([
      (name, FLAGS[name].value) for name in hparam_flags
  ])
  hparam_dict = optimizer_utils.remove_unused_flags('client', hparam_dict)
  training_utils.write_hparams_to_csv(
      hparam_dict,
      root_output_dir=FLAGS.root_output_dir,
      experiment_name=FLAGS.experiment_name)

  program_state_manager, metrics_managers = training_utils.create_managers(
      FLAGS.root_output_dir, FLAGS.experiment_name)
  tff.simulation.run_training_process(
      training_process=training_process,
      training_selection_fn=client_selection_fn,
      total_rounds=FLAGS.total_rounds,
      evaluation_fn=evaluation_fn,
      evaluation_selection_fn=evaluation_selection_fn,
      rounds_per_evaluation=FLAGS.rounds_per_eval,
      program_state_manager=program_state_manager,
      rounds_per_saving_program_state=FLAGS.rounds_per_checkpoint,
      metrics_managers=metrics_managers)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))
  _run_experiment()


if __name__ == '__main__':
  app.run(main)
