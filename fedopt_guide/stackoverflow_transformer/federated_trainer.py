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
"""Runs federated training on Stackoverflow dataset."""

from typing import Any, Callable, Optional

from absl import app
from absl import flags
import tensorflow as tf
import tensorflow_federated as tff

from fedopt_guide.stackoverflow_transformer import federated_main
from optimization.shared import optimizer_utils
from utils import utils_impl


with utils_impl.record_hparam_flags() as optimizer_flags:
  # Defining optimizer flags
  optimizer_utils.define_optimizer_flags('client')
  optimizer_utils.define_optimizer_flags('server')

with utils_impl.record_hparam_flags() as training_flags:
  # Federated training hyperparameters
  flags.DEFINE_integer('client_epochs_per_round', 1,
                       'Number of epochs in the client to take per round.')
  flags.DEFINE_integer('client_batch_size', 16, 'Batch size on the clients.')
  flags.DEFINE_integer('clients_per_round', 10,
                       'How many clients to sample per round.')
  flags.DEFINE_integer('client_datasets_random_seed', None,
                       'Random seed for client sampling.')

  # Training loop configuration
  flags.DEFINE_string(
      'experiment_name', None, 'The name of this experiment. Will be append to '
      '--root_output_dir to separate experiment results.')
  flags.mark_flag_as_required('experiment_name')
  flags.DEFINE_string('root_output_dir', '/tmp/fedopt_guide/',
                      'Root directory for writing experiment output.')
  flags.DEFINE_integer('total_rounds', 2, 'Number of total training rounds.')
  flags.DEFINE_integer(
      'rounds_per_eval', 1,
      'How often to evaluate the global model on the validation dataset.')
  flags.DEFINE_integer('rounds_per_checkpoint', 50,
                       'How often to checkpoint the global model.')

with utils_impl.record_hparam_flags() as transformer_flags:
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
  flags.DEFINE_integer('max_elements_per_user', 64,
                       'Max number of training examples to use per user.')

FLAGS = flags.FLAGS


def get_hparam_flags():
  """Returns an ordered dictionary of pertinent hyperparameter flags."""
  hparam_dict = utils_impl.lookup_flag_values(training_flags)

  # Update with optimizer flags corresponding to the chosen optimizers.
  opt_flag_dict = utils_impl.lookup_flag_values(optimizer_flags)
  opt_flag_dict = optimizer_utils.remove_unused_flags('client', opt_flag_dict)
  opt_flag_dict = optimizer_utils.remove_unused_flags('server', opt_flag_dict)
  hparam_dict.update(opt_flag_dict)

  # Update with task-specific flags.
  task_hparam_dict = utils_impl.lookup_flag_values(transformer_flags)
  hparam_dict.update(task_hparam_dict)

  return hparam_dict


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  client_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('client')
  server_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('server')

  def iterative_process_builder(
      model_fn: Callable[[], tff.learning.Model],
      client_weight_fn: Optional[Callable[[Any], tf.Tensor]] = None,
  ) -> tff.templates.IterativeProcess:
    """Creates an iterative process using a given TFF `model_fn`.

    Args:
      model_fn: A no-arg function returning a `tff.learning.Model`.
      client_weight_fn: Optional function that takes the output of
        `model.report_local_outputs` and returns a tensor providing the weight
        in the federated average of model deltas. If not provided, the default
        is the total number of examples processed on device.

    Returns:
      A `tff.templates.IterativeProcess`.
    """

    return tff.learning.build_federated_averaging_process(
        model_fn=model_fn,
        client_optimizer_fn=client_optimizer_fn,
        server_optimizer_fn=server_optimizer_fn,
        client_weighting=client_weight_fn,
        use_experimental_simulation_loop=True)

  federated_main.run_federated(
      iterative_process_builder=iterative_process_builder,
      client_epochs_per_round=FLAGS.client_epochs_per_round,
      client_batch_size=FLAGS.client_batch_size,
      clients_per_round=FLAGS.clients_per_round,
      max_elements_per_user=FLAGS.max_elements_per_user,
      total_rounds=FLAGS.total_rounds,
      vocab_size=FLAGS.vocab_size,
      dim_embed=FLAGS.dim_embed,
      dim_model=FLAGS.dim_model,
      dim_hidden=FLAGS.dim_hidden,
      num_heads=FLAGS.num_heads,
      num_layers=FLAGS.num_layers,
      dropout=FLAGS.dropout,
      experiment_name=FLAGS.experiment_name,
      root_output_dir=FLAGS.root_output_dir,
      client_datasets_random_seed=FLAGS.client_datasets_random_seed,
      rounds_per_eval=FLAGS.rounds_per_eval,
      rounds_per_checkpoint=FLAGS.rounds_per_checkpoint,
      hparam_dict=get_hparam_flags())


if __name__ == '__main__':
  app.run(main)
