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
"""An example training loop lossily compressing the server/client communication.

Example command line flags to use to run an experiment:
--client_optimizer=sgd
--client_learning_rate=0.2
--server_optimizer=sgd
--server_learning_rate=1.0
--use_compression=True
--broadcast_quantization_bits=8
--aggregation_quantization_bits=8
--use_sparsity_in_aggregation=True
"""

import functools

from absl import app
from absl import flags
import tensorflow as tf
import tensorflow_federated as tff

from compression import sparsity
from utils import training_utils
from utils import utils_impl
from utils.optimizers import optimizer_utils
from tensorflow_model_optimization.python.core.internal import tensor_encoding as te


with utils_impl.record_new_flags():
  # Training hyperparameters
  flags.DEFINE_integer('clients_per_round', 2,
                       'How many clients to sample per round.')
  flags.DEFINE_integer('client_epochs_per_round', 1,
                       'Number of epochs in the client to take per round.')
  flags.DEFINE_integer('client_batch_size', 20,
                       'Batch size used on the client.')
  flags.DEFINE_boolean(
      'only_digits', True, 'Whether to use the digit-only '
      'EMNIST dataset (10 characters) or the extended EMNIST '
      'dataset (62 characters).')

  # Optimizer configuration (this defines one or more flags per optimizer).
  optimizer_utils.define_optimizer_flags('server')
  optimizer_utils.define_optimizer_flags('client')

  # Compression hyperparameters.
  flags.DEFINE_boolean('use_compression', True,
                       'Whether to use compression code path.')
  flags.DEFINE_integer(
      'broadcast_quantization_bits', 8,
      'Number of quantization bits for server to client '
      'compression.')
  flags.DEFINE_integer(
      'aggregation_quantization_bits', 8,
      'Number of quantization bits for client to server '
      'compression.')
  flags.DEFINE_boolean('use_sparsity_in_aggregation', True,
                       'Whether to add sparsity to the aggregation. This will '
                       'only be used for client to server compression.')

with utils_impl.record_new_flags() as training_loop_flags:
  flags.DEFINE_integer('total_rounds', 200, 'Number of total training rounds.')
  flags.DEFINE_string(
      'experiment_name', None, 'The name of this experiment. Will be append to '
      '--root_output_dir to separate experiment results.')
  flags.DEFINE_string('root_output_dir', '/tmp/compression/',
                      'Root directory for writing experiment output.')
  flags.DEFINE_integer(
      'rounds_per_eval', 1,
      'How often to evaluate the global model on the validation dataset.')
  flags.DEFINE_integer('rounds_per_checkpoint', 50,
                       'How often to checkpoint the global model.')

# End of hyperparameter flags.

FLAGS = flags.FLAGS


def _broadcast_encoder_fn(value):
  """Function for building encoded broadcast.

  This method decides, based on the tensor size, whether to use lossy
  compression or keep it as is (use identity encoder). The motivation for this
  pattern is due to the fact that compression of small model weights can provide
  only negligible benefit, while at the same time, lossy compression of small
  weights usually results in larger impact on model's accuracy.

  Args:
    value: A tensor or variable to be encoded in server to client communication.

  Returns:
    A `te.core.SimpleEncoder`.
  """
  # TODO(b/131681951): We cannot use .from_tensor(...) because it does not
  # currently support Variables.
  spec = tf.TensorSpec(value.shape, value.dtype)
  if value.shape.num_elements() > 10000:
    return te.encoders.as_simple_encoder(
        te.encoders.uniform_quantization(FLAGS.broadcast_quantization_bits),
        spec)
  else:
    return te.encoders.as_simple_encoder(te.encoders.identity(), spec)


def _mean_encoder_fn(spec):
  """Function for building encoded mean.

  This method decides, based on the tensor size, whether to use lossy
  compression or keep it as is (use identity encoder). The motivation for this
  pattern is due to the fact that compression of small model weights can provide
  only negligible benefit, while at the same time, lossy compression of small
  weights usually results in larger impact on model's accuracy.

  Args:
    spec: A `tf.TensorSpec` for the value to be encoded in client to server
      communication.

  Returns:
    A `te.core.GatherEncoder`.
  """
  if spec.shape.num_elements() > 10000:
    if FLAGS.use_sparsity_in_aggregation:
      return te.encoders.as_gather_encoder(
          sparsity.sparse_quantizing_encoder(
              FLAGS.aggregation_quantization_bits), spec)
    else:
      return te.encoders.as_gather_encoder(
          te.encoders.uniform_quantization(FLAGS.aggregation_quantization_bits),
          spec)
  else:
    return te.encoders.as_gather_encoder(te.encoders.identity(), spec)


def run_experiment():
  """Data preprocessing and experiment execution."""
  train_client_spec = tff.simulation.baselines.ClientSpec(
      num_epochs=FLAGS.client_epochs_per_round,
      batch_size=FLAGS.client_batch_size)

  emnist_task = tff.simulation.baselines.emnist.create_character_recognition_task(
      train_client_spec,
      model_id=tff.simulation.baselines.emnist.CharacterRecognitionModel.CNN,
      only_digits=FLAGS.only_digits)

  # Build optimizer functions from flags
  client_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('client')
  server_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('server')

  if FLAGS.use_compression:
    # We create a `tff.templates.MeasuredProcess` for broadcast process and a
    # `tff.aggregators.WeightedAggregationFactory` for aggregation by providing
    # the `_broadcast_encoder_fn` and `_mean_encoder_fn` to corresponding
    # utilities. The fns are called once for each of the model weights created
    # by tff_model_fn, and return instances of appropriate encoders.
    encoded_broadcast_process = (
        tff.learning.framework.build_encoded_broadcast_process_from_model(
            emnist_task.model_fn, _broadcast_encoder_fn))
    aggregator = tff.aggregators.MeanFactory(
        tff.aggregators.EncodedSumFactory(_mean_encoder_fn))
  else:
    encoded_broadcast_process = None
    aggregator = None

  # Construct the iterative process
  iterative_process = tff.learning.build_federated_averaging_process(
      model_fn=emnist_task.model_fn,
      client_optimizer_fn=client_optimizer_fn,
      server_optimizer_fn=server_optimizer_fn,
      broadcast_process=encoded_broadcast_process,
      model_update_aggregation_factory=aggregator)
  emnist_train = emnist_task.datasets.train_data.preprocess(
      emnist_task.datasets.train_preprocess_fn)
  training_process = (
      tff.simulation.compose_dataset_computation_with_iterative_process(
          emnist_train.dataset_computation, iterative_process))

  client_selection_fn = functools.partial(
      tff.simulation.build_uniform_sampling_fn(emnist_train.client_ids),
      size=FLAGS.clients_per_round)

  validation_fn = training_utils.create_validation_fn(
      emnist_task, validation_frequency=FLAGS.rounds_per_eval)

  def validation_fn_from_state(state, round_num):
    return validation_fn(state.model, round_num)

  # Log hyperparameters to CSV
  hparam_dict = utils_impl.lookup_flag_values(utils_impl.get_hparam_flags())
  training_utils.write_hparams_to_csv(
      hparam_dict,
      root_output_dir=FLAGS.root_output_dir,
      experiment_name=FLAGS.experiment_name)

  checkpoint_manager, metrics_managers = training_utils.configure_managers(
      FLAGS.root_output_dir,
      FLAGS.experiment_name,
      rounds_per_checkpoint=FLAGS.rounds_per_checkpoint)
  tff.simulation.run_simulation(
      process=training_process,
      client_selection_fn=client_selection_fn,
      validation_fn=validation_fn_from_state,
      total_rounds=FLAGS.total_rounds,
      file_checkpoint_manager=checkpoint_manager,
      metrics_managers=metrics_managers)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))
  tff.backends.native.set_local_execution_context(max_fanout=25)
  run_experiment()


if __name__ == '__main__':
  app.run(main)
