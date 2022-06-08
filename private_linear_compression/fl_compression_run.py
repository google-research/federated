# Copyright 2021, Google LLC. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runs federated training tasks."""

import enum
import functools
import os
import pprint
from typing import Optional, Dict, Any, Tuple, Sequence

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_federated as tff

from private_linear_compression import count_sketching
from private_linear_compression import count_sketching_utils
from private_linear_compression import ddp_compression
from utils import task_utils
from utils import training_utils
from utils import utils_impl
from utils.optimizers import optimizer_utils

# Hard-code the total number of clients for the datasets. Task names are defined
# in `utiis/task_utils`. We limit to a subset of the tasks.
_SUPPORTED_TASKS_NUM_CLIENTS = {
    'emnist_character': 3400,
    'stackoverflow_word': 342477
}
_SUPPORTED_TASKS = list(_SUPPORTED_TASKS_NUM_CLIENTS.keys())


class AggregatorType(enum.Enum):
  NODP = 'nodp'
  DDP = 'ddp'
  CDP = 'cdp'


with utils_impl.record_hparam_flags() as optimizer_flags:
  optimizer_utils.define_optimizer_flags('client')
  optimizer_utils.define_optimizer_flags('server')

with utils_impl.record_hparam_flags() as shared_flags:
  # Federated training hyperparameters.
  _CLIENT_EPOCHS_PER_ROUND = flags.DEFINE_integer(
      'client_epochs_per_round', 1,
      'Number of epochs in the client to take per round.')
  _CLIENT_BATCH_SIZE = flags.DEFINE_integer('client_batch_size', 32,
                                            'Batch size on the clients.')
  _CLIENTS_PER_ROUND = flags.DEFINE_integer(
      'clients_per_round', 100, 'How many clients to sample per round.')
  _CLIENT_DATASETS_RANDOM_SEED = flags.DEFINE_integer(
      'client_datasets_random_seed', 42, 'Random seed for client sampling.')
  _MAX_ELEMENTS_PER_CLIENT = flags.DEFINE_integer(
      'max_elements_per_client', None, 'Maximum number of '
      'elements for each training client. If set to None, all '
      'available examples are used.')

  # Training loop configuration
  _TOTAL_ROUNDS = flags.DEFINE_integer('total_rounds', 1500,
                                       'Number of total training rounds.')
  _EXPERIMENT_NAME = flags.DEFINE_string(
      'experiment_name', None, 'The name of this experiment. Will be appended '
      'to --root_output_dir to separate experiment results.')
  _ROOT_OUTPUT_DIR = flags.DEFINE_string(
      'root_output_dir', '/tmp/ddp_fl/',
      'Root directory for writing experiment output.')
  _ROUNDS_PER_EVAL = flags.DEFINE_integer(
      'rounds_per_eval', 1,
      'How often to evaluate the global model on the validation dataset.')
  _NUM_VALIDATION_EXAMPLES = flags.DEFINE_integer(
      'num_validation_examples', -1, 'The number of validation'
      'examples to use. If set to -1, all available examples '
      'are used.')
  _ROUNDS_PER_CHECKPOINT = flags.DEFINE_integer(
      'rounds_per_checkpoint', 50, 'How often to checkpoint the global model.')

with utils_impl.record_hparam_flags() as compression_flags:
  _COMPRESSION_RATE = flags.DEFINE_float(
      'compression_rate', 1.,
      'Ratio of the gradient dimension to the compressed dimension. '
      'Higher indicates more compression. Must be >=1 unless it is set to 0 '
      'for no compression.')
  _DECODE_METHOD = flags.DEFINE_enum_class(
      'decode_method', count_sketching_utils.DecodeMethod.MEAN,
      count_sketching_utils.DecodeMethod,
      'Method for decoding compressed gradients.')
  _NUM_REPEATS = flags.DEFINE_integer(
      'num_repeats', 15,
      'Number of repeats to use for sketching, i.e., the number of rows in the'
      'sketch.')
  _ROTATION_TYPE = flags.DEFINE_enum(
      'rotation_type', 'hd', ['hd', 'dft'],
      '`hd` represents the randomized Hadamard transform and'
      '`dft` the discrete Fourier transform.')

with utils_impl.record_hparam_flags() as ddp_flags:
  _DP_MECHANISM = flags.DEFINE_enum(
      'dp_mechanism', 'distributed', ['central', 'distributed'],
      'Sets which type of Differential Privacy mechanism to use. Central DP is '
      'the baseline.')
  _NOISE_MULTIPLIER = flags.DEFINE_float(
      'noise_multiplier',
      0.5,
      'Amount of noise to add, specifying the epsilon together with delta. '
      'No DP is used if set to 0.0.',
      lower_bound=0.0)
  _NUM_BITS = flags.DEFINE_integer(
      'num_bits',
      16,
      'Number of bits for quantization.',
      lower_bound=0,
      upper_bound=32)

with utils_impl.record_hparam_flags() as task_flags:
  # Defines "--task" (options from `task_utils`) and "--<task>_<arg>" flags
  # aligned with input args at `tff.simulation.baselines.*` tasks.
  task_utils.define_task_flags()

FLAGS = flags.FLAGS


def _create_if_not_exists(path: str):
  """Creates a directory if it does not already exist."""
  try:
    tf.io.gfile.makedirs(path)
  except tf.errors.OpError:
    logging.info('Skipping creation of directory [%s], already exists', path)


def _write_hyper_params(params_dict: Dict[str, Any], experiment_name: str,
                        root_output_dir: str):
  """Writes hyper parameters to designated file location."""
  # Also add meta info.
  params_dict['_expname'] = experiment_name
  params_dict['_rootdir'] = root_output_dir
  results_dir = os.path.join(root_output_dir, 'results', experiment_name)
  hparam_file = os.path.join(results_dir, 'hparams.txt')
  _create_if_not_exists(results_dir)
  with tf.io.gfile.GFile(hparam_file, 'w') as f:
    pprint.pprint(params_dict, stream=f)


def _create_1m_cnn_model(only_digits: bool = False,
                         seed: Optional[int] = 0) -> tf.keras.Model:
  """A CNN model with slightly under 2^20 (roughly 1 million) params.

  A simple CNN model for the EMNIST character recognition task that is very
  similar to the default recommended model from `create_conv_dropout_model`
  but has slightly under 2^20 parameters. This is useful if the downstream task
  involves randomized Hadamard transform, which requires the model weights /
  gradients / deltas concatednated as a single vector to be padded to the
  nearest power-of-2 dimensions.

  This model is used in https://arxiv.org/abs/2102.06387.

  When `only_digits=False`, the returned model has 1,018,174 trainable
  parameters. For `only_digits=True`, the last dense layer is slightly smaller.

  Args:
    only_digits: If True, uses a final layer with 10 outputs, for use with the
      digits only EMNIST dataset. If False, uses 62 outputs for the larger
      dataset.
    seed: A random seed governing the model initialization and layer randomness.

  Returns:
    A `tf.keras.Model`.
  """
  data_format = 'channels_last'
  initializer = tf.keras.initializers.GlorotUniform(seed=seed)

  model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(
          32,
          kernel_size=(3, 3),
          activation='relu',
          data_format=data_format,
          input_shape=(28, 28, 1),
          kernel_initializer=initializer),
      tf.keras.layers.MaxPool2D(pool_size=(2, 2), data_format=data_format),
      tf.keras.layers.Conv2D(
          64,
          kernel_size=(3, 3),
          activation='relu',
          data_format=data_format,
          kernel_initializer=initializer),
      tf.keras.layers.Dropout(0.25),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(
          128, activation='relu', kernel_initializer=initializer),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(
          10 if only_digits else 62,
          activation=tf.nn.softmax,
          kernel_initializer=initializer),
  ])

  return model


def _get_aggregator_type(noise_multiplier: float,
                         dp_mechanism: str) -> AggregatorType:
  """Calculates the `AggregatorType` based on the `noise_multiplier`."""
  if noise_multiplier == 0.0:
    return AggregatorType.NODP
  elif dp_mechanism == 'central':
    return AggregatorType.CDP
  else:
    return AggregatorType.DDP


def _get_aggregator(
    noise_multiplier: float, dp_mechanism: str, expected_clients_per_round: int,
    bits: int, compression_rate: float, num_repeats: int,
    decode_method: count_sketching_utils.DecodeMethod, rotation_type: str
) -> Tuple[tff.aggregators.UnweightedAggregationFactory, Dict[str, Any]]:
  """Instantiates and logs an `agg_factory` using specified args."""

  aggregator_type = _get_aggregator_type(noise_multiplier, dp_mechanism)

  if aggregator_type == AggregatorType.NODP:
    if compression_rate > 0:
      params = {
          'min_compression_rate': compression_rate,
          'decode_method': decode_method,
          'num_repeats': num_repeats
      }
      agg_factory = tff.aggregators.UnweightedMeanFactory()
      agg_factory = count_sketching.GradientCountSketchFactory(
          **params, inner_agg_factory=agg_factory)
      logging.info(
          'Using vanilla aggregation with compression with parameters: ')
      logging.info(pprint.pformat(params))
    else:
      agg_factory = tff.aggregators.UnweightedMeanFactory()
      logging.info('Using vanilla aggregation without compression')

  elif aggregator_type == AggregatorType.CDP:
    params = {
        'noise_multiplier': noise_multiplier,
        'expected_clients_per_round': expected_clients_per_round,
        'compression_rate': compression_rate,
        'decode_method': decode_method,
        'num_repeats': num_repeats,
    }
    agg_factory = ddp_compression.compressed_central_dp_factory(**params)

    logging.info(
        'Using Central DP aggregatation with compression with parameters: ')
    logging.info(pprint.pformat(params))
  else:
    params = {
        'noise_multiplier': noise_multiplier,
        'expected_clients_per_round': expected_clients_per_round,
        'compression_rate': compression_rate,
        'bits': bits,
        'decode_method': decode_method,
        'num_repeats': num_repeats,
        'rotation_type': rotation_type
    }
    agg_factory = ddp_compression.compressed_ddp_factory(**params)

    logging.info(
        'Using Distributed DP aggregatation with compression with parameters: ')
    logging.info(pprint.pformat(params))

  return agg_factory, params


def main(argv: Sequence[Any]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Expected no command-line arguments, '
                         'got: {}'.format(argv))

  client_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('client')
  server_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('server')

  train_client_spec = tff.simulation.baselines.ClientSpec(
      num_epochs=_CLIENT_EPOCHS_PER_ROUND.value,
      batch_size=_CLIENT_BATCH_SIZE.value,
      max_elements=_MAX_ELEMENTS_PER_CLIENT.value)

  if FLAGS.task == 'emnist_character':
    # Since we use a custom model for EMNIST, we need to manually construct the
    # TFF datasets and the TFF `Task` object.
    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(
        only_digits=False)
    eval_client_spec = tff.simulation.baselines.ClientSpec(
        num_epochs=1, batch_size=64, shuffle_buffer_size=1)  # No shuffling.

    emnist_preprocessing = tff.simulation.baselines.emnist.emnist_preprocessing
    train_preprocess_fn = emnist_preprocessing.create_preprocess_fn(
        train_client_spec, emnist_task='character_recognition')
    eval_preprocess_fn = emnist_preprocessing.create_preprocess_fn(
        eval_client_spec, emnist_task='character_recognition')

    task_datasets = tff.simulation.baselines.task_data.BaselineTaskDatasets(
        train_data=emnist_train,
        test_data=emnist_test,
        validation_data=None,
        train_preprocess_fn=train_preprocess_fn,
        eval_preprocess_fn=eval_preprocess_fn)

    def emnist_model_fn():
      return tff.learning.from_keras_model(
          keras_model=_create_1m_cnn_model(),
          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
          input_spec=task_datasets.element_type_structure,
          metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    task = tff.simulation.baselines.baseline_task.BaselineTask(
        task_datasets, emnist_model_fn)

  elif FLAGS.task in _SUPPORTED_TASKS:
    task = task_utils.create_task_from_flags(train_client_spec)
  else:
    raise ValueError(f'Unsupported task "{FLAGS.task}". Must be one of '
                     f'{_SUPPORTED_TASKS}.')

  aggregation_factory, params = _get_aggregator(
      _NOISE_MULTIPLIER.value, _DP_MECHANISM.value, _CLIENTS_PER_ROUND.value,
      _NUM_BITS.value, _COMPRESSION_RATE.value, _NUM_REPEATS.value,
      _DECODE_METHOD.value, _ROTATION_TYPE.value)
  _write_hyper_params(params, _EXPERIMENT_NAME.value, _ROOT_OUTPUT_DIR.value)

  iterative_process = tff.learning.build_federated_averaging_process(
      model_fn=task.model_fn,
      server_optimizer_fn=server_optimizer_fn,
      client_weighting=tff.learning.ClientWeighting.UNIFORM,
      client_optimizer_fn=client_optimizer_fn,
      model_update_aggregation_factory=aggregation_factory,
      use_experimental_simulation_loop=True)

  train_data = task.datasets.train_data.preprocess(
      task.datasets.train_preprocess_fn)
  training_process = (
      tff.simulation.compose_dataset_computation_with_iterative_process(
          train_data.dataset_computation, iterative_process))

  training_selection_fn = functools.partial(
      tff.simulation.build_uniform_sampling_fn(
          train_data.client_ids,
          random_seed=_CLIENT_DATASETS_RANDOM_SEED.value),
      size=_CLIENTS_PER_ROUND.value)

  test_data = task.datasets.get_centralized_test_data()
  validation_data = test_data.take(_NUM_VALIDATION_EXAMPLES.value)
  federated_eval = tff.learning.build_federated_evaluation(task.model_fn)
  evaluation_selection_fn = lambda round_num: [validation_data]

  def evaluation_fn(state, evaluation_data):
    return federated_eval(state.model, evaluation_data)

  program_state_manager, metrics_managers = training_utils.create_managers(
      _ROOT_OUTPUT_DIR.value, _EXPERIMENT_NAME.value)
  state = tff.simulation.run_training_process(
      training_process=training_process,
      training_selection_fn=training_selection_fn,
      total_rounds=_TOTAL_ROUNDS.value,
      evaluation_fn=evaluation_fn,
      evaluation_selection_fn=evaluation_selection_fn,
      rounds_per_evaluation=_ROUNDS_PER_EVAL.value,
      program_state_manager=program_state_manager,
      rounds_per_saving_program_state=_ROUNDS_PER_CHECKPOINT.value,
      metrics_managers=metrics_managers)

  test_metrics = federated_eval(state.model, [test_data])
  for metrics_manager in metrics_managers:
    metrics_manager.release(test_metrics, _TOTAL_ROUNDS.value + 1)


if __name__ == '__main__':
  app.run(main)
