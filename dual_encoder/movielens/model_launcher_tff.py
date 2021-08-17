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
"""Dual encoder model runner based on Tensorflow keras API."""

import functools

from absl import app
from absl import flags
import tensorflow_federated as tff

from dual_encoder import model_utils as utils
from dual_encoder import run_utils
from dual_encoder.movielens import model as model_lib
from dual_encoder.movielens import movielens_data_gen
from utils.optimizers import optimizer_utils

# Iterative processes for training.
FEDERATED_AVERAGING = 'federated_averaging'

FLAGS = flags.FLAGS

# Model hyperparameter flags.
flags.DEFINE_enum('training_process',
                  FEDERATED_AVERAGING, [FEDERATED_AVERAGING],
                  'The iterative process for the TFF simulations.')
flags.DEFINE_integer('item_embedding_dim', 16,
                     'Output size of the item embedding layer.')
flags.DEFINE_integer('item_vocab_size', 3952, 'Size of the vocabulary.')
flags.DEFINE_enum('normalization_fn',
                  'l2_normalize', list(utils.NORMALIZATION_FN_MAP.keys()),
                  'The normalization function to be applied to embeddings. The '
                  'default function l2_normalize normalizes embeddings to a '
                  'hypersphere for cosine similarity. If none, no function is '
                  'applied and use dot product for similarity.')
flags.DEFINE_enum('context_encoder_type', model_lib.BOW,
                  [model_lib.FLATTEN, model_lib.BOW],
                  'Type of the encoder for context encoding.')
flags.DEFINE_enum('label_encoder_type', model_lib.FLATTEN,
                  [model_lib.FLATTEN, model_lib.BOW],
                  'Type of the encoder for the label encoding.')
flags.DEFINE_list(
    'context_hidden_dims', '16',
    'The number of units for each hidden layer in context tower.')
flags.DEFINE_list('context_hidden_activations', 'relu',
                  'The activations for each hidden layer in context tower.')
flags.DEFINE_list('label_hidden_dims', '',
                  'The number of units for each hidden layer in label tower.')
flags.DEFINE_list('label_hidden_activations', '',
                  'The activations for each hidden layer in label tower.')
flags.DEFINE_bool('output_embeddings', False,
                  'The output of the model are the context and label embeddings'
                  'if output_embeddings is True. Otherwise, the model outputs'
                  'batch or global similarities.')
flags.DEFINE_bool('use_global_similarity', False,
                  'Whether the model uses the global similarity or the batch'
                  'similarity. If true, using the global similarity. Otherwise,'
                  'using the batch similarity.')
flags.DEFINE_list('recall_k_values', '1,5,10',
                  'The list of recall_k values to evaluate.')
flags.DEFINE_enum('loss_function', model_lib.BATCH_SOFTMAX,
                  [model_lib.BATCH_SOFTMAX, model_lib.GLOBAL_SOFTMAX,
                   model_lib.HINGE],
                  'The loss function being used for the model training.')

# Spreadout scaling constants, see ../losses.py for more info.
flags.DEFINE_float('spreadout_context_lambda', 0.,
                   'Context spreadout scaling constant.')
flags.DEFINE_float('spreadout_label_lambda', 0.,
                   'Label spreadout scaling constant.')
flags.DEFINE_float('spreadout_cross_lambda', 0.,
                   'Cross spreadout scaling constant.')

# Regularization constants
flags.DEFINE_float('spreadout_lambda', 0.,
                   'Scaling constant for spreadout regularization.')
flags.DEFINE_float('l2_regularization', 0.,
                   'The constant to use to scale L2 regularization.')

# TFF simulation hyperparameter flags.
flags.DEFINE_integer('num_clients', 100, 'Report goal, the number of clients to'
                     'select for each training round. It applies to both the'
                     'training and evaluation.')
flags.DEFINE_integer('num_rounds', 1, 'Number of training round.')
flags.DEFINE_integer('num_evals_per_round', 1,
                     'Number of evaluations per round.')

# Optimizer configuration (this defines one or more flags per optimizer).
optimizer_utils.define_optimizer_flags('server')
optimizer_utils.define_optimizer_flags('client')

# MovieLens data generation parameter flags.
flags.DEFINE_string('input_data_dir', None,
                    'Path to the cns directory of input data.')
flags.DEFINE_float('train_fraction', 0.8,
                   'The fraction of examples used for training.')
flags.DEFINE_float('val_fraction', 0.1,
                   'The fraction of examples used for validation.')
flags.DEFINE_integer('batch_size', 16, 'The number of examples per batch.')
flags.DEFINE_integer('num_local_epochs', 1,
                     'The number of training epochs on the client per round.')
flags.DEFINE_integer('max_examples_per_user', 0,
                     'Limit the number of examples per user if larger than 0.')
flags.DEFINE_bool('shuffle_across_users', False,
                  'Whether to shuffle the examples across users.')
# Flags from 'movielens_data_gen'.
FLAGS.max_context_length = 10
FLAGS.min_timeline_length = 3

# Flags for log files.
flags.DEFINE_string('logdir', None,
                    'Path to directory where to store summaries.')


def model_fn(element_spec):
  """Create a 'tff.learning.Model' wrapping the Keras model."""
  model = model_lib.build_keras_model(
      item_vocab_size=FLAGS.item_vocab_size,
      item_embedding_dim=FLAGS.item_embedding_dim,
      spreadout_lambda=FLAGS.spreadout_lambda,
      l2_regularization=FLAGS.l2_regularization,
      normalization_fn=FLAGS.normalization_fn,
      context_encoder_type=FLAGS.context_encoder_type,
      label_encoder_type=FLAGS.label_encoder_type,
      context_hidden_dims=[int(dim) for dim in FLAGS.context_hidden_dims],
      context_hidden_activations=[
          act if act != 'None' else None
          for act in FLAGS.context_hidden_activations
      ],
      label_hidden_dims=[int(dim) for dim in FLAGS.label_hidden_dims],
      label_hidden_activations=[
          act if act != 'None' else None
          for act in FLAGS.label_hidden_activations
      ],
      output_embeddings=FLAGS.output_embeddings,
      use_global_similarity=FLAGS.use_global_similarity)

  loss = model_lib.get_loss(
      loss_function=FLAGS.loss_function,
      normalization_fn=FLAGS.normalization_fn,
      expect_embeddings=FLAGS.output_embeddings,
      spreadout_context_lambda=FLAGS.spreadout_context_lambda,
      spreadout_label_lambda=FLAGS.spreadout_label_lambda,
      spreadout_cross_lambda=FLAGS.spreadout_cross_lambda,
      use_global_similarity=FLAGS.use_global_similarity)

  metrics = model_lib.get_metrics(
      eval_top_k=[int(k) for k in FLAGS.recall_k_values],
      normalization_fn=FLAGS.normalization_fn,
      expect_embeddings=FLAGS.output_embeddings,
      use_global_similarity=FLAGS.use_global_similarity)

  if FLAGS.training_process == FEDERATED_AVERAGING:
    tff_model = tff.learning.from_keras_model(
        model,
        input_spec=element_spec,
        loss=loss,
        metrics=metrics)
  else:
    raise ValueError(
        f'Got unexpected training process function {FLAGS.training_process}.')

  return tff_model


def main(unused_argv):
  ratings_df = movielens_data_gen.read_ratings(FLAGS.input_data_dir)
  train_df, val_df, _ = movielens_data_gen.split_ratings_df(
      ratings_df=ratings_df,
      train_fraction=FLAGS.train_fraction,
      val_fraction=FLAGS.val_fraction)

  use_example_weight = not FLAGS.use_global_similarity
  train_client_datasets = movielens_data_gen.create_client_datasets(
      ratings_df=train_df,
      min_timeline_len=FLAGS.min_timeline_length,
      max_context_len=FLAGS.max_context_length,
      max_examples_per_user=FLAGS.max_examples_per_user,
      shuffle_across_users=FLAGS.shuffle_across_users,
      batch_size=FLAGS.batch_size,
      num_local_epochs=FLAGS.num_local_epochs,
      use_example_weight=use_example_weight)
  val_client_datasets = movielens_data_gen.create_client_datasets(
      ratings_df=val_df,
      min_timeline_len=FLAGS.min_timeline_length,
      max_context_len=FLAGS.max_context_length,
      max_examples_per_user=FLAGS.max_examples_per_user,
      shuffle_across_users=FLAGS.shuffle_across_users,
      batch_size=FLAGS.batch_size,
      num_local_epochs=FLAGS.num_local_epochs,
      use_example_weight=use_example_weight)

  element_spec = train_client_datasets[1].element_spec
  server_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('server')
  client_optimizer_fn = optimizer_utils.create_optimizer_fn_from_flags('client')

  if FLAGS.training_process == FEDERATED_AVERAGING:
    trainer_builder = tff.learning.build_federated_averaging_process
    eval_builder = tff.learning.build_federated_evaluation
  else:
    raise ValueError(
        f'Got unexpected training process function {FLAGS.training_process}.')

  trainer = trainer_builder(
      model_fn=functools.partial(model_fn, element_spec=element_spec),
      client_optimizer_fn=client_optimizer_fn,
      server_optimizer_fn=server_optimizer_fn
  )
  tff_evaluator = eval_builder(
      model_fn=functools.partial(model_fn, element_spec=element_spec))

  run_utils.train_and_eval(
      trainer=trainer,
      evaluator=tff_evaluator,
      num_rounds=FLAGS.num_rounds,
      train_datasets=train_client_datasets,
      test_datasets=val_client_datasets,
      num_clients_per_round=FLAGS.num_clients,
      root_output_dir=FLAGS.logdir)


if __name__ == '__main__':
  flags.mark_flag_as_required('input_data_dir')
  flags.mark_flag_as_required('logdir')
  app.run(main)
