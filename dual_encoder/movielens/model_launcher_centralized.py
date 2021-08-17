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
import tensorflow as tf

from dual_encoder import model_utils as utils
from dual_encoder.movielens import model as model_lib
from dual_encoder.movielens import movielens_data_gen

# Optimizers for training.
SGD = 'sgd'
ADAM = 'adam'
ADAGRAD = 'adagrad'
# Constant for generating dataset.
_SHUFFLE_BUFFER_SIZE = 100
_PREFETCH_BUFFER_SIZE = 1

FLAGS = flags.FLAGS

# Input data directory.
flags.DEFINE_string('training_data_filepattern', None,
                    'File pattern of the training data.')
flags.DEFINE_string('testing_data_filepattern', None,
                    'File pattern of the training data.')
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

# Keras fitting hyperparams.
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate.')
flags.DEFINE_integer('batch_size', 16, 'Training batch size.')
flags.DEFINE_integer('steps_per_epoch', 10000,
                     'Number of steps to run in each epoch.')
flags.DEFINE_integer('num_epochs', 100, 'Number of training epochs.')
flags.DEFINE_integer('num_eval_steps', 1000, 'Number of eval steps.')
flags.DEFINE_enum('optimizer', ADAM, [SGD, ADAM, ADAGRAD],
                  'The optimizer for training.')

# Flags for log files.
flags.DEFINE_string('logdir', None,
                    'Path to directory where to store summaries.')


def get_shard_filenames(filepattern):
  """Get a list of filenames given a pattern.

  Args:
    filepattern: File pattern.

  Returns:
    A list of shard patterns.

  Raises:
    ValueError: if using the shard pattern, if some shards don't exist.
  """
  filenames = tf.io.gfile.glob(filepattern)

  return filenames


def get_input_fn(data_filepattern):
  """Get input_fn for dual encoder model launcher."""

  def input_fn():
    """An input_fn satisfying the TF estimator spec.

    Returns:
      A Dataset where each element is a batch of `features` dicts, passed to the
      estimator `model_fn`.

    """
    use_example_weight = not FLAGS.use_global_similarity
    input_files = get_shard_filenames(data_filepattern)
    d = tf.data.TFRecordDataset(input_files)
    d.shuffle(len(input_files))
    d = d.repeat()
    d = d.shuffle(buffer_size=_SHUFFLE_BUFFER_SIZE)
    d = d.map(
        functools.partial(
            movielens_data_gen.decode_example,
            use_example_weight=use_example_weight))
    d = d.batch(FLAGS.batch_size, drop_remainder=True)
    d = d.prefetch(_PREFETCH_BUFFER_SIZE)
    return d

  return input_fn


def create_keras_model():
  """Construct and compile dual encoder keras model."""
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

  if FLAGS.optimizer == SGD:
    optimizer = tf.keras.optimizers.SGD(FLAGS.learning_rate)
  elif FLAGS.optimizer == ADAM:
    optimizer = tf.keras.optimizers.Adam(FLAGS.learning_rate)
  elif FLAGS.optimizer == ADAGRAD:
    optimizer = tf.keras.optimizers.Adagrad(FLAGS.learning_rate, clipnorm=1.0)
  else:
    raise ValueError(f'Unsupported optimizer: {FLAGS.optimizer}.')

  model.compile(optimizer=optimizer,
                loss=model_lib.get_loss(
                    loss_function=FLAGS.loss_function,
                    normalization_fn=FLAGS.normalization_fn,
                    expect_embeddings=FLAGS.output_embeddings,
                    spreadout_context_lambda=FLAGS.spreadout_context_lambda,
                    spreadout_label_lambda=FLAGS.spreadout_label_lambda,
                    spreadout_cross_lambda=FLAGS.spreadout_cross_lambda,
                    use_global_similarity=FLAGS.use_global_similarity),
                metrics=model_lib.get_metrics(
                    eval_top_k=[int(k) for k in FLAGS.recall_k_values],
                    normalization_fn=FLAGS.normalization_fn,
                    expect_embeddings=FLAGS.output_embeddings,
                    use_global_similarity=FLAGS.use_global_similarity))
  return model


def train_and_eval(model, train_input_fn, eval_input_fn,
                   steps_per_epoch, epochs, eval_steps, callbacks):
  """Train and evaluate."""

  train_dataset = train_input_fn()
  eval_dataset = eval_input_fn()
  history = model.fit(
      x=train_dataset,
      validation_data=eval_dataset,
      steps_per_epoch=steps_per_epoch,
      epochs=epochs,
      validation_steps=eval_steps,
      callbacks=callbacks)

  return history


def main(_):
  logger = tf.get_logger()

  logger.info('Setting up train and eval input_fns.')
  train_input_fn = get_input_fn(FLAGS.training_data_filepattern)
  eval_input_fn = get_input_fn(FLAGS.testing_data_filepattern)

  logger.info('Build keras model for training and evaluation.')
  model = create_keras_model()

  train_and_eval(
      model=model,
      train_input_fn=train_input_fn,
      eval_input_fn=eval_input_fn,
      steps_per_epoch=FLAGS.steps_per_epoch,
      epochs=FLAGS.num_epochs,
      eval_steps=FLAGS.num_eval_steps,
      callbacks=tf.keras.callbacks.TensorBoard(
          log_dir=FLAGS.logdir,
          histogram_freq=0,
          embeddings_freq=0,
          update_freq='epoch'))


if __name__ == '__main__':
  flags.mark_flag_as_required('training_data_filepattern')
  flags.mark_flag_as_required('testing_data_filepattern')
  flags.mark_flag_as_required('logdir')
  app.run(main)
