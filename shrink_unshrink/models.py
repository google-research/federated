# Copyright 2020, The TensorFlow Federated Authors.
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
"""Creates a pair of recurrant models for the Stack Overflow next word prediction task.

Modified version of
tff.simulation.baselines.stackoverflow.create_word_prediction_task and dependent
functions which allows for different sized recurrant models
"""
import functools
import tensorflow as tf
import tensorflow_federated as tff


class TransposableEmbedding(tf.keras.layers.Layer):
  """A Keras layer implementing a transposed projection output layer."""

  def __init__(self, embedding_layer: tf.keras.layers.Embedding):
    super().__init__()
    self.embeddings = embedding_layer.embeddings

  # Placing `tf.matmul` under the `call` method is important for backpropagating
  # the gradients of `self.embeddings` in graph mode.
  def call(self, inputs):
    return tf.matmul(inputs, self.embeddings, transpose_b=True)


def create_recurrent_model(vocab_size: int,
                           embedding_size: int = 96,
                           num_lstm_layers: int = 1,
                           lstm_size: int = 670,
                           shared_embedding: bool = False) -> tf.keras.Model:
  """Constructs a recurrent model with an initial embeding layer.

  The resulting model embeds sequences of integer tokens (whose values vary
  between `0` and `vocab_size-1`) into an `embedding_size`-dimensional space.
  It then applies `num_lstm_layers` LSTM layers, each of size `lstm_size`.
  Each LSTM is followed by a dense layer mapping the output to `embedding_size`
  units. The model then has a final dense layer mapping to `vocab_size` logits
  units. Note that this model does not compute any kind of softmax on the final
  logits. This should instead be done in the loss function for the purposes of
  backpropagation.

  Args:
    vocab_size: Vocabulary size to use in the initial embedding layer.
    embedding_size: The size of the embedding layer.
    num_lstm_layers: The number of LSTM layers in the model.
    lstm_size: The size of each LSTM layer.
    shared_embedding: If set to `True`, the final layer of the model is a dense
      layer given by the transposition of the embedding layer. If `False`, the
      final dense layer is instead learned separately.

  Returns:
    An uncompiled `tf.keras.Model`.
  """
  if vocab_size < 1:
    raise ValueError('vocab_size must be a positive integer.')
  if embedding_size < 1:
    raise ValueError('embedding_size must be a positive integer.')
  if num_lstm_layers < 1:
    raise ValueError('num_lstm_layers must be a positive integer.')
  if lstm_size < 1:
    raise ValueError('lstm_size must be a positive integer.')

  inputs = tf.keras.layers.Input(shape=(None,))
  input_embedding = tf.keras.layers.Embedding(
      input_dim=vocab_size, output_dim=embedding_size, mask_zero=True)
  embedded = input_embedding(inputs)
  projected = embedded

  for _ in range(num_lstm_layers):
    layer = tf.keras.layers.LSTM(lstm_size, return_sequences=True)
    processed = layer(projected)
    projected = tf.keras.layers.Dense(embedding_size)(processed)

  if shared_embedding:
    transposed_embedding = TransposableEmbedding(input_embedding)
    logits = transposed_embedding(projected)
  else:
    logits = tf.keras.layers.Dense(vocab_size, activation=None)(projected)

  return tf.keras.Model(inputs=inputs, outputs=logits)


def make_big_and_small_stackoverflow_model_fn(my_task,
                                              vocab_size=10000,
                                              num_out_of_vocab_buckets=1,
                                              big_embedding_size=96,
                                              big_lstm_size=670,
                                              small_embedding_size=72,
                                              small_lstm_size=503):
  """Generates two model functions for a given task.

  This code is a modified version of
  tff.simulation.baselines.stackoverflow.create_word_prediction_task

  Args:
    my_task: a tff.simulation.baselines.BaselineTask object
    vocab_size: an integer specifying the vocab size
    num_out_of_vocab_buckets: an integer specifying the number of out of vocab
      buckets
    big_embedding_size: an integer specifying the size of the embedding layer of
      the big model
    big_lstm_size: an integer specifying the size of the lstm layer of the big
      model
    small_embedding_size: an integer specifying the size of the embedding layer
      of the small model
    small_lstm_size: an integer specifying the size of the lstm layer of the
      small model

  Returns:
    Two model_fn functions
  """

  extended_vocab_size = vocab_size + 3 + num_out_of_vocab_buckets

  def big_stackoverflownwp_rnn_model_fn():
    return tff.learning.from_keras_model(
        keras_model=create_recurrent_model(
            vocab_size=extended_vocab_size,
            embedding_size=big_embedding_size,
            lstm_size=big_lstm_size),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        input_spec=my_task.datasets.element_type_structure,
        # metrics=metrics_builder()
    )

  # the standard size corresponding the stackoverflow baseline task
  # has embedding_size=96, lstm_size=670
  def small_stackoverflownwp_rnn_model_fn():
    return tff.learning.from_keras_model(
        keras_model=create_recurrent_model(
            vocab_size=extended_vocab_size,
            embedding_size=small_embedding_size,
            lstm_size=small_lstm_size),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        input_spec=my_task.datasets.element_type_structure,
        # metrics=metrics_builder()
    )

  return big_stackoverflownwp_rnn_model_fn, small_stackoverflownwp_rnn_model_fn


def create_conv_dropout_model(conv1_filters=32,
                              conv2_filters=64,
                              dense_size=128,
                              only_digits: bool = True) -> tf.keras.Model:
  """Create a convolutional network with dropout.

  When `only_digits=True`, the summary of returned model is
  ```
  Model: "sequential"
  _________________________________________________________________
  Layer (type)                 Output Shape              Param #
  =================================================================
  reshape (Reshape)            (None, 28, 28, 1)         0
  _________________________________________________________________
  conv2d (Conv2D)              (None, 26, 26, 32)        320
  _________________________________________________________________
  conv2d_1 (Conv2D)            (None, 24, 24, 64)        18496
  _________________________________________________________________
  max_pooling2d (MaxPooling2D) (None, 12, 12, 64)        0
  _________________________________________________________________
  dropout (Dropout)            (None, 12, 12, 64)        0
  _________________________________________________________________
  flatten (Flatten)            (None, 9216)              0
  _________________________________________________________________
  dense (Dense)                (None, 128)               1179776
  _________________________________________________________________
  dropout_1 (Dropout)          (None, 128)               0
  _________________________________________________________________
  dense_1 (Dense)              (None, 10)                1290
  =================================================================
  Total params: 1,199,882
  Trainable params: 1,199,882
  Non-trainable params: 0
  ```
  For `only_digits=False`, the last dense layer is slightly larger.

  Args:
    conv1_filters: The number of convolutional filters in the 1st convolutional
      layer
    conv2_filters: The number of convolutional filters in the 2nd convolutional
      layer
    dense_size: The number of neurons in the last dense layer
    only_digits: If `True`, uses a final layer with 10 outputs, for use with the
      digits only EMNIST dataset. If `False`, uses 62 outputs for the larger
      dataset.

  Returns:
    An uncompiled `tf.keras.Model`.
  """
  data_format = 'channels_last'
  model = tf.keras.models.Sequential([
      tf.keras.layers.Conv2D(
          conv1_filters,
          kernel_size=(3, 3),
          activation='relu',
          data_format=data_format,
          input_shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(
          conv2_filters,
          kernel_size=(3, 3),
          activation='relu',
          data_format=data_format),
      tf.keras.layers.MaxPool2D(pool_size=(2, 2), data_format=data_format),
      tf.keras.layers.Dropout(0.25),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(dense_size, activation='relu'),
      tf.keras.layers.Dropout(0.5),
      tf.keras.layers.Dense(
          10 if only_digits else 62, activation=tf.nn.softmax),
  ])

  return model


def create_original_fedavg_cnn_model(
    conv1_filters=32,
    conv2_filters=64,
    dense_size=512,
    only_digits: bool = True) -> tf.keras.Model:
  """Create a convolutional network without dropout.

  This recreates the CNN model used in the original FedAvg paper,
  https://arxiv.org/abs/1602.05629. The number of parameters when
  `only_digits=True` is (1,663,370), which matches what is reported in the
  paper. When `only_digits=True`, the summary of returned model is
  ```
  Model: "sequential"
  _________________________________________________________________
  Layer (type)                 Output Shape              Param #
  =================================================================
  reshape (Reshape)            (None, 28, 28, 1)         0
  _________________________________________________________________
  conv2d (Conv2D)              (None, 28, 28, 32)        832
  _________________________________________________________________
  max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0
  _________________________________________________________________
  conv2d_1 (Conv2D)            (None, 14, 14, 64)        51264
  _________________________________________________________________
  max_pooling2d_1 (MaxPooling2 (None, 7, 7, 64)          0
  _________________________________________________________________
  flatten (Flatten)            (None, 3136)              0
  _________________________________________________________________
  dense (Dense)                (None, 512)               1606144
  _________________________________________________________________
  dense_1 (Dense)              (None, 10)                5130
  =================================================================
  Total params: 1,663,370
  Trainable params: 1,663,370
  Non-trainable params: 0
  ```
  For `only_digits=False`, the last dense layer is slightly larger.

  Args:
    conv1_filters: The number of convolutional filters in the 1st convolutional
      layer
    conv2_filters: The number of convolutional filters in the 2nd convolutional
      layer
    dense_size: The number of neurons in the last dense layer
    only_digits: If `True`, uses a final layer with 10 outputs, for use with the
      digits only EMNIST dataset. If `False`, uses 62 outputs for the larger
      dataset.

  Returns:
    An uncompiled `tf.keras.Model`.
  """
  data_format = 'channels_last'
  max_pool = functools.partial(
      tf.keras.layers.MaxPooling2D,
      pool_size=(2, 2),
      padding='same',
      data_format=data_format)
  conv2d = functools.partial(
      tf.keras.layers.Conv2D,
      kernel_size=5,
      padding='same',
      data_format=data_format,
      activation=tf.nn.relu)
  model = tf.keras.models.Sequential([
      conv2d(filters=conv1_filters, input_shape=(28, 28, 1)),
      max_pool(),
      conv2d(filters=conv2_filters),
      max_pool(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(dense_size, activation=tf.nn.relu),
      tf.keras.layers.Dense(
          10 if only_digits else 62, activation=tf.nn.softmax),
  ])
  return model


def make_big_and_small_emnist_cnn_model_fn(my_task,
                                           big_conv1_filters=32,
                                           big_conv2_filters=64,
                                           big_dense_size=512,
                                           small_conv1_filters=24,
                                           small_conv2_filters=48,
                                           small_dense_size=384):
  """Generates two model functions for a given task.

  Args:
    my_task: a tff.simulation.baselines.BaselineTask object
    big_conv1_filters: The number of convolutional filters in the 1st
      convolutional layer of the big model
    big_conv2_filters: The number of convolutional filters in the 2nd
      convolutional layer of the big model
    big_dense_size: The number of neurons in the last dense layer of the big
      model
    small_conv1_filters: The number of convolutional filters in the 1st
      convolutional layer of the small model
    small_conv2_filters: The number of convolutional filters in the 2nd
      convolutional layer of the small model
    small_dense_size: The number of neurons in the last dense layer of the small
      model

  Returns:
    Two model_fn functions
  """

  def big_model_fn():
    return tff.learning.from_keras_model(
        keras_model=create_original_fedavg_cnn_model(
            only_digits=False,
            conv1_filters=big_conv1_filters,
            conv2_filters=big_conv2_filters,
            dense_size=big_dense_size),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        input_spec=my_task.datasets.element_type_structure,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

  def small_model_fn():
    return tff.learning.from_keras_model(
        keras_model=create_original_fedavg_cnn_model(
            only_digits=False,
            conv1_filters=small_conv1_filters,
            conv2_filters=small_conv2_filters,
            dense_size=small_dense_size),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        input_spec=my_task.datasets.element_type_structure,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

  return big_model_fn, small_model_fn


def make_big_and_small_emnist_cnn_dropout_model_fn(my_task,
                                                   big_conv1_filters=32,
                                                   big_conv2_filters=64,
                                                   big_dense_size=128,
                                                   small_conv1_filters=24,
                                                   small_conv2_filters=48,
                                                   small_dense_size=96):
  """Generates two model functions for a given task.

  Args:
    my_task: a tff.simulation.baselines.BaselineTask object
    big_conv1_filters: The number of convolutional filters in the 1st
      convolutional layer of the big model
    big_conv2_filters: The number of convolutional filters in the 2nd
      convolutional layer of the big model
    big_dense_size: The number of neurons in the last dense layer of the big
      model
    small_conv1_filters: The number of convolutional filters in the 1st
      convolutional layer of the small model
    small_conv2_filters: The number of convolutional filters in the 2nd
      convolutional layer of the small model
    small_dense_size: The number of neurons in the last dense layer of the small
      model

  Returns:
    Two model_fn functions.
  """

  def big_model_fn():
    return tff.learning.from_keras_model(
        keras_model=create_conv_dropout_model(
            only_digits=False,
            conv1_filters=big_conv1_filters,
            conv2_filters=big_conv2_filters,
            dense_size=big_dense_size),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        input_spec=my_task.datasets.element_type_structure,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

  def small_model_fn():
    return tff.learning.from_keras_model(
        keras_model=create_conv_dropout_model(
            only_digits=False,
            conv1_filters=small_conv1_filters,
            conv2_filters=small_conv2_filters,
            dense_size=small_dense_size),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        input_spec=my_task.datasets.element_type_structure,
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

  return big_model_fn, small_model_fn
