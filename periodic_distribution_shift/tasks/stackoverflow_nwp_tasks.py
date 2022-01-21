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
"""Library for creating periodic distribution shift tasks on Stack Overflow."""

# TODO(b/193904908): add unit tests.

from typing import Optional

import tensorflow as tf
import tensorflow_federated as tff

from periodic_distribution_shift.datasets import stackoverflow_nwp_preprocessing as word_prediction_preprocessing
from periodic_distribution_shift.models import keras_utils_dual_branch_kmeans_lm
from periodic_distribution_shift.tasks import dist_shift_task
from periodic_distribution_shift.tasks import dist_shift_task_data


constants = tff.simulation.baselines.stackoverflow.constants


def filter_qa_fn(q_or_a):

  def filter_(sample):
    return tf.math.equal(sample['type'], tf.constant(q_or_a))

  return filter_


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
  group = tf.keras.layers.Input(shape=(None,), name='group')

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

  return tf.keras.Model(inputs=[inputs, group], outputs=logits)


def create_recurrent_merge_branch_model(
    vocab_size: int,
    embedding_size: int = 96,
    num_lstm_layers: int = 1,
    lstm_size: int = 670,
    shared_embedding: bool = False) -> tf.keras.Model:
  """Constructs a recurrent model with an initial embeding layer.

  The output is
  the average of the two branches, to obtain the same capacity as the dual
  branch model.

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
  group = tf.keras.layers.Input(shape=(None,), name='group')

  input_embedding = tf.keras.layers.Embedding(
      input_dim=vocab_size, output_dim=embedding_size, mask_zero=True)
  embedded = input_embedding(inputs)
  projected = embedded

  embedding_out_proj_list = [
      tf.keras.layers.Dense(embedding_size, name=f'dist_head_{i}')
      for i in range(2)
  ]

  for n in range(num_lstm_layers):
    layer = tf.keras.layers.LSTM(lstm_size, return_sequences=True)
    processed = layer(projected)
    if n == num_lstm_layers - 1:
      # if output_all_logits:
      projected_list = []
      for branch in embedding_out_proj_list:
        projected_list.append(branch(processed))
    else:
      projected = tf.keras.layers.Dense(embedding_size)(processed)

  if shared_embedding:
    transposed_embedding = TransposableEmbedding(input_embedding)
    final_proj = transposed_embedding
  else:
    final_proj = tf.keras.layers.Dense(vocab_size, activation=None)

  logits = (final_proj(projected_list[0]) + final_proj(projected_list[1])) * 0.5

  return tf.keras.Model(inputs=[inputs, group], outputs=logits)


def create_recurrent_dual_branch_model(
    vocab_size: int,
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

  tokens = tf.keras.layers.Input(shape=(None,), name='tokens')
  group = tf.keras.layers.Input(shape=(None,), name='group')
  input_embedding = tf.keras.layers.Embedding(
      input_dim=vocab_size, output_dim=embedding_size, mask_zero=True)
  embedded = input_embedding(tokens)

  projected = embedded

  embedding_out_proj_list = [
      tf.keras.layers.Dense(embedding_size, name=f'dist_head_{i}')
      for i in range(2)
  ]

  for n in range(num_lstm_layers):
    layer = tf.keras.layers.LSTM(lstm_size, return_sequences=True)
    processed = layer(projected)
    if n == num_lstm_layers - 1:
      # if output_all_logits:
      projected_list = []
      for branch in embedding_out_proj_list:
        projected_list.append(branch(processed))
    else:
      projected = tf.keras.layers.Dense(embedding_size)(processed)

  # whether to output all logis from all heads for later processing
  if shared_embedding:
    final_proj = TransposableEmbedding(input_embedding)
  else:
    final_proj = tf.keras.layers.Dense(vocab_size, activation=None)
  rets = [final_proj(projected) for projected in projected_list]

  # Get the features for K-means clustering
  nopad_mask = tf.expand_dims(
      tf.cast(tf.math.not_equal(tokens, 0), tf.float32), axis=-1)
  n_valid = tf.reduce_sum(nopad_mask, axis=1)
  kmeans_feat_list = [
      tf.reduce_sum(nopad_mask * pj, axis=1) / (n_valid + 1e-7)
      for pj in projected_list
  ]
  rets.append(tf.concat(kmeans_feat_list, axis=1))
  return tf.keras.Model(inputs=[tokens, group], outputs=rets)


def create_word_prediction_task_from_datasets(
    train_client_spec: tff.simulation.baselines.ClientSpec,
    eval_client_spec: Optional[tff.simulation.baselines.ClientSpec],
    sequence_length: int,
    vocab_size: int,
    num_out_of_vocab_buckets: int,
    train_data: tff.simulation.datasets.ClientData,
    test_data: tff.simulation.datasets.ClientData,
    validation_data: tff.simulation.datasets.ClientData,
    model_type: str = 'single_branch',
    shared_embedding: bool = False,
    num_validation_examples: int = 2000,
    aggregated_kmeans: bool = False,
    label_smooth_w: float = 0.,
    label_smooth_eps: float = 0.0,
    batch_majority_voting: bool = False,
    use_mixed: bool = False,
) -> dist_shift_task.DistShiftTask:
  """Creates a baseline task for next-word prediction on Stack Overflow.

  The goal of the task is to take `sequence_length` words from a post and
  predict the next word. Here, all posts are drawn from the Stack Overflow
  forum, and a client corresponds to a user.

  Args:
    train_client_spec: A `tff.simulation.baselines.ClientSpec` specifying how to
      preprocess train client data.
    eval_client_spec: An optional `tff.simulation.baselines.ClientSpec`
      specifying how to preprocess evaluation client data. If set to `None`, the
      evaluation datasets will use a batch size of 64 with no extra
      preprocessing.
    sequence_length: A positive integer dictating the length of each word
      sequence in a client's dataset. By default, this is set to
      `tff.simulation.baselines.stackoverflow.DEFAULT_SEQUENCE_LENGTH`.
    vocab_size: Integer dictating the number of most frequent words in the
      entire corpus to use for the task's vocabulary. By default, this is set to
      `tff.simulation.baselines.stackoverflow.DEFAULT_WORD_VOCAB_SIZE`.
    num_out_of_vocab_buckets: The number of out-of-vocabulary buckets to use.
    train_data: A `tff.simulation.datasets.ClientData` used for training.
    test_data: A `tff.simulation.datasets.ClientData` used for testing.
    validation_data: A `tff.simulation.datasets.ClientData` used for validation.
    model_type: Model type, only for the baseline. `single_branch` to train
      a single branch model. Otherwise, we take the average of two branches.
    shared_embedding: Whether to share the word embedding with the prediction
      layer.
    num_validation_examples: Max number of validation samples.
    aggregated_kmeans: Whether to use aggregated k-means. If set to `True`, we
      will create a dual branch model, and use k-means based on the feautres to
      select branches in the forward pass.
    label_smooth_w: Weight of label smoothing regularization on the unselected
      branch. Only effective when `aggregated_kmeans = True`.
    label_smooth_eps: Epsilon of the label smoothing for the unselected branch.
      The value should be within 0 to 1, where 1 enforces the prediction to be
      uniform on all labels, and 0 falls back to cross entropy loss on one-hot
      label. The label smoothing regularization is defined as
      `L_{CE}(g(x), (1 - epsilon) * y + epsilon * 1/n)`, where L_{CE} is the
      cross entropy loss, g(x) is the prediction, epsilon represents the
      smoothness. Only effective when `aggregated_kmeans = True`.
    batch_majority_voting: Whether to use batch-wise majority voting to select
      the branch during test time. If set to True, we select the branch
      according to the majority within the minibatch during inference.
      Otherwise, we select the branch for each sample. Only effective when
      `aggregated_kmeans = True`.
    use_mixed: Whether to use the weighted prediction of the two branches.
      Weights are defined as the distance to the k-means cluster centers.

  Returns:
    A `dist_shift_task.DistShiftTask`.
  """
  if sequence_length < 1:
    raise ValueError('sequence_length must be a positive integer')
  if vocab_size < 1:
    raise ValueError('vocab_size must be a positive integer')
  if num_out_of_vocab_buckets < 1:
    raise ValueError('num_out_of_vocab_buckets must be a positive integer')

  vocab = list(
      tff.simulation.datasets.stackoverflow.load_word_counts(
          vocab_size=vocab_size).keys())

  if eval_client_spec is None:
    eval_client_spec = tff.simulation.baselines.ClientSpec(
        num_epochs=1, batch_size=512, shuffle_buffer_size=1)

  train_preprocess_fn = word_prediction_preprocessing.create_preprocess_fn(
      train_client_spec,
      vocab,
      sequence_length=sequence_length,
      num_out_of_vocab_buckets=num_out_of_vocab_buckets)
  eval_preprocess_fn = word_prediction_preprocessing.create_preprocess_fn(
      eval_client_spec,
      vocab,
      sequence_length=sequence_length,
      num_out_of_vocab_buckets=num_out_of_vocab_buckets)

  full_validation_set = validation_data.create_tf_dataset_from_all_clients()
  if num_validation_examples is not None:
    full_validation_set = full_validation_set.take(num_validation_examples)
  question_val_set = full_validation_set.filter(filter_qa_fn('question'))
  answer_val_set = full_validation_set.filter(filter_qa_fn('answer'))
  dataset_dict = {}
  dataset_dict['full'] = eval_preprocess_fn(full_validation_set)
  dataset_dict['question'] = eval_preprocess_fn(question_val_set)
  dataset_dict['answer'] = eval_preprocess_fn(answer_val_set)

  task_datasets = dist_shift_task_data.DistShiftDatasets(
      train_data=train_data,
      test_data=test_data,
      validation_data_dict=dataset_dict,
      train_preprocess_fn=train_preprocess_fn,
      eval_preprocess_fn=eval_preprocess_fn)

  special_tokens = word_prediction_preprocessing.get_special_tokens(
      vocab_size, num_out_of_vocab_buckets=num_out_of_vocab_buckets)
  pad_token = special_tokens.padding
  oov_tokens = special_tokens.out_of_vocab
  eos_token = special_tokens.end_of_sentence

  def metrics_builder():
    return [
        tff.simulation.baselines.keras_metrics.NumTokensCounter(
            masked_tokens=[pad_token]),
        tff.simulation.baselines.keras_metrics.MaskedCategoricalAccuracy(
            name='accuracy', masked_tokens=[pad_token]),
        tff.simulation.baselines.keras_metrics.MaskedCategoricalAccuracy(
            name='accuracy_without_out_of_vocab',
            masked_tokens=[pad_token] + oov_tokens),
        # Notice that the beginning of sentence token never appears in the
        # ground truth label.
        tff.simulation.baselines.keras_metrics.MaskedCategoricalAccuracy(
            name='accuracy_without_out_of_vocab_or_end_of_sentence',
            masked_tokens=[pad_token, eos_token] + oov_tokens),
    ]
  # The total vocabulary size is the number of words in the vocabulary, plus
  # the number of out-of-vocabulary tokens, plus three tokens used for
  # padding, beginning of sentence and end of sentence.
  extended_vocab_size = (
      vocab_size + special_tokens.get_number_of_special_tokens())

  def model_fn() -> tff.learning.Model:
    if aggregated_kmeans:
      return keras_utils_dual_branch_kmeans_lm.from_keras_model(
          keras_model=create_recurrent_dual_branch_model(
              vocab_size=extended_vocab_size,
              shared_embedding=shared_embedding),
          loss=tf.keras.losses.SparseCategoricalCrossentropy(
              from_logits=True, reduction='none'),
          input_spec=task_datasets.element_type_structure,
          metrics=metrics_builder(),
          from_logits=True,
          uniform_reg=label_smooth_w,
          label_smoothing=label_smooth_eps,
          batch_majority_voting=batch_majority_voting,
          use_mixed=use_mixed)
    else:
      if model_type == 'single_branch':
        model_fn = create_recurrent_model
      else:
        model_fn = create_recurrent_merge_branch_model
      return tff.learning.keras_utils.from_keras_model(
          keras_model=model_fn(vocab_size=extended_vocab_size),
          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
          input_spec=task_datasets.element_type_structure,
          metrics=metrics_builder())

  return dist_shift_task.DistShiftTask(task_datasets, model_fn)


def cid_qa_filter_fn(cid):
  mid_cid = tf.constant(1816184)

  def filter_(sample):
    if tf.math.less(tf.strings.to_number(cid, mid_cid.dtype), mid_cid):
      return tf.math.equal(sample['type'], tf.constant('question'))
    else:
      return tf.math.equal(sample['type'], tf.constant('answer'))

  return filter_


def create_word_prediction_task(
    train_client_spec: tff.simulation.baselines.ClientSpec,
    eval_client_spec: Optional[tff.simulation.baselines.ClientSpec] = None,
    sequence_length: int = constants.DEFAULT_SEQUENCE_LENGTH,
    vocab_size: int = constants.DEFAULT_WORD_VOCAB_SIZE,
    num_out_of_vocab_buckets: int = 1,
    cache_dir: Optional[str] = None,
    use_synthetic_data: bool = False,
    model_type='share_second',
    shared_embedding: bool = False,
    num_val_samples: int = 2000,
    aggregated_kmeans: bool = False,
    label_smooth_w: float = 0.,
    label_smooth_eps: float = 1.0,
    batch_majority_voting: bool = False,
    use_mixed: bool = False,
) -> dist_shift_task.DistShiftTask:
  """Creates a baseline task for next-word prediction on Stack Overflow.

  The goal of the task is to take `sequence_length` words from a post and
  predict the next word. Here, all posts are drawn from the Stack Overflow
  forum, and a client corresponds to a user.

  Args:
    train_client_spec: A `tff.simulation.baselines.ClientSpec` specifying how to
      preprocess train client data.
    eval_client_spec: An optional `tff.simulation.baselines.ClientSpec`
      specifying how to preprocess evaluation client data. If set to `None`, the
      evaluation datasets will use a batch size of 64 with no extra
      preprocessing.
    sequence_length: A positive integer dictating the length of each word
      sequence in a client's dataset. By default, this is set to
      `tff.simulation.baselines.stackoverflow.DEFAULT_SEQUENCE_LENGTH`.
    vocab_size: Integer dictating the number of most frequent words in the
      entire corpus to use for the task's vocabulary. By default, this is set to
      `tff.simulation.baselines.stackoverflow.DEFAULT_WORD_VOCAB_SIZE`.
    num_out_of_vocab_buckets: The number of out-of-vocabulary buckets to use.
    cache_dir: An optional directory to cache the downloadeded datasets. If
      `None`, they will be cached to `~/.tff/`.
    use_synthetic_data: A boolean indicating whether to use synthetic Stack
      Overflow data. This option should only be used for testing purposes, in
      order to avoid downloading the entire Stack Overflow dataset.
    model_type: Model type, only for the baseline. `single_branch` to train
      a single branch model. Otherwise, we take the average of two branches.
    shared_embedding: Whether to share the word embedding with the prediction
      layer.
    num_val_samples: Max number of validation samples.
    aggregated_kmeans: Whether to use aggregated k-means. If set to `True`, we
      will create a dual branch model, and use k-means based on the feautres to
      select branches in the forward pass.
    label_smooth_w: Weight of label smoothing regularization on the unselected
      branch. Only effective when `aggregated_kmeans = True`.
    label_smooth_eps: Epsilon of the label smoothing for the unselected branch.
      The value should be within 0 to 1, where 1 enforces the prediction to be
      uniform on all labels, and 0 falls back to cross entropy loss on one-hot
      label. The label smoothing regularization is defined as
      `L_{CE}(g(x), (1 - epsilon) * y + epsilon * 1/n)`, where L_{CE} is the
      cross entropy loss, g(x) is the prediction, epsilon represents the
      smoothness. Only effective when `aggregated_kmeans = True`.
    batch_majority_voting: Whether to use batch-wise majority voting to select
      the branch during test time. If set to True, we select the branch
      according to the majority within the minibatch during inference.
      Otherwise, we select the branch for each sample. Only effective when
      `aggregated_kmeans = True`.
    use_mixed: Whether to use the weighted prediction of the two branches.
      Weights are defined as the distance to the k-means cluster centers.

  Returns:
    A `dist_shift_task.DistShiftTask`.
  """
  if use_synthetic_data:
    synthetic_data = tff.simulation.datasets.stackoverflow.get_synthetic()
    stackoverflow_train = synthetic_data
    stackoverflow_validation = synthetic_data
    stackoverflow_test = synthetic_data
  else:
    stackoverflow_train, stackoverflow_validation, stackoverflow_test = (
        tff.simulation.datasets.stackoverflow.load_data(cache_dir=cache_dir))

  return create_word_prediction_task_from_datasets(
      train_client_spec,
      eval_client_spec,
      sequence_length,
      vocab_size,
      num_out_of_vocab_buckets,
      stackoverflow_train,
      stackoverflow_test,
      stackoverflow_validation,
      model_type=model_type,
      num_validation_examples=num_val_samples,
      shared_embedding=shared_embedding,
      aggregated_kmeans=aggregated_kmeans,
      label_smooth_w=label_smooth_w,
      label_smooth_eps=label_smooth_eps,
      batch_majority_voting=batch_majority_voting,
      use_mixed=use_mixed,
  )
