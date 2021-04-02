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
"""Centralized experiments on the Stackoverflow datasets."""

from typing import Any, Mapping, Optional

import tensorflow as tf

from fedopt_guide.stackoverflow_transformer import transformer_models
from optimization.shared import keras_metrics
from utils import centralized_training_loop
from utils.datasets import stackoverflow_word_prediction


def run_centralized(optimizer: tf.keras.optimizers.Optimizer,
                    num_epochs: int,
                    batch_size: int,
                    decay_epochs: Optional[int] = None,
                    lr_decay: Optional[float] = None,
                    vocab_size: int = 10000,
                    num_oov_buckets: int = 1,
                    dim_embed: int = 96,
                    dim_model: int = 512,
                    dim_hidden: int = 2048,
                    num_heads: int = 8,
                    num_layers: int = 1,
                    max_position_encoding: int = 1000,
                    dropout: float = 0.1,
                    num_validation_examples: int = 10000,
                    sequence_length: int = 20,
                    experiment_name: str = 'centralized_stackoverflow',
                    root_output_dir: str = '/tmp/fedopt_guide',
                    hparams_dict: Optional[Mapping[str, Any]] = None,
                    max_batches: Optional[int] = None):
  """Trains an Transformer on the Stack Overflow next word prediction task.

  Args:
    optimizer: A `tf.keras.optimizers.Optimizer` used to perform training.
    num_epochs: The number of training epochs.
    batch_size: The batch size, used for train, validation, and test.
    decay_epochs: The number of epochs of training before decaying the learning
      rate. If None, no decay occurs.
    lr_decay: The amount to decay the learning rate by after `decay_epochs`
      training epochs have occurred.
    vocab_size: Vocab size for normal tokens.
    num_oov_buckets: Number of out of vocabulary buckets.
    dim_embed: Dimension of the token embeddings.
    dim_model: Dimension of features of MultiHeadAttention layers.
    dim_hidden: Dimension of hidden layers of the FFN.
    num_heads: Number of attention heads.
    num_layers: Number of Transformer blocks.
    max_position_encoding: Maximum number of positions for position embeddings.
    dropout: Dropout rate.
    num_validation_examples: The number of test examples to use for validation.
    sequence_length: The maximum number of words to take for each sequence.
    experiment_name: The name of the experiment. Part of the output directory.
    root_output_dir: The top-level output directory for experiment runs. The
      `experiment_name` argument will be appended, and the directory will
      contain tensorboard logs, metrics written as CSVs, and a CSV of
      hyperparameter choices (if `hparams_dict` is used).
    hparams_dict: A mapping with string keys representing the hyperparameters
      and their values. If not None, this is written to CSV.
    max_batches: If set to a positive integer, datasets are capped to at most
      that many batches. If set to None or a nonpositive integer, the full
      datasets are used.
  """

  train_dataset, validation_dataset, test_dataset = stackoverflow_word_prediction.get_centralized_datasets(
      vocab_size,
      sequence_length,
      train_batch_size=batch_size,
      num_validation_examples=num_validation_examples,
      num_oov_buckets=num_oov_buckets,
  )

  if max_batches and max_batches >= 1:
    train_dataset = train_dataset.take(max_batches)
    validation_dataset = validation_dataset.take(max_batches)
    test_dataset = test_dataset.take(max_batches)

  model = transformer_models.create_transformer_lm(
      vocab_size=vocab_size,
      num_oov_buckets=num_oov_buckets,
      dim_embed=dim_embed,
      dim_model=dim_model,
      dim_hidden=dim_hidden,
      num_heads=num_heads,
      num_layers=num_layers,
      max_position_encoding=max_position_encoding,
      dropout=dropout,
      name='stackoverflow-transformer')

  special_tokens = stackoverflow_word_prediction.get_special_tokens(
      vocab_size=vocab_size, num_oov_buckets=num_oov_buckets)
  pad_token = special_tokens.pad
  oov_tokens = special_tokens.oov
  eos_token = special_tokens.eos

  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=optimizer,
      metrics=[
          keras_metrics.MaskedCategoricalAccuracy(
              name='accuracy_with_oov', masked_tokens=[pad_token]),
          keras_metrics.MaskedCategoricalAccuracy(
              name='accuracy_no_oov', masked_tokens=[pad_token] + oov_tokens),
          keras_metrics.MaskedCategoricalAccuracy(
              name='accuracy_no_oov_or_eos',
              masked_tokens=[pad_token, eos_token] + oov_tokens),
      ])

  centralized_training_loop.run(
      keras_model=model,
      train_dataset=train_dataset,
      validation_dataset=validation_dataset,
      test_dataset=test_dataset,
      experiment_name=experiment_name,
      root_output_dir=root_output_dir,
      num_epochs=num_epochs,
      hparams_dict=hparams_dict,
      decay_epochs=decay_epochs,
      lr_decay=lr_decay)
