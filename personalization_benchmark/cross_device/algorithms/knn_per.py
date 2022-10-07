# Copyright 2022, Google LLC.
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
"""Implemnetation of kNN-Per algorithm.

This algorithm is proposed by Marfoq et al., "Personalized Federated Learning
through Local Memorization", ICML 2022. https://arxiv.org/abs/2111.09360
"""

from collections.abc import Sequence
import functools
from typing import Optional

from absl import app
from absl import flags
import numpy as np
from sklearn import neighbors as sklearn_neighbors
import tensorflow as tf
import tensorflow_federated as tff

from personalization_benchmark.cross_device import constants
from personalization_benchmark.cross_device.datasets import emnist
from personalization_benchmark.cross_device.datasets import landmark
from personalization_benchmark.cross_device.datasets import stackoverflow
from personalization_benchmark.cross_device.datasets import ted_multi
from utils import keras_metrics
from utils.datasets import stackoverflow_word_prediction

_DATASET_NAME = flags.DEFINE_enum('dataset_name', None, constants.DATASET_NAMES,
                                  'Which dataset to use for experiments.')
_NUM_NEIGHBORS = flags.DEFINE_integer('num_neighbors', 10,
                                      'The value of k used in the kNN model.')
_PATH_TO_INITIAL_MODEL_WEIGHTS = flags.DEFINE_string(
    'path_to_initial_model_weights', None, 'Path to saved Keras model used for '
    'initialization. If None, use random initialization. See '
    '`checkpoint_util.py` for how to extract the model weights from a '
    'checkpoint created by our trainer.')
_VALID_CLIENTS_PER_EVALUATION = flags.DEFINE_integer(
    'valid_clients_per_evaluation', 100, 'Number of validation clients '
    'sampled to perform evaluation.')
_TEST_CLIENTS_PER_EVALUATION = flags.DEFINE_integer(
    'test_clients_per_evaluation', 100, 'Number of test clients sampled to '
    'perform evaluation.')
# Debugging flags
_USE_SYNTHETIC_DATA = flags.DEFINE_bool(
    'use_synthetic_data', False, 'Whether to use synthetic data. This should '
    'only be set to True for debugging purposes.')


def _compute_knn_softmax(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    eval_embeddings: np.ndarray,
    num_labels: int,
) -> np.ndarray:
  """Builds a kNN model, and use it to get a softmax output on the eval set.

  This function builds a *single* kNN model using the embeddings and labels in
  the training set, and evaluate the model on the eval embeddings. For each eval
  example, its prediction is given by the labels of its k nearest neighbors in
  the training set, and the probability is proportional to
  `exp(-d(eval_emb, neighbor_emb))`, where `d` is the Euclidean distance. This
  follows Equation (6) of Marfoq et al., "Personalized Federated Learning
  through Local Memorization", ICML 2022, https://arxiv.org/abs/2111.09360.

  Args:
    train_embeddings: An array of shape (num_train_examples, emb_size).
    train_labels: An array of shape (num_train_examples,).
    eval_embeddings: An array of shape (num_eval_examples, emb_size).
    num_labels: Number of total labels.

  Returns:
    An array of shape (num_eval_examples, num_labels), where each row sums to 1.
  """
  neigh = sklearn_neighbors.NearestNeighbors(n_neighbors=_NUM_NEIGHBORS.value)
  neigh.fit(train_embeddings)
  distances, neighbors = neigh.kneighbors(
      eval_embeddings, _NUM_NEIGHBORS.value, return_distance=True)
  num_eval_examples = eval_embeddings.shape[0]
  knn_softmax = np.zeros((num_eval_examples, num_labels))
  for eval_example_i, (dist_list,
                       nbr_indices) in enumerate(zip(distances, neighbors)):
    nbr_labels = train_labels[nbr_indices]
    for nbr_dist, nbr_label in zip(dist_list, nbr_labels):
      knn_softmax[eval_example_i, nbr_label] += np.exp(-nbr_dist)
    knn_softmax[eval_example_i, :] = knn_softmax[eval_example_i, :] / np.sum(
        knn_softmax[eval_example_i, :])  # Normalize each row to sum to 1.
  return knn_softmax


def knn_per_for_vision_data(personalization_data: tuple[tf.Tensor, tf.Tensor],
                            eval_data: tuple[tf.Tensor, tf.Tensor],
                            global_model: tf.keras.Model,
                            embedding_model: tf.keras.Model,
                            ensemble_coefficient: float) -> float:
  """Compute the accuracy of kNN-Per for a single client of a vision dataset.

  Each client's local dataset is split into two sets: a personalization set
  (used to train a kNN model) and an eval set (used to evaluate the personalized
  model). This function works for EMNIST and Landmarks datasets, which use a
  CNN model.

  Args:
    personalization_data: A tuple of input features and labels used to train a
      kNN model.
    eval_data: A tuple of eval input features and labels.
    global_model: The global model already trained with FedAvg. It will be
      ensembled with the local kNN model to form a personalized model.
    embedding_model: The model to extract the embeddings for the input features,
      used for building the kNN model. This is usually given by part of the
      global model, i.e., use the second to the last layer as the embedding.
    ensemble_coefficient: A value between [0, 1]. The personalized softmax is
      given by ensemble_coefficient * knn_softmax + (1-ensemble_coefficient) *
      global_softmax.

  Returns:
    The `SparseCategoricalAccuracy` evaluated on the eval set.
  """
  train_embeddings = embedding_model.predict(personalization_data[0], verbose=0)
  eval_embeddings = embedding_model.predict(eval_data[0], verbose=0)
  train_labels = personalization_data[1].numpy()
  global_softmax = global_model.predict(eval_data[0])
  assert np.isclose(np.sum(global_softmax[0, :]), 1.0), (
      'Expected each row of `global_softmax` sums to 1, but the first row sums '
      f'to {np.sum(global_softmax[0, :])}.')
  num_labels = global_softmax.shape[-1]
  knn_softmax = _compute_knn_softmax(train_embeddings, train_labels,
                                     eval_embeddings, num_labels)
  personalized_softmax = (
      ensemble_coefficient * knn_softmax +
      (1.0 - ensemble_coefficient) * global_softmax)
  metric = tf.keras.metrics.SparseCategoricalAccuracy()
  eval_labels = eval_data[1].numpy()
  metric.update_state(eval_labels, personalized_softmax)
  return metric.result().numpy()


def knn_per_for_language_data(
    personalization_data: tuple[tf.Tensor, tf.Tensor],
    eval_data: tuple[tf.Tensor, tf.Tensor],
    global_model: tf.keras.Model,
    embedding_model: tf.keras.Model,
    vocab_size: int,
    ensemble_coefficient: float,
) -> float:
  """Compute the accuracy of kNN-Per for a single client of a language dataset.

  Each client's local dataset is split into two sets: a personalization set
  (used to train a kNN model) and an eval set (used to evaluate the personalized
  model). This function works for StackOverflow (use LSTM model) and TedMulti
  (use Transformer model) datasets.

  Args:
    personalization_data: A tuple of input features and labels used to train a
      kNN model.
    eval_data: A tuple of eval input features and labels.
    global_model: The global model already trained with FedAvg. It will be
      ensembled with the local kNN model to form a personalized model.
    embedding_model: The model to extract the embeddings for the input features,
      used for building the kNN model. This is usually given by part of the
      global model, i.e., use the second to the last layer as the embedding.
    vocab_size: Size of the vocabulary used in the language model. Used for
      computing the metrics `MaskedCategoricalAccuracy` that masked out the
      special PAD and EOS tokens.
    ensemble_coefficient: A value between [0, 1]. The personalized softmax is
      given by ensemble_coefficient * knn_softmax + (1-ensemble_coefficient) *
      global_softmax.

  Returns:
    The `MaskedCategoricalAccuracy` evaluated on the eval set.

  Raises:
    ValueError: If the output dimension of `global_model` does not equal to
      `vocab_size` + number of special tokens such as EOS/OOV/BOS/PAD.
  """
  train_embeddings = embedding_model.predict(personalization_data[0], verbose=0)
  embedding_dim = train_embeddings.shape[-1]
  train_embeddings = train_embeddings.reshape((-1, embedding_dim))
  train_labels = personalization_data[1].numpy().reshape((-1,))
  eval_embeddings = embedding_model.predict(eval_data[0], verbose=0)
  eval_embeddings = eval_embeddings.reshape((-1, embedding_dim))
  eval_labels = eval_data[1].numpy().reshape((-1,))
  global_logits = global_model.predict(eval_data[0])
  special_tokens = stackoverflow_word_prediction.get_special_tokens(
      vocab_size, num_oov_buckets=1)
  num_special_tokens = special_tokens.get_number_of_special_tokens()
  output_dim = vocab_size + num_special_tokens
  if global_logits.shape[-1] != output_dim:
    raise ValueError(
        'Expected the `global_model` produces an output whose last dimension '
        f'is {output_dim}, which is `vocab_size` + {num_special_tokens}, found '
        f'an output dimension {global_logits.shape[-1]}. Please check that the '
        'provided `global_model` and the `vocab_size` are correct.')
  global_logits = global_logits.reshape((-1, output_dim))
  global_softmax = tf.nn.softmax(global_logits)
  knn_softmax = _compute_knn_softmax(train_embeddings, train_labels,
                                     eval_embeddings, output_dim)
  personalized_softmax = (
      ensemble_coefficient * knn_softmax +
      (1.0 - ensemble_coefficient) * global_softmax)
  metric = keras_metrics.MaskedCategoricalAccuracy(
      name='accuracy_without_out_of_vocab_or_end_of_sentence',
      masked_tokens=[special_tokens.pad, special_tokens.eos] +
      special_tokens.oov)
  metric.update_state(eval_labels, personalized_softmax)
  return metric.result().numpy()


def knn_per_avg_clients(federated_data: tff.simulation.datasets.ClientData,
                        split_data_fn: constants.SplitDataFnType,
                        sampled_client_ids: list[str],
                        ensemble_coefficient: float,
                        global_model: tf.keras.Model,
                        embedding_model: tf.keras.Model,
                        vocab_size: Optional[int] = None) -> list[float]:
  """Compute the per-client accuracy of kNN-Per for a list of clients."""
  if vocab_size is not None:
    single_client_knn_per = functools.partial(
        knn_per_for_language_data, vocab_size=vocab_size)
  else:
    single_client_knn_per = knn_per_for_vision_data
  client_accs = []
  for client_id in sampled_client_ids:
    client_data = federated_data.create_tf_dataset_for_client(client_id)
    client_data_after_split = split_data_fn(client_data)
    client_personalization_data = client_data_after_split[
        constants.PERSONALIZATION_DATA_KEY]
    client_eval_data = client_data_after_split[constants.TEST_DATA_KEY]
    # For all datasets we use, each client's local dataset has less than 2000
    # samples, so using 2000 as the batch size means taking all examples.
    personalization_data = iter(client_personalization_data.batch(2000)).next()
    eval_data = iter(client_eval_data.batch(2000)).next()
    client_accs.append(
        single_client_knn_per(
            personalization_data=personalization_data,
            eval_data=eval_data,
            global_model=global_model,
            embedding_model=embedding_model,
            ensemble_coefficient=ensemble_coefficient))
  return client_accs


def create_model_and_data(dataset_name: str, use_synthetic_data: bool) ->...:
  """Obtain the model fn and federated datasets given a dataset name."""
  # This `train_batch_size` is only used in training clients, not validation and
  # test clients, which are the ones we used to evaluation the personalization
  # performance. For validation and test clients, batching is applied after
  # splitting their local data into a personalization set and an eval set (i.e.,
  # inside `knn_per_avg_clients` above).
  unused_batch_size = 20
  if dataset_name == 'emnist':
    return emnist.create_model_and_data(
        num_local_epochs=1,
        train_batch_size=unused_batch_size,
        use_synthetic_data=use_synthetic_data)
  elif dataset_name == 'stackoverflow':
    return stackoverflow.create_model_and_data(
        num_local_epochs=1,
        train_batch_size=unused_batch_size,
        use_synthetic_data=use_synthetic_data)
  elif dataset_name == 'landmark':
    return landmark.create_model_and_data(
        num_local_epochs=1,
        train_batch_size=unused_batch_size,
        use_synthetic_data=use_synthetic_data)
  elif dataset_name == 'ted_multi':
    return ted_multi.create_model_and_data(
        num_local_epochs=1,
        train_batch_size=unused_batch_size,
        use_synthetic_data=use_synthetic_data)
  raise ValueError(f'Accepted dataset names: {constants.DATASET_NAMES}, but '
                   f'found {dataset_name}. Please provide a valid name.')


def embedding_layer_index(dataset_name: str) -> int:
  """Which layer output in the global model is used as the embedding."""
  if dataset_name == 'emnist':
    # Use the second to the last layer as the embedding. For EMNIST model, the
    # embedding dimension is 512.
    return -2
  elif dataset_name == 'stackoverflow':
    # For StackOverflow model, the output of the second layer is the output of
    # the LSTM layer, which has dimension 670.
    return 2
  elif dataset_name == 'landmark':
    # Use the second to the last layer as the embedding. For Landmarks model,
    # the embedding dimension is 1280.
    return -2
  elif dataset_name == 'ted_multi':
    # For TedMulti model, the output if the first layer is the output of
    # the transformer model, which has dimension 96.
    return 1
  raise ValueError(f'Accepted dataset names: {constants.DATASET_NAMES}, but '
                   f'found {dataset_name}. Please provide a valid name.')


def get_vocab_size(dataset_name: str) -> Optional[int]:
  """Vocabulary size used by the language model."""
  if dataset_name == 'stackoverflow':
    if _USE_SYNTHETIC_DATA.value:
      vocab_size = len(
          tff.simulation.datasets.stackoverflow.get_synthetic_word_counts())
    else:
      vocab_size = 10000
  elif dataset_name == 'ted_multi':
    if _USE_SYNTHETIC_DATA.value:
      vocab_size = len(ted_multi._synethetic_vocab())  # pylint:disable=protected-access
    else:
      vocab_size = 15814
  elif dataset_name in ['emnist', 'landmark']:
    vocab_size = None
  else:
    raise ValueError(f'Accepted dataset names: {constants.DATASET_NAMES}, but '
                     f'found {dataset_name}. Please provide a valid name.')
  return vocab_size


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  model_fn, datasets, _, split_data_fn, _ = create_model_and_data(
      _DATASET_NAME.value, _USE_SYNTHETIC_DATA.value)

  # Create the global model, and initialize it with the checkpoint (if given).
  if _PATH_TO_INITIAL_MODEL_WEIGHTS.value is not None:
    global_model = tf.keras.models.load_model(
        _PATH_TO_INITIAL_MODEL_WEIGHTS.value)
  else:
    global_model = model_fn()._keras_model  # pylint:disable=protected-access
  # Create the model to map input features to a embedding vector. The embeddings
  # will then be used by every client to train a local kNN model.
  layer_name = global_model.layers[embedding_layer_index(
      _DATASET_NAME.value)].name
  embedding_model = tf.keras.Model(
      inputs=global_model.input,
      outputs=global_model.get_layer(layer_name).output)
  vocab_size = get_vocab_size(_DATASET_NAME.value)
  valid_avg_accs = []
  test_avg_accs = []
  valid_client_data = datasets[constants.VALID_CLIENTS_KEY]
  test_client_data = datasets[constants.TEST_CLIENTS_KEY]
  sampled_valid_client_ids = np.random.choice(
      valid_client_data.client_ids, _VALID_CLIENTS_PER_EVALUATION.value)
  sampled_test_client_ids = np.random.choice(test_client_data.client_ids,
                                             _TEST_CLIENTS_PER_EVALUATION.value)
  # Tuning the ensemble coefficient in the range [0, 0.1, ..., 1.0].
  ensemble_coefficients = np.array(range(11)) * 0.1
  for coeff in ensemble_coefficients:
    current_valid_accs = knn_per_avg_clients(valid_client_data, split_data_fn,
                                             sampled_valid_client_ids, coeff,
                                             global_model, embedding_model,
                                             vocab_size)
    valid_avg_accs.append(np.mean(current_valid_accs))
    current_test_accs = knn_per_avg_clients(test_client_data, split_data_fn,
                                            sampled_test_client_ids, coeff,
                                            global_model, embedding_model,
                                            vocab_size)
    test_avg_accs.append(np.mean(current_test_accs))

  best_coeff = ensemble_coefficients[np.argmax(valid_avg_accs)]
  avg_test_acc_at_best_coeff = test_avg_accs[np.argmax(valid_avg_accs)]
  print(f'The best ensemble coefficient is {best_coeff}. The average test '
        f'accuracy at the best coefficient is {avg_test_acc_at_best_coeff}.')


if __name__ == '__main__':
  app.run(main)
