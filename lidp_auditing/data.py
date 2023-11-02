# Copyright 2023, Google LLC.
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
"""Utilities to load the data.

This file enables the loading of the following datasets:
* FashionMNIST
* Purchase-100

Our dataset iterator returns a triplet (x, y, z):
  * x: the input features (image or otherwise)
  * y: the output label for our classification problem
  * z: an integer determining the canary status (details below)

A canary status of z < 0 denotes a non-canary. For z >= 0, the canary
status depends also on the type of poisoning:
  * static data canary: z = i denotes the i^th singular direction.
    Since we already modify the dataset upfront, the value z >= 0 has no
      significance in practice. We choose i randomly between `min_dimension`
      and `max_dimension`, which are passed as inputs.
  * random gradient canary: z = i denotes the random seed used to generate
    the random gradient direction used as a canary. This modification is
    also made on the fly.
      - NOTE: pass min_dimension = 0 and max_dimension as a very large number
        for random gradient canary. The "dimension" will simply be used as
        a seed to generate the random gradient.
  * no canary: no data points are modified
    - some training points are designated as canaries
    - test_canaries = some test points (for membership inference-style tests)
"""

from absl import logging
import numpy as np
import pandas as pd
import sklearn.decomposition
import tensorflow as tf
import tensorflow_datasets as tfds

from lidp_auditing import constants


def get_datasets(
    dataset_name: str,
    canary_type: str,
    num_canaries: int,
    num_canaries_test: int,
    seed: int,
    canary_data_scale: float,
    min_dimension: int,
    max_dimension: int,
    canary_class: int | None = None,
    csv_data_path: str | None = None,
    normalize: bool = True,
    validation_mode: bool = False,
) -> tuple[
    int,
    tuple[
        tf.data.Dataset,
        tf.data.Dataset,
        tf.data.Dataset | None,
        tf.data.Dataset | None,
    ],
]:
  """Return a tuple of training and test, both regular and canaries.

  Args:
    dataset_name: name of the dataset: fashion_mnist, cifar10, purchase.
    canary_type: type of canary: static_data, adaptive_data, random_gradient.
    num_canaries: number of canaries to introduce in the training.
    num_canaries_test: number of canaries to introduce for testing only.
    seed: random seed to generate the canaries.
    canary_data_scale: scaling constant for static data canaries.
    min_dimension: smallest dimension to choose for static/adaptive data canary.
    max_dimension: largest dimension to choose for static/adaptive data canary.
      Samples are produced in {min_dimension, ..., max_dimension-1}.
    canary_class: What class to assign for the canaries.
    csv_data_path: The path to load the data from using pd.read_csv
    normalize: if True, normalize the features to zero mean and unit variance.
    validation_mode: if True, use a part of the training data for evaluation. If
      using random gradient canaries, the `z` component of the datasets is
      actually a random seed. This random seed is to be used to produce the
      (random gradient poison) with an appropriate norm.

  Returns:
    a tuple (dataset_size,
    (train_data, test_data, canary_data, canary_data_test)).

  Raises:
    RuntimeError: when dataset is not known.
  """
  args = dict(
      canary_type=canary_type,
      num_canaries=num_canaries,
      num_canaries_test=num_canaries_test,
      seed=seed,
      canary_data_scale=canary_data_scale,
      min_dimension=min_dimension,
      max_dimension=max_dimension,
      normalize=normalize,
      validation_mode=validation_mode,
  )
  if canary_class is not None:
    args['canary_class'] = canary_class
  if dataset_name == constants.FASHION_MNIST_DATASET:
    return load_fashion_mnist(**args)
  elif dataset_name == constants.PURCHASE_DATASET:
    args['csv_data_path'] = csv_data_path
    return load_purchase100(**args)
  else:
    raise RuntimeError('Unknown dataset %s' % (dataset_name,))


def load_fashion_mnist(
    canary_type: str,
    num_canaries: int,
    num_canaries_test: int,
    seed: int,
    canary_data_scale: float = 1.0,
    min_dimension: int = 600,
    max_dimension: int = 784,
    max_num_examples: int = 60000,
    canary_class: int = 5,
    normalize: bool = True,
    validation_mode: bool = False,
    synthetic: bool = False,
) -> tuple[
    int,
    tuple[
        tf.data.Dataset,
        tf.data.Dataset,
        tf.data.Dataset | None,
        tf.data.Dataset | None,
    ],
]:
  """Generate data for FashionMNIST."""
  if normalize:
    ds_mean, ds_std = 0.2860405969887955, 0.3530242445149223  # precomputed
  else:
    ds_mean, ds_std = 0.0, 1.0
  if synthetic:
    # Return synthetic data of the same shape and datatypes for testing.
    dataset = dict(
        train=tf.data.Dataset.from_tensor_slices((
            tf.cast(
                tf.random.uniform(
                    shape=(60000, 28, 28, 1), minval=0, maxval=255
                ),
                tf.uint8,
            ),
            tf.random.uniform(
                shape=(60000,), minval=0, maxval=10, dtype=tf.int64
            ),
        )),
        test=tf.data.Dataset.from_tensor_slices((
            tf.cast(
                tf.random.uniform(
                    shape=(10000, 28, 28, 1), minval=0, maxval=255
                ),
                tf.uint8,
            ),
            tf.random.uniform(
                shape=(10000,), minval=0, maxval=10, dtype=tf.int64
            ),
        )),
    )
  else:
    dataset = tfds.load('fashion_mnist', as_supervised=True)

  # Fix the testing dataset.
  if validation_mode:
    train_data = next(
        iter(
            dataset['train']
            .take(max_num_examples)
            .shuffle(max_num_examples, seed=seed + 100)
            .batch(max_num_examples)
        ),
        None,
    )
    x_train, y_train = train_data
    threshold = len(y_train) // 5  # Take 20% of the dataset for validation
    x_test, y_test = x_train[:threshold], y_train[:threshold]
    train_data = (x_train[threshold:], y_train[threshold:])
    logging.warn('Using validation mode!')
  else:
    train_data = next(
        iter(dataset['train'].take(max_num_examples).batch(max_num_examples)),
        None,
    )
    x_test, y_test = next(iter(dataset['test'].batch(max_num_examples)), None)
  x_test = (tf.cast(x_test, tf.float32) / 255 - ds_mean) / ds_std

  # Design the training dataset.
  if train_data is None:
    raise ValueError('No data found in FashionMNIST')
  x_train = tf.cast(train_data[0], tf.float32) / 255  # (n, 28, 28, 1)
  x_train = (x_train - ds_mean) / ds_std
  y_train = train_data[1]  # (n,)
  return create_dataset_from_tensors(
      x_train,
      y_train,
      x_test,
      y_test,
      canary_type,
      num_canaries,
      num_canaries_test,
      seed,
      canary_data_scale,
      min_dimension,
      max_dimension,
      canary_class,
  )


def load_purchase100(
    csv_data_path: str | None,
    canary_type: str,
    num_canaries: int,
    num_canaries_test: int,
    seed: int,
    canary_data_scale: float = 1.0,
    min_dimension: int = 400,
    max_dimension: int = 600,
    max_num_examples: int = 20000,
    canary_class: int = 5,
    normalize: bool = False,
    validation_mode: bool = False,
) -> tuple[
    int,
    tuple[
        tf.data.Dataset,
        tf.data.Dataset,
        tf.data.Dataset | None,
        tf.data.Dataset | None,
    ],
]:
  """Load the preprocessed Purchase-100 dataset."""
  del normalize
  logging.warning('Loading Purchase Data!')
  if csv_data_path is None:
    raise RuntimeError('Path to input file needed.')
  df = pd.read_csv(csv_data_path)
  y = df['label'].to_numpy()
  x = df.drop(columns=['label']).to_numpy().astype(np.float32)
  # Split train and test
  rng = np.random.RandomState(0)
  idxs = rng.choice(x.shape[0], size=max_num_examples, replace=False)  # train
  idxs2 = np.asarray(list(set(range(x.shape[0])) - set(idxs.tolist())))  # test
  x1, y1 = x[idxs], y[idxs]
  x2, y2 = x[idxs2], y[idxs2]
  if validation_mode:  # Split the train set into train and val
    n_train = int(x1.shape[0] * 0.8)  # Keep 80% of the data
    idxs = rng.choice(x1.shape[0], size=n_train, replace=False)
    idxs2 = np.asarray(list(set(range(x2.shape[0])) - set(idxs.tolist())))
    x2, y2 = x1[idxs2], y1[idxs2]
    x1, y1 = x1[idxs], y1[idxs]

  x1, y1 = tf.convert_to_tensor(x1), tf.convert_to_tensor(y1)
  x2, y2 = tf.convert_to_tensor(x2), tf.convert_to_tensor(y2)
  return create_dataset_from_tensors(
      x1,
      y1,
      x2,
      y2,
      canary_type,
      num_canaries,
      num_canaries_test,
      seed,
      canary_data_scale,
      min_dimension,
      max_dimension,
      canary_class,
  )


def create_dataset_from_tensors(
    x_train: tf.Tensor,
    y_train: tf.Tensor,
    x_test: tf.Tensor,
    y_test: tf.Tensor,
    canary_type: str,
    num_canaries: int,
    num_canaries_test: int,
    seed: int,
    canary_data_scale: float,
    min_dimension: int,
    max_dimension: int,
    canary_class: int,
) -> tuple[
    int,
    tuple[
        tf.data.Dataset,
        tf.data.Dataset,
        tf.data.Dataset | None,
        tf.data.Dataset | None,
    ],
]:
  """Create datasets and add canaries."""
  assert canary_type in constants.CANARY_TYPES
  original_shape = x_train.shape[1:]
  z_test = tf.constant(-1, shape=x_test.shape[0], dtype=tf.int64)
  test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test, z_test))
  out = _generate_canary_info(
      x_train.shape[0],
      num_canaries,
      num_canaries_test,
      seed,
      min_dimension,
      max_dimension,
  )
  z_train = out[0]  # z_train: shape (n,)
  canary_idxs, canary_principal_components = out[1:3]
  canary_idxs_test, canary_principal_components_test = out[3:]

  if canary_type == constants.STATIC_DATA_CANARY:
    # Perform static posioning upfront.
    # Ensure canaries are exchangeable with:
    #   * x = randomly sampled principal component between min_dim and max_dim
    #   * y = canary_class
    x_train_numpy = x_train.numpy().reshape(x_train.shape[0], -1)
    y_train_numpy = y_train.numpy()
    canary_data, canary_data_test = _generate_static_data_poisoning(
        x_train_numpy,
        canary_principal_components,
        canary_principal_components_test,
        canary_data_scale,
    )
    x_train_numpy[canary_idxs] = canary_data  # change canary x
    y_train_numpy[canary_idxs] = canary_class  # change canary y
    x_train = tf.convert_to_tensor(
        np.ascontiguousarray(x_train_numpy.reshape(-1, *original_shape))
    )
    y_train = tf.convert_to_tensor(np.ascontiguousarray(y_train_numpy))
    # z_train has already been set above. No changes necessary here.
    x_canary_test = tf.convert_to_tensor(
        np.ascontiguousarray(canary_data_test.reshape(-1, *original_shape))
    )
    y_canary_test = tf.convert_to_tensor(
        canary_class * np.ones(x_canary_test.shape[0], dtype=np.int64)
    )
  elif canary_type == constants.NO_CANARY:
    # Sample some test points as canaries:
    # z is to be ignored for these points.
    rng = np.random.RandomState(seed + 1)
    canary_idxs_test = rng.choice(
        x_test.shape[0], size=num_canaries_test, replace=False
    )
    x_canary_test = _extract(x_test, canary_idxs_test)
    y_canary_test = _extract(y_test, canary_idxs_test)
  else:  # Random gradient: nothing to do here.
    x_canary_test = _extract(x_train, canary_idxs_test)
    y_canary_test = _extract(y_train, canary_idxs_test)

  train_dataset = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train, z_train)
  )
  canary_dataset = tf.data.Dataset.from_tensor_slices((
      _extract(x_train, canary_idxs),
      _extract(y_train, canary_idxs),
      _extract(z_train, canary_idxs),
  ))
  canary_dataset_test = tf.data.Dataset.from_tensor_slices((
      x_canary_test,
      y_canary_test,
      tf.convert_to_tensor(canary_principal_components_test),
  ))

  if num_canaries == 0:
    canary_dataset = None
  if num_canaries_test == 0:
    canary_dataset_test = None

  datasets = (train_dataset, test_dataset, canary_dataset, canary_dataset_test)
  return x_train.shape[0], datasets


def _extract(tensor: tf.Tensor, index: np.ndarray) -> tf.Tensor:
  """Return tensor[index] with appropriate type conversions."""
  return tf.convert_to_tensor(np.ascontiguousarray(tensor.numpy()[index]))


def _generate_canary_info(
    n, num_canaries, num_canaries_test, seed, min_dimension, max_dimension
):
  """Generate info about the canaries."""
  logging.warning(
      'Generating %d train and %d test canaries with seed %d',
      num_canaries,
      num_canaries_test,
      seed,
  )
  z_train = np.full(n, fill_value=-1, dtype=np.int64)
  rng = np.random.RandomState(seed)
  # Sample some examples to be designated as canaries
  idxs = rng.choice(n, size=num_canaries, replace=False)
  idxs_test = rng.choice(n, size=num_canaries_test, replace=False)
  # Sample some data principal components to be designated as canaries
  # Important: sample *with* replacement for exchangeability
  random_dims = min_dimension + rng.choice(
      max_dimension - min_dimension, size=num_canaries, replace=True
  )
  random_dims_test = min_dimension + rng.choice(
      max_dimension - min_dimension, size=num_canaries_test, replace=True
  )
  z_train[idxs] = random_dims
  z_train = tf.convert_to_tensor(z_train)
  return z_train, idxs, random_dims, idxs_test, random_dims_test


def _generate_static_data_poisoning(
    x_numpy,
    canary_principal_components,
    canary_principal_components_test,
    canary_data_scale,
):
  """Run PCA and return `canary_data_scale` times the principal components."""
  pca = sklearn.decomposition.PCA(random_state=0)  # Consistent results for PCA
  pca.fit(x_numpy)
  singular_vectors = pca.components_  # (n_components, n_features)
  canary_data = (
      canary_data_scale * (singular_vectors[canary_principal_components])
      + pca.mean_
  )
  canary_data_test = (
      canary_data_scale * (singular_vectors[canary_principal_components_test])
      + pca.mean_
  )
  return canary_data, canary_data_test
