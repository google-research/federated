# Copyright 2020, Google LLC.
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
"""Matrix factorization models for MovieLens rating prediction.

Usage:

  `get_matrix_factorization_model` produces all outputs required for either
    TFF simulation or server-side training using the Keras API.

  Call one of:
    `build_reconstruction_model`: to produce a builder for a
      `ReconstructionModel`.
    `build_tff_model`: to produce a builder for a vanilla `tff.learning.Model`.
    `build_keras_model`: to compile a standard Keras model for server-side use.

  To retrieve loss and metrics functions: `get_loss_fn` and `get_metrics_fn`.
"""

import collections
from typing import Callable, List

import attr
import tensorflow as tf
import tensorflow_federated as tff

from reconstruction import keras_utils
from reconstruction import reconstruction_model


# A wrapper for everything needed for TFF simulations or Keras training.
@attr.s(eq=False, frozen=True)
class MatrixFactorizationModel:
  model: tf.keras.Model = attr.ib()
  global_layers: List[tf.keras.layers.Layer] = attr.ib()
  local_layers: List[tf.keras.layers.Layer] = attr.ib()
  input_spec: tff.Type = attr.ib()


def get_matrix_factorization_model(
    num_users: int,
    num_items: int,
    num_latent_factors: int,
    personal_model: bool = False,
    add_biases: bool = True,
    l2_regularization: float = 0.0,
    spreadout_lambda: float = 0.0) -> MatrixFactorizationModel:
  """Defines a Keras matrix factorization model.

  Factorizes a num_users x num_items preferences matrix to a
  num_users x num_latent_factors users matrix U and a
  num_items x num_latent_factors items matrix I. If personal_model is True,
  each user only possesses their own row of U.

  The predicted rating for a user u and an item i is
  dot(U[u], I[i]) + b_u + b_i + mu, where b_u is a user-specific bias, b_i
  is an item-specific bias, and mu is a global bias. We minimize the MSE between
  the predicted rating and the actual one.

  Note that unlike some other methods, we learn the biases using gradient
  descent, rather than calculating them using the (weighted) mean of observed
  data.

  Also note that the bias terms are optional. All of the weights in this model
  can also optionally be L2 regularized.

  Observe that if the bias terms and L2 regularization are used, the predicted
  rating and the resulting loss are as in the "SVD" algorithm first
  presented in Simon Funk's famous "Try This At Home" blog post:
  https://sifter.org/~simon/journal/20061211.html

  Args:
    num_users: Number of users in the preferences matrix. Must be 1 if
      personal_model is True, otherwise raises a ValueError.
    num_items: Number of items in the preferences matrix.
    num_latent_factors: Number of latent factors to factorize the preferences
      matrix.
    personal_model: If True, the model contains only a user's own parameters and
      the model only expects item IDs as input. If True, num_users must be 1. If
      False, the model contains parameters for all users, and user IDs are
      expected as input along with item IDs. This should be set to True for
      experiments with federated reconstruction and False for experiments with
      server-side data.
    add_biases: If true, add three bias terms: (1) user-specific bias, (2)
      item-specific bias, and (3) global bias. These correspond to the b_u, b_i,
      and mu terms above.
    l2_regularization: The constant to use to scale L2 regularization on all
      weights, including the factorized matrices and the (optional) biases. A
      value of 0.0 indicates no regularization.
    spreadout_lambda: Scaling constant for spreadout regularization on item
      embeddings. This ensures that item embeddings are generally spread far
      apart, and that random items have dissimilar embeddings. See
      `EmbeddingSpreadoutRegularizer` for details. A value of 0.0 indicates no
      regularization.

  Returns:
    A `MatrixFactorizationModel` containing the Keras model (non-compiled)
    performing matrix factorization and global/local layers containing
    layers with parameters.
  """
  if personal_model and num_users != 1:
    raise ValueError('If personal_model is True, num_users must be 1.')

  # We won't include the input, flatten, or regularization layers in these lists
  # since they don't contain parameters.
  global_layers = []
  local_layers = []

  # Extract the item embedding.
  item_input = tf.keras.layers.Input(shape=[1], name='Item')
  item_embedding_layer = tf.keras.layers.Embedding(
      num_items,
      num_latent_factors,
      embeddings_regularizer=EmbeddingSpreadoutRegularizer(
          spreadout_lambda=spreadout_lambda,
          l2_normalize=False,
          l2_regularization=l2_regularization),
      name='ItemEmbedding')
  global_layers.append(item_embedding_layer)
  flat_item_vec = tf.keras.layers.Flatten(name='FlattenItems')(
      item_embedding_layer(item_input))

  # Extract the user embedding.
  if personal_model:
    user_embedding_layer = UserEmbedding(
        num_latent_factors,
        embedding_regularizer=tf.keras.regularizers.l2(l2_regularization),
        name='UserEmbedding')
    local_layers.append(user_embedding_layer)
    # The item_input never gets used by the user embedding layer,
    # but it allows the model to
    # hook up to a user_embedding w/o any user_input.
    flat_user_vec = user_embedding_layer(item_input)
  else:
    user_input = tf.keras.layers.Input(shape=[1], name='User')
    user_embedding_layer = tf.keras.layers.Embedding(
        num_users,
        num_latent_factors,
        embeddings_regularizer=tf.keras.regularizers.l2(l2_regularization),
        name='UserEmbedding')
    local_layers.append(user_embedding_layer)
    flat_user_vec = tf.keras.layers.Flatten(name='FlattenUsers')(
        user_embedding_layer(user_input))

  # Compute the dot product between the user embedding, and the item one.
  pred = tf.keras.layers.Dot(
      1, normalize=False, name='Dot')([flat_user_vec, flat_item_vec])

  # Optionally add three bias terms: (1) user-specific bias, (2) item-specific
  # bias, and (3) global bias.
  if add_biases:
    if personal_model:
      user_bias_layer = UserEmbedding(
          1,
          embedding_regularizer=tf.keras.regularizers.l2(l2_regularization),
          name='UserBias')
      local_layers.append(user_bias_layer)
      # The item_input never gets used by the user_bias_layer,
      # but it allows the model to
      # hook up w/o any user_input.
      flat_user_bias = user_bias_layer(item_input)
    else:
      user_bias_layer = tf.keras.layers.Embedding(
          num_users,
          1,
          embeddings_regularizer=tf.keras.regularizers.l2(l2_regularization),
          name='UserBias')
      local_layers.append(user_bias_layer)
      flat_user_bias = tf.keras.layers.Flatten(name='FlattenUserBias')(
          user_bias_layer(user_input))

    item_bias_layer = tf.keras.layers.Embedding(
        num_items,
        1,
        embeddings_regularizer=tf.keras.regularizers.l2(l2_regularization),
        name='ItemBias')
    global_layers.append(item_bias_layer)
    flat_item_bias = tf.keras.layers.Flatten(name='FlattenItemBias')(
        item_bias_layer(item_input))
    pred = tf.keras.layers.Add()([pred, flat_user_bias, flat_item_bias])

    global_bias_layer = AddBias(l2_regularization, name='GlobalBias')
    global_layers.append(global_bias_layer)
    pred = global_bias_layer(pred)

  # Produce different input_specs and Keras inputs depending on whether the
  # model expects user IDs in addition to item IDs as input.
  if personal_model:
    input_spec = collections.OrderedDict(
        x=tf.TensorSpec(shape=[None, 1], dtype=tf.int64),
        y=tf.TensorSpec(shape=[None, 1], dtype=tf.float32))
    keras_inputs = item_input
  else:
    input_spec = collections.OrderedDict(
        x=(tf.TensorSpec(shape=[None, 1], dtype=tf.int64),
           tf.TensorSpec(shape=[None, 1], dtype=tf.int64)),
        y=tf.TensorSpec(shape=[None, 1], dtype=tf.float32))
    keras_inputs = [user_input, item_input]

  model = tf.keras.Model(inputs=keras_inputs, outputs=pred)

  return MatrixFactorizationModel(model, global_layers, local_layers,
                                  input_spec)


def build_reconstruction_model(
    model_builder: Callable[[], MatrixFactorizationModel],
    global_variables_only: bool = False,
) -> Callable[[], reconstruction_model.ReconstructionModel]:
  """Returns a function to build the matrix factorization `ReconstructionModel`.

  Can be used to produce a `ReconstructionModel` in which some variables are
  global and some variables are local (for use with Federated Reconstruction).

  Taking in and outputting builders rather than the actual
  `MatrixFactorizationModel` and `ReconstructionModel` is required for TFF
  execution.

  Args:
    model_builder: A no-arg function returning a `MatrixFactorization` model
      containing the underlying Keras model, its global layers, its local
      layers, and tff.Type input spec.
    global_variables_only: If True, the returned `ReconstructionModel` contains
      all model variables as global variables. This can be useful for
      baselines involving aggregating all variables.

  Returns:
    A no-arg function that returns a `ReconstructionModel`.
  """

  def reconstruction_model_fn() -> reconstruction_model.ReconstructionModel:
    matrix_factorization_model = model_builder()

    global_layers = matrix_factorization_model.global_layers
    local_layers = matrix_factorization_model.local_layers
    # Merge local layers into global layers if needed.
    if global_variables_only:
      global_layers.extend(local_layers)
      local_layers = []

    return keras_utils.from_keras_model(
        keras_model=matrix_factorization_model.model,
        global_layers=global_layers,
        local_layers=local_layers,
        input_spec=matrix_factorization_model.input_spec)

  return reconstruction_model_fn


def build_tff_model(
    model_builder: Callable[[], MatrixFactorizationModel],
    accuracy_threshold: float = 0.5,
) -> Callable[[], tff.learning.Model]:
  """Returns a non-custom `tff.learning.Model` builder.

  Can be used to produce a vanilla `tff.learning.Model`, without global and
  local variables, for use with vanilla federated averaging.

  Taking in and outputting builders rather than the actual
  `MatrixFactorizationModel` and `tff.learning.Model` is required for TFF
  execution.

  Args:
    model_builder: A no-arg function returning a `MatrixFactorization` model
      containing the underlying Keras model and tff.Type input spec.
    accuracy_threshold: Threshold to use to determine whether a prediction is
      considered correct for `ReconstructionAccuracyMetric`.

  Returns:
    A no-arg function returning a `tff.learning.Model`, usable with
      tff.learning.build_federated_averaging_process.
  """

  def tff_model_fn() -> tff.learning.Model:
    matrix_factorization_model = model_builder()
    loss_fn = get_loss_fn()
    metrics_fn = get_metrics_fn(accuracy_threshold)
    return tff.learning.from_keras_model(
        keras_model=matrix_factorization_model.model,
        loss=loss_fn(),
        input_spec=matrix_factorization_model.input_spec,
        metrics=metrics_fn())

  return tff_model_fn


def build_keras_model(
    matrix_factorization_model: MatrixFactorizationModel,
    optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.SGD(1.0),
    accuracy_threshold: float = 0.5,
) -> tf.keras.Model:
  """Returns a compiled Keras model for server-side training/evaluation.

  Useful for producing baselines for federated training and hyperparameter
  tuning.

  Args:
    matrix_factorization_model: A `MatrixFactorization` model containing the
      underlying Keras model.
    optimizer: a Keras optimizer to compile the model with.
    accuracy_threshold: Threshold to use to determine whether a prediction is
      considered correct for `ReconstructionAccuracyMetric`.

  Returns:
    A compiled `tf.keras.Model` model.
  """
  keras_model = matrix_factorization_model.model
  loss_fn = get_loss_fn()
  metrics_fn = get_metrics_fn(accuracy_threshold)
  keras_model.compile(optimizer=optimizer, loss=loss_fn(), metrics=metrics_fn())
  return keras_model


def get_loss_fn() -> Callable[[], tf.keras.losses.Loss]:
  """Returns a builder for mean squared error loss."""

  def loss_fn() -> tf.keras.losses.Loss:
    return tf.keras.losses.MeanSquaredError()

  return loss_fn


def get_metrics_fn(
    accuracy_threshold: float = 0.5,
) -> Callable[[], List[tf.keras.metrics.Metric]]:
  """Returns a builder for matrix factorization model metrics.

  Metrics are `ReconstructionAccuracyMetric`, `NumExamplesCounter`, and
  `NumBatchesCounter`.

  Args:
    accuracy_threshold: Threshold to use to determine whether a prediction is
      considered correct for `ReconstructionAccuracyMetric`.

  Returns:
    A no-argument function returning a list of Keras metrics.
  """

  def metrics_fn() -> List[tf.keras.metrics.Metric]:
    return [
        ReconstructionAccuracyMetric(accuracy_threshold),
        NumExamplesCounter(),
        NumBatchesCounter()
    ]

  return metrics_fn


class UserEmbedding(tf.keras.layers.Layer):
  """Keras layer representing an embedding for a single user."""

  def __init__(self, num_latent_factors, embedding_regularizer=None, **kwargs):
    super().__init__(**kwargs)
    self.num_latent_factors = num_latent_factors
    self.regularizer = tf.keras.regularizers.get(embedding_regularizer)

  def build(self, input_shape):
    self.embedding = self.add_weight(
        shape=(1, self.num_latent_factors),
        initializer='uniform',
        regularizer=self.regularizer,
        dtype=tf.float32,
        name='UserEmbeddingKernel')
    super().build(input_shape)

  def call(self, inputs):
    return self.embedding

  def compute_output_shape(self):
    return (1, self.num_latent_factors)


class AddBias(tf.keras.layers.Layer):
  """Simple Keras Layer for adding a (regularized) bias to inputs."""

  def __init__(self, l2_regularization=0.0, name='AddBias'):
    super().__init__(name=name)
    self.regularizer = tf.keras.regularizers.l2(l2_regularization)

  def build(self, input_shape):
    self.bias = self.add_weight(
        shape=(),
        initializer='zeros',
        regularizer=self.regularizer,
        dtype=tf.float32,
        name='Bias')
    super().build(input_shape)

  def call(self, inputs):
    return inputs + self.bias


class EmbeddingSpreadoutRegularizer(tf.keras.regularizers.Regularizer):
  """Regularizer for ensuring embeddings are spreadout within embedding space.

  This is a variation on approaches for ensuring that embeddings/encodings of
  different items in either an input space or an output space are "spread out".
  This corresponds to randomly selected pairs of embeddings having a low
  dot product or cosine similarity, and can be used to effectively add a form
  of negative sampling to models. The original paper that proposed
  regularization for this purpose is at https://arxiv.org/abs/1708.06320,
  although we do a few things different here: (1) we apply spreadout on input
  embeddings, not the learned encodings of a two-tower model, (2) we apply the
  regularization per batch, instead of per epoch, (3) we leave L2 normalization
  before applying spreadout as optional–since using the dot product here.

  spreadout_lambda scales the regularization magnitude. l2_normalize is
  whether to perform l2_normalization on embeddings before applying spreadout.
  If this is True, spreadout will be performed using cosine similarities,
  otherwise it will be performed using dot products. If l2_regularization is
  nonzero, standard L2 regularization with l2_regularization as a scaling
  constant is applied. Note that l2_regularization is applied before any L2
  normalization, so this has the same effect as adding L2 regularization with
  tf.keras.regularizers.l2.

  Usage example:
    embedding_layer = tf.keras.layers.Embedding(
        ...
        embeddings_regularizer=EmbeddingSpreadoutRegularizer(spreadout_lambda))
  """

  def __init__(self,
               spreadout_lambda: float = 0.0,
               l2_normalize: bool = False,
               l2_regularization: float = 0.0):
    self.spreadout_lambda = spreadout_lambda
    self.l2_normalize = l2_normalize
    self.l2_regularization = l2_regularization

  def __call__(self, weights):
    total_regularization = 0.0

    # Apply optional L2 regularization before normalization.
    if self.l2_regularization:
      total_regularization += self.l2_regularization * tf.keras.backend.sum(
          tf.keras.backend.square(weights))

    if self.l2_normalize:
      weights = tf.keras.backend.l2_normalize(weights, axis=-1)

    similarities = tf.keras.backend.dot(weights,
                                        tf.keras.backend.transpose(weights))
    similarities = tf.linalg.set_diag(
        similarities,
        tf.zeros([tf.keras.backend.shape(weights)[0]], dtype=tf.float32))
    similarities_norm = tf.sqrt(tf.reduce_sum(tf.square(similarities)))

    total_regularization += self.spreadout_lambda * similarities_norm

    return total_regularization

  def get_config(self):
    return {
        'spreadout_lambda': self.spreadout_lambda,
        'l2_normalize': self.l2_normalize,
        'l2_regularization': self.l2_regularization,
    }


class ReconstructionAccuracyMetric(tf.keras.metrics.Mean):
  """Keras metric computing accuracy of reconstructed preferences.

  Computes the fraction of predicted preferences within a threshold of the
  actual preferences. Examples across batches are weighted equally unless
  manually specified sample_weights are provided to update_state.
  """

  def __init__(self,
               threshold: float = 0.5,
               name: str = 'reconstruction_accuracy_metric',
               **kwargs):
    super().__init__(name=name, **kwargs)
    self.threshold = threshold

  def update_state(self, y_true: tf.Tensor,
                   y_pred: tf.Tensor,
                   sample_weight: tf.Tensor = None):
    y_true = tf.keras.backend.cast(y_true, self._dtype)
    y_pred = tf.keras.backend.cast(y_pred, self._dtype)
    absolute_diffs = tf.keras.backend.abs(y_true - y_pred)
    # A [batch_size, 1] tf.bool tensor indicating correctness within the
    # threshold for each example in a batch.
    example_accuracies = tf.keras.backend.less_equal(absolute_diffs,
                                                     self.threshold)

    super().update_state(example_accuracies, sample_weight=sample_weight)

  def get_config(self):
    config = {'threshold': self.threshold}
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


class NumExamplesCounter(tf.keras.metrics.Sum):
  """A custom sum that counts the number of examples seen.

  `sample_weight` is unused since this is just a counter.
  """

  def __init__(self,
               name: str = 'num_examples',
               **kwargs):
    super().__init__(name=name, **kwargs)

  def update_state(self, y_true, y_pred, sample_weight=None):
    return super().update_state(tf.shape(y_pred)[0])


class NumBatchesCounter(tf.keras.metrics.Sum):
  """A custom sum that counts the number of batches seen.

  `sample_weight` is unused since this is just a counter.
  """

  def __init__(self,
               name: str = 'num_batches',
               **kwargs):
    super().__init__(name=name, **kwargs)

  def update_state(self, y_true, y_pred, sample_weight=None):
    return super().update_state(1)
