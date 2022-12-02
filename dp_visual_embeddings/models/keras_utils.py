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
"""Utils for Keras embedding models."""

import collections
from collections.abc import Callable
import dataclasses
from typing import Any, Optional, Union
import warnings
from absl import logging

import tensorflow as tf
import tensorflow_federated as tff

from dp_visual_embeddings.models import embedding_model as model_lib

LossType = Union[tf.keras.losses.Loss, list[tf.keras.losses.Loss]]
# A finalizer of a Keras metric is a `tf.function` decorated callable that takes
# in the unfinalized values of this Keras metric (i.e., the tensor values of the
# variables in `keras_metric.variables`), and returns the value of
# `keras_metric.result()`.
KerasMetricFinalizerType = collections.OrderedDict[str,
                                                   Callable[[list[tf.Tensor]],
                                                            Any]]


@dataclasses.dataclass
class EmbeddingModel:
  """Data class for Keras embedding models.

  The `global_variables` tracks the backbone (feature extractor) of the
  embedding model, and the `client_variables` tracks the head for training.
  """
  model: tf.keras.Model
  global_variables: tff.learning.ModelWeights
  client_variables: tff.learning.ModelWeights


class EmbedNormLayer(tf.keras.layers.Layer):
  """A keras layer to normalize the output embeddings of an embedding model.

  If always_normalize is False, only normalizes the embeddings during the
  inference time, controlled by the `training` flag in `call`. If
  always_normalize is True, normalizes the embeddings for both training and
  inference.
  """

  def __init__(self, always_normalize: bool = False, **kwargs):
    super().__init__(**kwargs)
    self._always_normalize = always_normalize

  def call(self, embeddings: tf.Tensor, training: bool = False) -> tf.Tensor:
    if training and not self._always_normalize:
      return embeddings
    else:
      norms = tf.norm(embeddings, ord='euclidean', axis=1, keepdims=True)
      normalized_embeddings = tf.math.divide_no_nan(embeddings, norms)
      return normalized_embeddings


class DenseNormLayer(tf.keras.layers.Dense):
  """A dense keras layer that has weights for each output to have unit norm.

  If the input is also normalized to have unit norm, then the output will be
  the cos similarity between input and weight vectors. We then scale the [-1, 1]
  outputs by a trainable scalar `global_scale`.
  """

  def __init__(self, units, **kwargs):
    super().__init__(units, **kwargs)
    if self.use_bias:
      raise ValueError('DenseNormLayer does not support bias.')

  def build(self, input_shape):
    super().build(input_shape)
    self.global_scale = self.add_weight(
        'global_scale',
        shape=[],
        initializer=tf.keras.initializers.Ones(),
        dtype=self.dtype,
        trainable=True)
    if self.kernel.shape.rank != 2:
      raise ValueError(
          f'Invalid DenseNormLayer kernel rank {self.kernel.shape.rank}')

  def call(self, inputs):
    outputs = super().call(inputs)
    scales = tf.norm(self.kernel, ord='euclidean', axis=0, keepdims=True)
    norm_outputs = tf.math.divide_no_nan(outputs, scales)
    return self.global_scale * norm_outputs


def add_embedding_head(base_model: tf.keras.Model,
                       num_identities: int,
                       fix_normal_init: bool = False,
                       use_normalize: bool = False) -> EmbeddingModel:
  """Adds a head to a base Keras embedding model.

  The base Keras model (e.g., a MobileNetV2 base) processes an image into an
  embedding vector, and this method adds a head to the model that dots this
  predicted embedding (via a dense layer) against a set of reference embeddings.
  The output is a `num_identities`-dimension vector of similarities of the input
  image's embedding with the reference embeddings.

  The `global_variables` of the `base_model` and the `client_variables` of the
  added head are also returned for federated partially local training.

  Args:
    base_model: The tf.keras.Model defining the prediction base of the neural
      network.
    num_identities: The number of reference identities to learn embeddings.
    fix_normal_init: A boolean to control the initialization and regularization
      of the head. If True, use a dense layer with initialization and
      regularizationthat is specialized for ResNet with fixed variance and l2
      regularization. Otherwise, use the default Xavier/Glorot initialization.
    use_normalize: If `True`, use `DenseNormLayer` for the added head.

  Returns:
    A `EmbeddingModel` includes (`tf.keras.Model`, global_variables,
      client_variables).
  """
  img_input = base_model.input
  predicted_embeddings = base_model(img_input)

  # This dense layer has a weight matrix with dimensions: embedding_dim_size x
  # num_identities. During training it will learn a reference embedding for each
  # identity.
  # The output of this layer has dimensions: batch_size x num_identities. For
  # each input in the batch, this is a vector of how similar the input is
  # predicted to be to each of the reference identities.
  if use_normalize:
    dense_layer_fn = DenseNormLayer
  else:
    dense_layer_fn = tf.keras.layers.Dense
  if fix_normal_init:
    # In limited experiments, the initialization does not affect the performance
    # much, so default to False.
    embedding_layer = dense_layer_fn(
        num_identities,
        activation=None,
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        use_bias=False,
        name='similarity_with_reference_embeddings')
  else:
    embedding_layer = dense_layer_fn(
        num_identities,
        activation=None,
        use_bias=False,
        name='similarity_with_reference_embeddings')
  similarity_with_reference_embeddings = embedding_layer(predicted_embeddings)

  global_variables = tff.learning.ModelWeights(
      trainable=base_model.trainable_variables,
      non_trainable=base_model.non_trainable_variables)
  client_variables = tff.learning.ModelWeights(
      trainable=embedding_layer.trainable_variables,
      non_trainable=embedding_layer.non_trainable_variables)
  model = tf.keras.models.Model(
      inputs=img_input,
      outputs=[similarity_with_reference_embeddings, predicted_embeddings])
  return EmbeddingModel(
      model=model,
      global_variables=global_variables,
      client_variables=client_variables)


# TODO(b/197746608): Remove the code path that takes in constructed Keras
# metrics, because reconstructing metrics via `from_config` can cause problems.
def from_keras_model(
    keras_model: tf.keras.Model,
    *,
    global_variables: tff.learning.ModelWeights,
    client_variables: tff.learning.ModelWeights,
    loss: LossType,
    input_spec: Any,
    loss_weights: Optional[list[float]] = None,
    metrics: Optional[Union[list[tf.keras.metrics.Metric],
                            list[Callable[[], tf.keras.metrics.Metric]]]] = None
) -> model_lib.Model:
  """Builds a `embedding_model.Model` from a `tf.keras.Model`.

  The code is mostly forked from `tff.learning.from_keras_model` with supports
  on `global_variables` and `client_variables`. The `global_variables` are
  aggregated across clients, while the `client_variables` are updated and kept
  local on clients.

  The `tff.learning.Model` returned by this function uses `keras_model` for
  its forward pass and autodifferentiation steps. The returned model will have
  three additional metrics including: loss, num_examples, and num_batches.

  Notice that since TFF couples the `tf.keras.Model` and `loss`,
  TFF needs a slightly different notion of "fully specified type" than
  pure Keras does. That is, the model `M` takes inputs of type `x` and
  produces predictions of type `p`; the loss function `L` takes inputs of type
  `<p, y>` (where `y` is the ground truth label type) and produces a scalar.
  Therefore in order to fully specify the type signatures for computations in
  which the generated `tff.learning.Model` will appear, TFF needs the type `y`
  in addition to the type `x`.

  Note: This function does not currently accept subclassed `tf.keras.Models`,
  as it makes assumptions about presence of certain attributes which are
  guaranteed to exist through the functional or Sequential API but are
  not necessarily present for subclassed models.

  Note: This function raises a UserWarning if the `tf.keras.Model` contains a
  BatchNormalization layer, as the batch mean and variance will be treated as
  non-trainable variables and won't be updated during the training (see
  b/186845846 for more information). Consider using Group Normalization instead.

  Args:
    keras_model: A `tf.keras.Model` object that is not compiled.
    global_variables: A `tff.learning.ModelWeights` for global variables.
    client_variables: A `tff.learning.ModelWeights` for client local variables.
    loss: A single `tf.keras.losses.Loss` or a list of losses-per-output. If a
      single loss is provided, then all model output (as well as all prediction
      information) is passed to the loss; this includes situations of multiple
      model outputs and/or predictions. If multiple losses are provided as a
      list, then each loss is expected to correspond to a model output; the
      model will attempt to minimize the sum of all individual losses
      (optionally weighted using the `loss_weights` argument).
    input_spec: A structure of `tf.TensorSpec`s or `tff.Type` specifying the
      type of arguments the model expects. If `input_spec` is a `tff.Type`, its
      leaf nodes must be `TensorType`s. Note that `input_spec` must be a
      compound structure of two elements, specifying both the data fed into the
      model (x) to generate predictions as well as the expected type of the
      ground truth (y). If provided as a list, it must be in the order [x, y].
      If provided as a dictionary, the keys must explicitly be named `'{}'` and
      `'{}'`.
    loss_weights: (Optional) A list of Python floats used to weight the loss
      contribution of each model output (when providing a list of losses for the
      `loss` argument).
    metrics: (Optional) a list of `tf.keras.metrics.Metric` objects or a list of
      no-arg callables that each constructs a `tf.keras.metrics.Metric`. Note:
      if metrics with names `num_examples` or `num_batches` are found, they will
      be used as-is. If they are not found in the `metrics` argument list, they
      will be added by default using `tff.learning.metrics.NumExamplesCounter`
      and `tff.learning.metrics.NumBatchesCounter` respectively.

  Returns:
    A `embedding_model.Model` object.

  Raises:
    ValueError: If `keras_model` was compiled, if `loss` is a list of unequal
      length to the number of outputs of `keras_model`, if `loss_weights` is
      specified but `loss` is not a list, if `input_spec` does not contain
      exactly two elements, or if `input_spec` is a dictionary and does not
      contain keys `'x'` and `'y'`.
  """.format(model_lib.MODEL_ARG_NAME, model_lib.MODEL_LABEL_NAME)
  if keras_model._is_compiled:  # pylint: disable=protected-access
    raise ValueError('`keras_model` must not be compiled')

  # Validate and normalize `loss` and `loss_weights`
  if not isinstance(loss, list):
    if loss_weights is not None:
      raise ValueError('`loss_weights` cannot be used if `loss` is not a list.')
    loss = [loss]
    loss_weights = [1.0]
  else:
    if len(loss) != len(keras_model.outputs):
      raise ValueError('If a loss list is provided, `keras_model` must have '
                       'equal number of outputs to the losses.\nloss: {}\nof '
                       'length: {}.\noutputs: {}\nof length: {}.'.format(
                           loss, len(loss), keras_model.outputs,
                           len(keras_model.outputs)))

    if loss_weights is None:
      loss_weights = [1.0] * len(loss)
    else:
      if len(loss) != len(loss_weights):
        raise ValueError(
            '`keras_model` must have equal number of losses and loss_weights.'
            '\nloss: {}\nof length: {}.'
            '\nloss_weights: {}\nof length: {}.'.format(loss, len(loss),
                                                        loss_weights,
                                                        len(loss_weights)))

  if len(input_spec) != 2:
    raise ValueError('The top-level structure in `input_spec` must contain '
                     'exactly two top-level elements, as it must specify type '
                     'information for both inputs to and predictions from the '
                     'model. You passed input spec {}.'.format(input_spec))
  if isinstance(input_spec, collections.abc.Mapping):
    if model_lib.MODEL_ARG_NAME not in input_spec:
      raise ValueError(
          'The `input_spec` is a collections.abc.Mapping (e.g., a dict), so it '
          'must contain an entry with key `\'{}\'`, representing the input(s) '
          'to the Keras model.'.format(model_lib.MODEL_ARG_NAME))
    if model_lib.MODEL_LABEL_NAME not in input_spec:
      raise ValueError(
          'The `input_spec` is a collections.abc.Mapping (e.g., a dict), so it '
          'must contain an entry with key `\'{}\'`, representing the label(s) '
          'to be used in the Keras loss(es).'.format(
              model_lib.MODEL_LABEL_NAME))

  if metrics is None:
    metrics = []

  for layer in keras_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
      warnings.warn(
          'Batch Normalization contains non-trainable variables that won\'t be '
          'updated during the training. Consider using Group Normalization '
          'instead.', UserWarning)
      break

  return _KerasModel(
      keras_model,
      global_variables=global_variables,
      client_variables=client_variables,
      input_spec=input_spec,
      loss_fns=loss,
      loss_weights=loss_weights,
      metrics=metrics)


class _KerasModel(model_lib.Model):
  """Internal wrapper class for `tf.keras.Model` objects."""

  def __init__(self, keras_model: tf.keras.Model, *,
               global_variables: tff.learning.ModelWeights,
               client_variables: tff.learning.ModelWeights, input_spec,
               loss_fns: list[tf.keras.losses.Loss], loss_weights: list[float],
               metrics: Union[list[tf.keras.metrics.Metric],
                              list[Callable[[], tf.keras.metrics.Metric]]]):
    self._keras_model = keras_model
    self._global_variables = global_variables
    self._client_variables = client_variables
    self._input_spec = input_spec
    self._loss_fns = loss_fns
    self._loss_weights = loss_weights

    self._metrics: list[tf.keras.metrics.Metric] = []
    self._metric_constructors = []

    metric_names = set([])
    if metrics:
      has_keras_metric = False
      has_keras_metric_constructor = False

      for metric in metrics:
        if isinstance(metric, tf.keras.metrics.Metric):
          self._metrics.append(metric)
          metric_names.add(metric.name)
          has_keras_metric = True
        elif callable(metric):
          constructed_metric = metric()
          if not isinstance(constructed_metric, tf.keras.metrics.Metric):
            raise TypeError(
                f'Metric constructor {metric} is not a no-arg callable that '
                'creates a `tf.keras.metrics.Metric`, it created a '
                f'{type(constructed_metric).__name__}.')
          metric_names.add(constructed_metric.name)
          self._metric_constructors.append(metric)
          self._metrics.append(constructed_metric)
          has_keras_metric_constructor = True
        else:
          raise TypeError(
              'Expected the input metric to be either a '
              '`tf.keras.metrics.Metric` or a no-arg callable that constructs '
              'a `tf.keras.metrics.Metric`, found a non-callable '
              f'{type(metric)}.')

      if has_keras_metric and has_keras_metric_constructor:
        raise TypeError(
            'Expected the input `metrics` to be either a list of '
            '`tf.keras.metrics.Metric` objects or a list of no-arg callables '
            'that each constructs a `tf.keras.metrics.Metric`, '
            f'found both types in the `metrics`: {metrics}.')

    # This is defined here so that it closes over the `loss_fn`.
    class _WeightedMeanLossMetric(tf.keras.metrics.Mean):
      """A `tf.keras.metrics.Metric` wrapper for the loss function."""

      def __init__(self, name='loss', dtype=tf.float32):
        super().__init__(name=name, dtype=dtype)
        self._loss_fns = loss_fns
        self._loss_weights = loss_weights

      def update_state(self, y_true, y_pred, sample_weight=None):  # pytype: disable=signature-mismatch
        if isinstance(y_pred, list):
          batch_size = tf.shape(y_pred[0])[0]
        else:
          batch_size = tf.shape(y_pred)[0]

        if len(self._loss_fns) == 1:
          batch_loss = self._loss_fns[0](y_true, y_pred)
        else:
          batch_loss = tf.zeros(())
          for i in range(len(self._loss_fns)):
            batch_loss += self._loss_weights[i] * self._loss_fns[i](y_true[i],
                                                                    y_pred[i])

        return super().update_state(batch_loss, batch_size)

    extra_metrics_constructors = [_WeightedMeanLossMetric]
    if 'num_examples' not in metric_names:
      logging.info('Adding default num_examples metric to model')
      extra_metrics_constructors.append(tff.learning.metrics.NumExamplesCounter)
    if 'num_batches' not in metric_names:
      logging.info('Adding default num_batches metric to model')
      extra_metrics_constructors.append(tff.learning.metrics.NumBatchesCounter)
    self._metrics.extend(m() for m in extra_metrics_constructors)
    if not metrics or self._metric_constructors:
      self._metric_constructors.extend(extra_metrics_constructors)

  @property
  def trainable_variables(self):
    return self._global_variables.trainable

  @property
  def non_trainable_variables(self):
    return self._global_variables.non_trainable

  @property
  def client_trainable_variables(self):
    return self._client_variables.trainable

  @property
  def client_non_trainable_variables(self):
    return self._client_variables.non_trainable

  @property
  def local_variables(self):
    local_variables = []
    for metric in self.get_metrics():
      local_variables.extend(metric.variables)
    return local_variables

  def get_metrics(self) -> list[tf.keras.metrics.Metric]:
    return self._metrics

  @property
  def input_spec(self):
    return self._input_spec

  @tf.function
  def predict_on_batch(self, x, training=True):
    return self._keras_model(x, training=training)

  def _forward_pass(self, batch_input, training=True):
    if isinstance(batch_input, collections.abc.Mapping):
      inputs = batch_input.get('x')
    else:
      inputs = batch_input[0]
    if inputs is None:
      raise KeyError('Received a batch_input that is missing required key `x`. '
                     f'Instead have keys {list(batch_input.keys())}')
    predictions = self.predict_on_batch(inputs, training)

    if isinstance(batch_input, collections.abc.Mapping):
      y_true = batch_input.get('y')
    else:
      y_true = batch_input[1]
    if y_true is not None:
      if len(self._loss_fns) == 1:
        loss_fn = self._loss_fns[0]
        # Note: we add each of the per-layer regularization losses to the loss
        # that we use to update trainable parameters, in addition to the
        # user-provided loss function. Keras does the same in the
        # `tf.keras.Model` training step. This is expected to have no effect if
        # no per-layer losses are added to the model.
        batch_loss = tf.add_n([loss_fn(y_true=y_true, y_pred=predictions)] +
                              self._keras_model.losses)

      else:
        # Note: we add each of the per-layer regularization losses to the losses
        # that we use to update trainable parameters, in addition to the
        # user-provided loss functions. Keras does the same in the
        # `tf.keras.Model` training step. This is expected to have no effect if
        # no per-layer losses are added to the model.
        batch_loss = tf.add_n([tf.zeros(())] + self._keras_model.losses)
        for i in range(len(self._loss_fns)):
          loss_fn = self._loss_fns[i]
          loss_wt = self._loss_weights[i]
          batch_loss += loss_wt * loss_fn(
              y_true=y_true[i], y_pred=predictions[i])
    else:
      batch_loss = None

    # TODO(b/145308951): Follow up here to pass through sample_weight in the
    # case that we have a model supporting masking.
    for metric in self.get_metrics():
      metric.update_state(y_true=y_true, y_pred=predictions)

    def nrows(t):
      return t.nrows() if isinstance(t, tf.RaggedTensor) else tf.shape(t)[0]

    return tff.learning.BatchOutput(
        loss=batch_loss,
        predictions=predictions,
        num_examples=nrows(tf.nest.flatten(inputs)[0]))

  @tf.function
  def forward_pass(self, batch_input, training=True):
    return self._forward_pass(batch_input, training=training)

  @tf.function
  def report_local_unfinalized_metrics(
      self) -> collections.OrderedDict[str, list[tf.Tensor]]:
    """Creates an `OrderedDict` of metric names to unfinalized values.

    Returns:
      An `OrderedDict` of metric names to lists of unfinalized metric values.
      For a Keras metric, its unfinalized values are the tensor values of its
      variables tracked during local training. The returned `OrderedDict` has
      the same keys (metric names) as the `OrderedDict` returned by the method
      `metric_finalizers()`, and can be used as input to the finalizers to get
      the finalized metric values. This method and the `metric_finalizers()`
      method can be used to construct a cross-client metrics aggregator when
      defining the federated training processes or evaluation computations.
    """
    outputs = collections.OrderedDict()
    for metric in self.get_metrics():
      outputs[metric.name] = [v.read_value() for v in metric.variables]
    return outputs

  def metric_finalizers(self) -> KerasMetricFinalizerType:  # pylint: disable=g-bare-generic
    """Creates an `OrderedDict` of metric names to finalizers.

    Returns:
      An `OrderedDict` of metric names to finalizers. A finalizer of a Keras
      metric is a `tf.function` decorated callable that takes in this metric's
      unfinalized values (created by `report_local_unfinalized_metrics`), and
      returns the metric value computed by `tf.keras.metrics.Metric.result()`.
      This method and the `report_local_unfinalized_metrics` method can be used
      to construct a cross-client metrics aggregator when defining the federated
      training processes or evaluation computations.
    """
    finalizers = collections.OrderedDict()
    for metric in self.get_metrics():
      finalizers[
          metric.name] = tff.learning.metrics.create_keras_metric_finalizer(
              metric)
    return finalizers

  @tf.function
  def reset_metrics(self):
    """Resets metrics variables to initial value."""
    raise NotImplementedError(
        'The `reset_metrics` method isn\'t implemented for your custom '
        '`tff.learning.Model`. Please implement it before using this method. '
        'You can leave this method unimplemented if you won\'t use this method.'
    )
