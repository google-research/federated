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
"""Utility methods for working with Keras in TensorFlow Federated.

The model wrapper is defined for language models to be used for StackOverflow.

There are several changes worth highlighting: (1) the usage of two-branch
networks; (2) the usage of label smoothing and regularization; (3) batch-based
voting for determing the branch.
"""

# TODO(b/193904908): add unit tests.

import collections
from typing import Callable, List, Optional, OrderedDict, Union
import warnings

import attr
import tensorflow as tf
import tensorflow_federated as tff


Loss = Union[tf.keras.losses.Loss, List[tf.keras.losses.Loss]]


# TODO(b/197746608): Remove the code path that takes in constructed Keras
# metrics, because reconstructing metrics via `from_config` can cause problems.
def from_keras_model(
    keras_model: tf.keras.Model,
    loss: Loss,
    input_spec,
    loss_weights: Optional[List[float]] = None,
    metrics: Optional[Union[List[tf.keras.metrics.Metric],
                            List[Callable[[],
                                          tf.keras.metrics.Metric]]]] = None,
    from_logits: bool = True,
    uniform_reg: float = 0.,
    label_smoothing: float = 1.,
    batch_majority_voting: bool = False,
    use_mixed: bool = False,
) -> tff.learning.Model:
  """Builds a `tff.learning.Model` from a `tf.keras.Model`.

  The `tff.learning.Model` returned by this function uses `keras_model` for
  its forward pass and autodifferentiation steps.

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
      no-arg callables that each constructs a `tf.keras.metrics.Metric`.

  Returns:
    A `tff.learning.Model` object.

  Raises:
    TypeError: If `keras_model` is not an instance of `tf.keras.Model`, if
      `loss` is not an instance of `tf.keras.losses.Loss` nor a list of
      instances of `tf.keras.losses.Loss`, if `input_spec` is a `tff.Type` but
      the leaf nodes are not `tff.TensorType`s, if `loss_weight` is provided but
      is not a list of floats, or if `metrics` is provided but is not a list of
      instances of `tf.keras.metrics.Metric`.
    ValueError: If `keras_model` was compiled, if `loss` is a list of unequal
      length to the number of outputs of `keras_model`, if `loss_weights` is
      specified but `loss` is not a list, if `input_spec` does not contain
      exactly two elements, or if `input_spec` is a dictionary and does not
      contain keys `'x'` and `'y'`.
  """.format(tff.learning.model.MODEL_ARG_NAME,
             tff.learning.model.MODEL_LABEL_NAME)
  # Validate `keras_model`
  tff.types.py_typecheck.check_type(keras_model, tf.keras.Model)
  if keras_model._is_compiled:  # pylint: disable=protected-access
    raise ValueError('`keras_model` must not be compiled')

  # Validate and normalize `loss` and `loss_weights`
  if not isinstance(loss, list):
    tff.types.py_typecheck.check_type(loss, tf.keras.losses.Loss)
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
    for loss_fn in loss:
      tff.types.py_typecheck.check_type(loss_fn, tf.keras.losses.Loss)

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
      for loss_weight in loss_weights:
        tff.types.py_typecheck.check_type(loss_weight, float)

  if len(input_spec) != 2:
    raise ValueError('The top-level structure in `input_spec` must contain '
                     'exactly two top-level elements, as it must specify type '
                     'information for both inputs to and predictions from the '
                     'model. You passed input spec {}.'.format(input_spec))
  if isinstance(input_spec, tff.types.Type):
    if not tff.types.is_structure_of_tensors(input_spec):
      raise TypeError(
          'Expected a `tff.Type` with all the leaf nodes being '
          '`tff.TensorType`s, found an input spec {}.'.format(input_spec))
    input_spec = tff.structure_from_tensor_type_tree(
        lambda tensor_type: tf.TensorSpec(tensor_type.shape, tensor_type.dtype),
        input_spec)
  else:
    tensor_spec = (tf.TensorSpec, tf.RaggedTensorSpec)
    tf.nest.map_structure(
        lambda s: tff.types.py_typecheck.check_type(  # pylint:disable=g-long-lambda,line-too-long
            s, tensor_spec, 'input spec member'),
        input_spec)
  if isinstance(input_spec, collections.abc.Mapping):
    if tff.learning.model.MODEL_ARG_NAME not in input_spec:
      raise ValueError(
          'The `input_spec` is a collections.abc.Mapping (e.g., a dict), so it '
          'must contain an entry with key `\'{}\'`, representing the input(s) '
          'to the Keras model.'.format(tff.learning.model.MODEL_ARG_NAME))
    if tff.learning.model.MODEL_LABEL_NAME not in input_spec:
      raise ValueError(
          'The `input_spec` is a collections.abc.Mapping (e.g., a dict), so it '
          'must contain an entry with key `\'{}\'`, representing the label(s) '
          'to be used in the Keras loss(es).'.format(
              tff.learning.model.MODEL_LABEL_NAME))

  if metrics is None:
    metrics = []
  else:
    tff.types.py_typecheck.check_type(metrics, list)

  for layer in keras_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
      warnings.warn(
          'Batch Normalization contains non-trainable variables that won\'t be '
          'updated during the training. Consider using Group Normalization '
          'instead.', UserWarning)
      break

  return _KerasModel(
      keras_model,
      input_spec=input_spec,
      loss_fns=loss,
      loss_weights=loss_weights,
      metrics=metrics,
      uniform_reg=uniform_reg,
      from_logits=from_logits,
      label_smoothing=label_smoothing,
      batch_majority_voting=batch_majority_voting,
      use_mixed=use_mixed)


@attr.s(frozen=True, slots=True, eq=False)
class BatchOutput():
  """A structure that holds the output of a `tff.learning.Model`.

  Note: All fields are optional (may be None).

  Attributes:
    loss: The scalar mean loss on the examples in the batch. If the model has
      multiple losses, it is the sum of all the individual losses.
    predictions: Tensor of predictions on the examples. The first dimension must
      be the same size (the size of the batch).
    num_examples: Number of examples seen in the batch.
    features: The specified faetures for k-means.
  """
  loss = attr.ib()
  predictions = attr.ib()
  num_examples = attr.ib()
  features = attr.ib()


def kld_no_reduce(preds, kld_fn, from_logits, labels, smoothing, nopad_mask,
                  global_avg):
  """Computes the KL divergence between the prediction and the smoothed label.

  Args:
    preds: The predictions.
    kld_fn: The KLD operator without reduction.
    from_logits: Whether the predictions are logits or softmax scores.
    labels: The one-hot labels.
    smoothing: The eball smoothing coefficient.
    nopad_mask: The mask indicating where it is not padded token.
    global_avg: Whether to do the average over all tokens of the minibatch, or d
      o the sentence-level average.

  Returns:
    Per-sentence label smoothing loss.

  """
  uniform_p = tf.ones_like(preds) / tf.cast(tf.shape(preds)[-1], preds.dtype)
  if smoothing < 1.:
    labels_onehot = tf.one_hot(labels, tf.shape(preds)[-1], dtype=preds.dtype)
    uniform_p = uniform_p * smoothing + tf.cast(
        labels_onehot, dtype=preds.dtype) * (1 - smoothing)
  if from_logits:
    sm_score = tf.nn.softmax(preds, axis=-1)
  else:
    sm_score = preds
  ret = kld_fn(uniform_p, sm_score)
  if global_avg:
    tps = tf.reduce_sum(nopad_mask) / tf.cast(tf.shape(preds)[0], tf.float32)
  else:
    tps = tf.reduce_sum(nopad_mask, axis=1)
  ret = tf.reduce_sum(ret * nopad_mask, axis=1) / tps
  return ret


class _KerasModel(tff.learning.Model):
  """Internal wrapper class for tf.keras.Model objects."""

  def __init__(self,
               keras_model: tf.keras.Model,
               input_spec,
               loss_fns: List[tf.keras.losses.Loss],
               loss_weights: List[float],
               metrics: Union[List[tf.keras.metrics.Metric],
                              List[Callable[[], tf.keras.metrics.Metric]]],
               from_logits: bool = True,
               label_smoothing: float = 1.,
               uniform_reg: float = 0.,
               batch_majority_voting: bool = False,
               pad_token: int = 0,
               use_mixed: bool = False):

    self._keras_model = keras_model
    self._input_spec = input_spec
    self._loss_fns = loss_fns
    self._loss_weights = loss_weights
    self._metrics = metrics
    self._label_smoothing = label_smoothing
    self._uniform_reg = uniform_reg
    self._batch_majority_voting = batch_majority_voting
    self._use_mixed = use_mixed
    self._pad_token = pad_token

    self._reg_kld = tf.keras.losses.KLDivergence()
    self._kld_no_reduce = tf.keras.losses.KLDivergence(
        reduction=tf.keras.losses.Reduction.NONE)
    self._from_logits = from_logits

    self._loss_no_reduction = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=self._from_logits, reduction=tf.keras.losses.Reduction.NONE)

    self._head_acc_metric = tf.keras.metrics.Mean(name='head_sel_acc')
    self._gt_g0_ratio = tf.keras.metrics.Mean(name='gt_g0_ratio')

    self._metrics = []
    self._metric_constructors = []
    if metrics:
      has_keras_metric = False
      has_keras_metric_constructor = False

      for metric in metrics:
        if isinstance(metric, tf.keras.metrics.Metric):
          self._metrics.append(metric)
          has_keras_metric = True
        elif callable(metric):
          constructed_metric = metric()
          if not isinstance(constructed_metric, tf.keras.metrics.Metric):
            raise TypeError(
                f'Metric constructor {metric} is not a no-arg callable that '
                'creates a `tf.keras.metrics.Metric`.')
          self._metric_constructors.append(metric)
          self._metrics.append(constructed_metric)
          has_keras_metric_constructor = True
        else:
          raise TypeError(
              'Expected the input metric to be either a '
              '`tf.keras.metrics.Metric` or a no-arg callable that constructs '
              'a `tf.keras.metrics.Metric`, found a non-callable '
              f'{tff.types.py_typecheck.type_string(type(metric))}.')

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
        super().__init__(name, dtype)
        self._loss_fns = loss_fns
        self._loss_weights = loss_weights

      def update_state(self, y_true, y_pred, sample_weight=None):
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

    self._metrics.append(_WeightedMeanLossMetric())
    if not metrics or self._metric_constructors:
      self._metric_constructors.append(_WeightedMeanLossMetric)

  @property
  def trainable_variables(self):
    return self._keras_model.trainable_variables

  @property
  def non_trainable_variables(self):
    return self._keras_model.non_trainable_variables

  @property
  def local_variables(self):
    local_variables = []
    for metric in self.get_metrics():
      local_variables.extend(metric.variables)  # pytype: disable=attribute-error
    return local_variables

  def get_metrics(self):
    return self._metrics + [self._head_acc_metric, self._gt_g0_ratio]

  @property
  def input_spec(self):
    return self._input_spec

  @tf.function
  def predict_on_batch(self, x, training=True):
    return self._keras_model(x, training=training)

  def _forward_training(self, nopad_mask, y_true, pred_1, pred_2, features,
                        groups, main_branch, kmeans_centers, dist_scalar):
    assert main_branch is not None
    tps = tf.reduce_sum(nopad_mask)
    loss_fn = self._loss_fns[0]

    # The label smoothing regularizations.
    reg1 = kld_no_reduce(
        pred_1,
        self._kld_no_reduce,
        from_logits=self._from_logits,
        labels=y_true,
        smoothing=self._label_smoothing,
        nopad_mask=nopad_mask,
        global_avg=True)
    reg2 = kld_no_reduce(
        pred_2,
        self._kld_no_reduce,
        from_logits=self._from_logits,
        labels=y_true,
        smoothing=self._label_smoothing,
        nopad_mask=nopad_mask,
        global_avg=True)

    # Only for computing the accuracy.
    pred_3 = tf.cond(
        tf.math.equal(main_branch, 0), lambda: pred_1, lambda: pred_2)
    reg3 = tf.cond(
        tf.math.equal(main_branch, 0), lambda: tf.reduce_mean(reg2),
        lambda: tf.reduce_mean(reg1))
    head_correct = tf.cast(
        tf.equal(tf.cast(main_branch, groups.dtype), groups), tf.float32)

    self._head_acc_metric.update_state(head_correct)
    loss3 = loss_fn(y_true=y_true, y_pred=pred_3)
    loss3 = tf.reduce_sum(loss3 * nopad_mask) / tps

    batch_loss = loss3 + self._uniform_reg * reg3
    if self._use_mixed:
      # Compute the distance to cluster centers.
      kmeans_centers = tf.expand_dims(kmeans_centers, axis=0)
      kmeans_dists = tf.norm(
          tf.expand_dims(features, axis=1) - kmeans_centers, axis=-1)
      # Assuming only two centers.
      g1_mindist = tf.reduce_min(kmeans_dists[:, :1], axis=1) * dist_scalar
      g2_mindist = tf.reduce_min(kmeans_dists[:, 1:], axis=1)
      # Weigh the predictions by the distances: we want the weight to be
      # smaller for the cluster with larger distance.
      g1_ratio = g2_mindist / (g1_mindist + g2_mindist + 1e-7)
      for _ in range(len(pred_1.shape) - 1):
        g1_ratio = tf.expand_dims(g1_ratio, axis=1)

      predictions = g1_ratio * pred_1 + (1. - g1_ratio) * pred_2
      mixed_loss = loss_fn(y_true=y_true, y_pred=predictions)
      mixed_loss = tf.reduce_sum(mixed_loss * nopad_mask) / tps
      batch_loss = mixed_loss + batch_loss
    else:
      predictions = pred_3

    return predictions, batch_loss

  def _forward_eval(self, nopad_mask, y_true, pred_1, pred_2, features, groups,
                    kmeans_centers, dist_scalar):
    assert kmeans_centers is not None
    tps = tf.reduce_sum(nopad_mask)
    loss_fn = self._loss_fns[0]

    n_centers = tf.shape(kmeans_centers)[0]
    features = tf.expand_dims(features, axis=1)
    kmeans_centers = tf.expand_dims(kmeans_centers, axis=0)
    dists = tf.norm(features - kmeans_centers, axis=-1)
    dists = tf.concat([dists[:, :1] * dist_scalar, dists[:, 1:]], axis=1)
    if self._use_mixed:
      g1_mindist = tf.reduce_min(dists[:, :n_centers // 2], axis=1)
      g2_mindist = tf.reduce_min(dists[:, n_centers // 2:], axis=1)
      sel_head1 = g2_mindist / (g1_mindist + g2_mindist + 1e-7)
    else:
      sel_idxes = tf.argmin(dists, axis=1)
      sel_head1 = tf.cast(
          tf.math.less(sel_idxes,
                       tf.cast(n_centers // 2, dtype=sel_idxes.dtype)),
          dtype=tf.float32)
    for _ in range(len(pred_1.shape) - 1):
      sel_head1 = tf.expand_dims(sel_head1, axis=1)
    if self._batch_majority_voting:
      # TODO(b/193904908): the splitting should be done in dataset
      # creation and preprocessing.
      # We assume each client to come from one distribution, and hence we
      # split the clients in the original dataset in evaluation. The original
      # clients have data from distribution 1 & 2 (marked by groups), and two
      # "virtual" clients are created from a single client: client 1 which
      # has only data from distribution 1, and client 2 has only data from
      # distribution 2. The majority voting are done for client 1 & 2
      # separately.
      gt_1 = tf.cast(
          tf.math.equal(groups, tf.constant(0, dtype=groups.dtype)), tf.float32)
      gt_1 = tf.expand_dims(tf.expand_dims(gt_1, axis=1), axis=1)
      # votes for client 1
      votes_1_1 = tf.reduce_sum(sel_head1 * gt_1)
      votes_1_2 = tf.reduce_sum((1.0 - sel_head1) * gt_1)
      if tf.math.greater(votes_1_1, votes_1_2):
        group_1 = tf.ones_like(sel_head1) * gt_1
      else:
        group_1 = tf.zeros_like(sel_head1)

      # votes for client 2
      votes_2_1 = tf.reduce_sum(sel_head1 * (1. - gt_1))
      votes_2_2 = tf.reduce_sum((1.0 - sel_head1) * (1. - gt_1))
      if tf.math.greater(votes_2_1, votes_2_2):
        group_2 = tf.ones_like(sel_head1) * (1 - gt_1)
      else:
        group_2 = tf.zeros_like(sel_head1)
      sel_head1 = group_1 + group_2

      head_correct = tf.cast(
          tf.math.equal(tf.cast(1. - sel_head1[:, 0, 0], groups.dtype), groups),
          tf.float32)
      self._head_acc_metric.update_state(head_correct)
    predictions = sel_head1 * pred_1 + (1. - sel_head1) * pred_2

    batch_loss = loss_fn(y_true=y_true, y_pred=predictions)
    batch_loss = tf.reduce_sum(batch_loss * nopad_mask) / tps

    return predictions, batch_loss

  def _forward_pass(self,
                    batch_input,
                    training=True,
                    get_head_scores=False,
                    main_branch=None,
                    kmeans_centers=None,
                    dist_scalar=1.):
    if hasattr(batch_input, '_asdict'):
      batch_input = batch_input._asdict()
    if isinstance(batch_input, collections.abc.Mapping):
      inputs = batch_input.get('x')
    else:
      inputs = batch_input[0]
    if inputs is None:
      raise KeyError('Received a batch_input that is missing required key `x`. '
                     'Instead have keys {}'.format(list(batch_input.keys())))

    pred_1, pred_2, features = self.predict_on_batch(inputs, training)

    if isinstance(batch_input, collections.abc.Mapping):
      y_true = batch_input.get('y')
    else:
      y_true = batch_input[1]

    nopad_mask = tf.cast(
        tf.not_equal(inputs[0],
                     tf.constant(self._pad_token, dtype=inputs[0].dtype)),
        dtype=pred_1.dtype)
    if get_head_scores:
      return BatchOutput(
          loss=None,
          predictions=None,
          num_examples=tf.shape(tf.nest.flatten(inputs)[0])[0],
          features=features)

    if training:
      predictions, batch_loss = self._forward_training(nopad_mask, y_true,
                                                       pred_1, pred_2, features,
                                                       inputs[0], main_branch,
                                                       kmeans_centers,
                                                       dist_scalar)
    else:
      predictions, batch_loss = self._forward_eval(nopad_mask, y_true, pred_1,
                                                   pred_2, features, inputs[0],
                                                   kmeans_centers, dist_scalar)

    for metric in self.get_metrics()[:-2]:
      metric.update_state(y_true=y_true, y_pred=predictions)  # pytype: disable=attribute-error
    self._gt_g0_ratio.update_state(batch_input[0][1])
    return tff.learning.model.BatchOutput(
        loss=batch_loss,
        predictions=predictions,
        num_examples=tf.shape(tf.nest.flatten(inputs)[0])[0],
    )

  @tf.function
  def forward_pass(self,
                   batch_input,
                   training=True,
                   get_head_scores=False,
                   main_branch=None,
                   kmeans_centers=None,
                   dist_scalar=1.):
    return self._forward_pass(
        batch_input,
        training=training,
        get_head_scores=get_head_scores,
        main_branch=main_branch,
        kmeans_centers=kmeans_centers,
        dist_scalar=dist_scalar)

  @tf.function
  def report_local_unfinalized_metrics(
      self) -> OrderedDict[str, List[tf.Tensor]]:
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

  def metric_finalizers(
      self
  ) -> OrderedDict[str, tff.learning.metrics.finalizer.KerasMetricFinalizer]:
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
      finalizers[metric.name] = (
          tff.learning.metrics.finalizer.create_keras_metric_finalizer(metric))  # pytype: disable=attribute-error
    return finalizers
