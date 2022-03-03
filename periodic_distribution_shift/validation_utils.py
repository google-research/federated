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
"""Creating validation or test functions."""
# TODO(b/193904908): add unit tests.

import collections
from typing import Any, Callable, Union, Optional

import tensorflow as tf
import tensorflow_federated as tff

from periodic_distribution_shift.tasks import dist_shift_task

# Convenience aliases.
SequenceType = tff.types.SequenceType
dataset_reduce = tff.learning.framework.dataset_reduce
optimizer_utils = tff.learning.framework.optimizer_utils


def build_federated_evaluation(
    model_fn: Callable[[], tff.learning.Model],
    broadcast_process: Optional[tff.templates.MeasuredProcess] = None,
    use_experimental_simulation_loop: bool = True,
    k_total: int = 2,
    feature_dim: int = 128,
) -> tff.Computation:
  """Builds the TFF computation for federated evaluation of the given model.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`. This method
      must *not* capture TensorFlow tensors or variables and use them. The model
      must be constructed entirely from scratch on each invocation, returning
      the same pre-constructed model each call will result in an error.
    broadcast_process: A `tff.templates.MeasuredProcess` that broadcasts the
      model weights on the server to the clients. It must support the signature
      `(input_values@SERVER -> output_values@CLIENTS)` and have empty state. If
      set to default None, the server model is broadcast to the clients using
      the default tff.federated_broadcast.
    use_experimental_simulation_loop: Controls the reduce loop function for
      input dataset. An experimental reduce loop is used for simulation.
    k_total: Number of clusters/branches for periodic distribution shift models.
    feature_dim: Dimension for feature embeddings used for clustering in FedTKM.

  Returns:
    A federated computation (an instance of `tff.Computation`) that accepts
    model parameters and federated data, and returns the aggregated evaluation
    metrics.
  """
  if broadcast_process is not None:
    if not isinstance(broadcast_process, tff.templates.MeasuredProcess):
      raise ValueError('`broadcast_process` must be a `MeasuredProcess`, got '
                       f'{type(broadcast_process)}.')
    if optimizer_utils.is_stateful_process(broadcast_process):
      raise ValueError(
          'Cannot create a federated evaluation with a stateful '
          'broadcast process, must be stateless, has state: '
          f'{broadcast_process.initialize.type_signature.result!r}')
  # Construct the model first just to obtain the metadata and define all the
  # types needed to define the computations that follow.
  # TODO(b/124477628): Ideally replace the need for stamping throwaway models
  # with some other mechanism.
  with tf.Graph().as_default():
    model = model_fn()
    model_weights_type = tff.learning.framework.weights_type_from_model(model)
    batch_type = tff.types.to_type(model.input_spec)
    kmeans_centers_type = tff.types.type_from_tensors(
        tf.zeros([k_total, feature_dim]))
    dist_scalar_type = tff.types.type_from_tensors(tf.constant(1., tf.float32))
    unfinalized_metrics_type = tff.framework.type_from_tensors(
        model.report_local_unfinalized_metrics())
    metrics_aggregation_computation = tff.learning.metrics.sum_then_finalize(
        model.metric_finalizers(), unfinalized_metrics_type)

  @tff.tf_computation(model_weights_type, kmeans_centers_type, dist_scalar_type,
                      SequenceType(batch_type))
  @tf.function
  def client_eval(incoming_model_weights, kmeans_centers, dist_scalar, dataset):
    """Returns local outputs after evaluting `model_weights` on `dataset`."""
    with tf.init_scope():
      model = model_fn()
    model_weights = tff.learning.ModelWeights.from_model(model)
    tf.nest.map_structure(lambda v, t: v.assign(t), model_weights,
                          incoming_model_weights)

    def reduce_fn(num_examples, batch):
      model_output = model.forward_pass(
          batch,
          training=False,
          kmeans_centers=kmeans_centers,
          dist_scalar=dist_scalar)
      if model_output.num_examples is None:
        # Compute shape from the size of the predictions if model didn't use the
        # batch size.
        return num_examples + tf.shape(
            model_output.predictions, out_type=tf.int64)[0]
      else:
        return num_examples + tf.cast(model_output.num_examples, tf.int64)

    dataset_reduce_fn = dataset_reduce.build_dataset_reduce_fn(
        use_experimental_simulation_loop)
    num_examples = dataset_reduce_fn(
        reduce_fn=reduce_fn,
        dataset=dataset,
        initial_state_fn=lambda: tf.zeros([], dtype=tf.int64))
    return collections.OrderedDict(
        local_outputs=model.report_local_unfinalized_metrics(),
        num_examples=num_examples)

  @tff.federated_computation(
      tff.types.at_server(model_weights_type),
      tff.types.at_server(kmeans_centers_type),
      tff.types.at_server(dist_scalar_type),
      tff.types.at_clients(SequenceType(batch_type)),
  )
  def server_eval(
      server_model_weights,
      kmeans_centers,
      dist_scalar,
      federated_dataset,
  ):
    client_outputs = tff.federated_map(client_eval, [
        tff.federated_broadcast(server_model_weights),
        tff.federated_broadcast(kmeans_centers),
        tff.federated_broadcast(dist_scalar),
        federated_dataset,
    ])
    model_metrics = metrics_aggregation_computation(
        client_outputs.local_outputs)
    statistics = collections.OrderedDict(
        num_examples=tff.federated_sum(client_outputs.num_examples))
    return tff.federated_zip(
        collections.OrderedDict(eval=model_metrics, stat=statistics))

  return server_eval


def create_general_validation_fn(
    task: dist_shift_task.DistShiftTask,
    validation_frequency: int,
    kmeans_eval: bool = False,
    k_total: int = 2,
    feature_dim: int = 128,
) -> Union[Callable[[tff.learning.ModelWeights, Any, Any, int], Any], Callable[
    [tff.learning.ModelWeights, int], Any]]:
  """Creates a function for validating performance of a `tff.learning.Model`.

  Args:
    task: A `DistShiftTask` that defines the model.
    validation_frequency: Frequency of validation,
    kmeans_eval: Boolean to indicate whether to use clusters for multi-branch
      prediction.
    k_total: Number of clusters. Default to 2 for periodic distribution shift.
    feature_dim: Dimension of feature embeddings for clustering.
  Returns:
    A validation function.
  """

  dataset_dict = task.datasets.validation_data_dict

  if kmeans_eval:
    evaluate_fn = build_federated_evaluation(
        task.model_fn,
        use_experimental_simulation_loop=True,
        k_total=k_total,
        feature_dim=feature_dim)

    def validation_fn(model_weights, kmeans_centers, dist_scalar, round_num):
      if round_num % validation_frequency == 0:
        all_metrics = collections.OrderedDict()
        for key, dataset in dataset_dict.items():
          metrics = evaluate_fn(model_weights, kmeans_centers, dist_scalar,
                                [dataset])
          for mkey, mval in metrics.items():
            all_metrics[key + '_' + mkey] = mval
        return all_metrics
      else:
        return {}
  else:
    evaluate_fn = tff.learning.build_federated_evaluation(
        task.model_fn, use_experimental_simulation_loop=True)

    def validation_fn(model_weights, round_num):
      if round_num % validation_frequency == 0:
        all_metrics = collections.OrderedDict()
        for key, dataset in dataset_dict.items():
          metrics = evaluate_fn(model_weights, [dataset])
          for mkey, mval in metrics.items():
            all_metrics[key + '_' + mkey] = mval
        return all_metrics
      else:
        return {}

  return validation_fn


def create_general_test_fn(
    task: dist_shift_task.DistShiftTask,
    kmeans_eval: bool = False,
    k_total: int = 2,
    feature_dim: int = 192,
) -> Union[Callable[[tff.learning.ModelWeights, Any, Any], Any], Callable[
    [tff.learning.ModelWeights], Any]]:
  """Creates a function for testing performance of a `tff.learning.Model`.

  Args:
    task: A `DistShiftTask` that defines the model.
    kmeans_eval: Boolean to indicate whether to use clusters for multi-branch
      prediction.
    k_total: Number of clusters. Default to 2 for periodic distribution shift.
    feature_dim: Dimension of feature embeddings for clustering.
  Returns:
    A validation function.
  """

  test_set = task.datasets.get_centralized_test_data()

  if kmeans_eval:
    evaluate_fn = build_federated_evaluation(
        task.model_fn,
        use_experimental_simulation_loop=True,
        k_total=k_total,
        feature_dim=feature_dim)

    def test_fn(model_weights,
                kmeans_centers,
                dist_scalar=tf.constant(1., tf.float32)):
      return evaluate_fn(model_weights, kmeans_centers, dist_scalar, [test_set])
  else:
    evaluate_fn = tff.learning.build_federated_evaluation(
        task.model_fn, use_experimental_simulation_loop=True)

    def test_fn(model_weights):
      return evaluate_fn(model_weights, [test_set])

  return test_fn
