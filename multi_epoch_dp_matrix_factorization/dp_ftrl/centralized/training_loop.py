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
"""Sketch of integrating DPMatFac with DFT based mechanism for example-level privacy."""

import asyncio
import collections
from collections.abc import Callable, Iterator, Mapping, Sequence
import os
from typing import Any

import haiku as hk
import jax
from jax import numpy as jnp
import numpy as np
import optax
import tensorflow_federated as tff

from multi_epoch_dp_matrix_factorization.dp_ftrl.centralized import gradient_processors
from utils import training_utils

ModelParamsType = Any
PyTree = Any
RNGType = jnp.ndarray
DataIteratorType = Iterator[tuple[np.ndarray, np.ndarray]]
# Full model function, often `apply` property of a transformed Haiku model.
ModelFnType = Callable[[ModelParamsType, RNGType, jnp.ndarray], jnp.ndarray]
# Partially evaluated model function, closing over RNG and parameters.
ModelForwardType = Callable[[jnp.ndarray], jnp.ndarray]

_EPOCHS_PER_CHECKPOINT = 10


def _stream_accuracy(
    model_fn: ModelForwardType, data_stream: DataIteratorType
) -> jnp.ndarray:
  """Computes average accuracy over the data stream."""
  correct_sum, total_sum = 0.0, 0.0
  for batch, labels in data_stream:
    logits = model_fn(batch)
    preds = logits.argmax(axis=-1)
    batch_correct = (preds == labels).sum()
    batch_total = np.reshape(labels, [-1]).shape[0]
    correct_sum += batch_correct
    total_sum += batch_total
  # Ensure we don't nan in the case that we processed no batches.
  return correct_sum / max(1.0, total_sum)


_EVAL_FNS = {'accuracy': _stream_accuracy}


def _save_checkpoint(
    *,
    epoch: int,
    model_params: ModelParamsType,
    batch_grad_processor_state: PyTree,
    optimizer_state: PyTree,
    checkpoint_dir: str,
):
  ckpt_manager = tff.program.FileProgramStateManager(checkpoint_dir)
  asyncio.run(
      ckpt_manager.save(
          (model_params, batch_grad_processor_state, optimizer_state), epoch
      )
  )


def _load_most_recent_checkpoint(
    *,
    initial_model_params: ModelParamsType,
    batch_grad_processor_state: PyTree,
    optimizer_state: PyTree,
    checkpoint_dir: str,
) -> tuple[int, ModelParamsType, PyTree, PyTree]:
  """Loads most recent checkpoint, returning epoch and state."""
  # This is a bit of a weird implementation, to take a TFF dep just to get this
  # utility. But the other options are search way deep in APIs or reinvent
  # wheels (flatten / repackage, ensure containers are traversed correctly,
  # manage the filesystem and provide guarantees, etc). So preferring to do
  # neither of those, and just swallow this direct dep, since we have a
  # transitive one anyway.
  ckpt_manager = tff.program.FileProgramStateManager(checkpoint_dir)
  versions = asyncio.run(ckpt_manager.get_versions())
  if not versions:
    return 0, initial_model_params, batch_grad_processor_state, optimizer_state
  max_version = max(versions)
  return max_version, *asyncio.run(
      ckpt_manager.load(
          max_version,
          (initial_model_params, batch_grad_processor_state, optimizer_state),
      )
  )


def train_one_epoch(
    train_data: Iterator[tuple[np.ndarray, np.ndarray]],
    model_fn: ModelFnType,
    initial_model_params: ModelParamsType,
    loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    batch_grad_processor: gradient_processors.DPGradientBatchProcessor,
    batch_grad_processor_state: PyTree,
    post_dp_optimizer: optax.GradientTransformation,
    optimizer_state: PyTree,
    rng: jnp.ndarray,
    metrics_managers: Sequence[tff.program.ReleaseManager] = (),
    initial_iterate: int = 0,
) -> tuple[ModelParamsType, PyTree, PyTree]:
  """Runs one epoch of training on the given iterator of data.

  This standalone function is useful for testing in isolation, and should be
  kept as minimal as possible.

  Args:
    train_data: Iterator of train data to use.
    model_fn: A JAX-implemented callable which takes parameters, a JAX PRNG, and
      a batch of data on which to predict (represented as a tensor), returning a
      tensor of predictions. Can be assumed to be the `apply` function from a
      Haiku model.
    initial_model_params: The parameters to use as the starting point for the
      model on the given epoch.
    loss_fn: A JAX-implemented function which accepts batches of predictions and
      labels (in that order), returning a loss scalar.
    batch_grad_processor: An instance of
      `gradient_processors.DPGradientBatchProcessor` which will be treated as a
      black box for the purpose of making differentially private a batch of
      per-example gradients, represented as a Python list.
    batch_grad_processor_state: Initial state for the gradient processors in
      this epoch.
    post_dp_optimizer: An `optax` optimizer which will be called with the result
      of the `batch_grad_processor`'s `apply` method to compute the model's
      state at each iteration. This optimizer bears a relationship with the
      operation represented by the `batch_grad_processor`; e.g., if the
      `batch_grad_processor` represents momentum residuals, then this optimizer
      should likely contain no momentum.
    optimizer_state: Initial epoch state for the `post_dp_optimizer`.
    rng: Jax PRNG key to use for the model forward passes.
    metrics_managers: Sequence of `tff.program.ReleaseManagers` to use for
      releasing training metrics.
    initial_iterate: Integer representing the global index of the initial step
      for this epoch.

  Returns:
    A three-tuple of updated model parameters, DP processor state, and optimizer
    state.
  """
  model_params = initial_model_params

  def _single_example_loss(params, batch, labels):
    # TODO(b/244756081): It seems unfortunate to insert this dimension, only to
    # re-squeeze it. The haiku model only works on batched inputs I
    # guess? To revisit and see if we can avoid.
    result = model_fn(params, rng, jnp.expand_dims(batch, 0))
    return jnp.mean(loss_fn(jnp.squeeze(result, 0), labels))

  # The key here is that we must compute per-example grads. Thus the extra
  # work.
  val_and_grad_fn = jax.value_and_grad(_single_example_loss)
  # We map over the batch dimension of the second and third arguments to
  # val_and_grad_fn, which has the same input signature as _single_example_loss.
  per_ex_values_and_grads = jax.jit(
      jax.vmap(val_and_grad_fn, in_axes=[None, 0, 0])
  )

  for idx, (batch, labels) in enumerate(train_data):
    losses, grads = per_ex_values_and_grads(model_params, batch, labels)
    batch_grad_processor_state, dp_grad_estimate = batch_grad_processor.apply(
        batch_grad_processor_state, grads
    )
    updates, optimizer_state = post_dp_optimizer.update(
        dp_grad_estimate, optimizer_state, model_params
    )
    model_params = optax.apply_updates(model_params, updates)

    loss_mean = np.mean(np.array(losses))
    loss_structure = collections.OrderedDict(train_loss=loss_mean)
    loss_type = tff.types.infer_unplaced_type(loss_structure)
    for manager in metrics_managers:
      asyncio.run(
          manager.release(
              value=loss_structure,
              type_signature=loss_type,
              key=idx + initial_iterate,
          )
      )

  return model_params, batch_grad_processor_state, optimizer_state


def evaluate_model(
    model_fn: ModelFnType,
    model_params: ModelParamsType,
    eval_fns: Mapping[
        str,
        Callable[
            [Callable[[jnp.ndarray], jnp.ndarray], DataIteratorType],
            jnp.ndarray,
        ],
    ],
    data: DataIteratorType,
    rng: RNGType,
    epoch: int,
    prefix: str = 'eval',
    metrics_managers: Sequence[tff.program.ReleaseManager] = (),
) -> Mapping[str, jnp.ndarray]:
  """Evaluates given model on provided data, writing results."""
  eval_metrics = collections.OrderedDict()

  def _model_fwd(data):
    return model_fn(model_params, rng, data)

  for k, eval_fn in eval_fns.items():
    # Force into a numpy array here for ease of passing around
    eval_metrics[prefix + '/' + k] = np.array(eval_fn(_model_fwd, data))

  eval_metrics_type = tff.types.infer_unplaced_type(eval_metrics)
  for manager in metrics_managers:
    asyncio.run(
        manager.release(
            value=eval_metrics, type_signature=eval_metrics_type, key=epoch
        )
    )

  return eval_metrics


def train(
    train_data: collections.abc.Collection[tuple[np.ndarray, np.ndarray]],
    eval_data: collections.abc.Collection[tuple[np.ndarray, np.ndarray]],
    test_data: collections.abc.Collection[tuple[np.ndarray, np.ndarray]],
    model: hk.Transformed,
    loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    batch_grad_processor: gradient_processors.DPGradientBatchProcessor,
    post_dp_optimizer: optax.GradientTransformation,
    num_epochs: int,
    rng: jnp.ndarray,
    root_output_dir: str,
    run_name: str,
    hparams_dict: dict[str, float],
):
  """Trains provided haiku model for specified number of epochs.

  Saves state and attempts to load internally to provide fault tolerance.
  Writes results to a TF file summary writer under a 'logdir' directory,
  directly under the `root_output_dir` parameter.

  Args:
    train_data: Collection of tuples of numpy arrays. Iterators derived from
      this collection are passed to the `train_one_epoch` function above.
    eval_data: Collection of tuples of numpy arrays. Iterators derived from this
      collection are passed to the `eval` function above, on epoch end.
    test_data: Collection of tuples of numpy arrays. Iterators derived from this
      collection are passed to the `eval` function above, at training end.
    model: Output of a `hk.transform` call, representing the model to be
      trained.
    loss_fn: Function taking in batched predictions and labels, returning per-
      batch loss.
    batch_grad_processor: An instance of
      `gradient_processors.DPGradientBatchProcessor` which will be treated as a
      black box for the purpose of making differentially private a batch of
      per-example gradients, represented as a Python list.
    post_dp_optimizer: An `optax` optimizer which will be called with the result
      of the `batch_grad_processor`'s `apply` method to compute the model's
      state at each iteration. This optimizer bears a relationship with the
      operation represented by the `batch_grad_processor`; e.g., if the
      `batch_grad_processor` represents momentum residuals, then this optimizer
      should likely contain no momentum.
    num_epochs: The number of epochs for which to train.
    rng: Jax PRNGKey, passed to initialization and forward pass of the model.
    root_output_dir: Root of directory in which to write checkpoints and
      metrics.
    run_name: Unique name of the run. Will be used to separate results under
      root_output_dir.
    hparams_dict: Dictionary of hyperparameters. Will be written to results
      directory for ease of analysis.
  """

  example_batch = next(iter(train_data))[0]
  initial_model_params = model.init(rng, example_batch)
  num_training_steps = len(train_data)

  batch_grad_processor_state = batch_grad_processor.init()
  opt_state = post_dp_optimizer.init(initial_model_params)

  _, metrics_managers = training_utils.create_managers(
      root_output_dir, run_name
  )
  training_utils.write_hparams_to_csv(hparams_dict, root_output_dir, run_name)

  checkpoint_dir = os.path.join(root_output_dir, 'ckpt', run_name)

  epoch, model_params, batch_grad_processor_state, opt_state = (
      _load_most_recent_checkpoint(
          initial_model_params=initial_model_params,
          batch_grad_processor_state=batch_grad_processor_state,
          optimizer_state=opt_state,
          checkpoint_dir=checkpoint_dir,
      )
  )

  while epoch < num_epochs:
    # TODO(b/244756081): If we wanted to we could make all this stuff async,
    # and interleave more effectively. But this may not be worth our time.
    model_params, batch_grad_processor_state, opt_state = train_one_epoch(
        train_data=iter(train_data),
        model_fn=model.apply,
        initial_model_params=model_params,
        loss_fn=loss_fn,
        batch_grad_processor=batch_grad_processor,
        batch_grad_processor_state=batch_grad_processor_state,
        post_dp_optimizer=post_dp_optimizer,
        optimizer_state=opt_state,
        rng=rng,
        metrics_managers=metrics_managers,
        initial_iterate=epoch * num_training_steps,
    )
    epoch += 1
    if epoch % _EPOCHS_PER_CHECKPOINT == 0:
      _save_checkpoint(
          epoch=epoch,
          model_params=model_params,
          batch_grad_processor_state=batch_grad_processor_state,
          optimizer_state=opt_state,
          checkpoint_dir=checkpoint_dir,
      )
    # We evaluate every epoch. For now, we simply close over some default
    # global eval fns, including accuracy.
    evaluate_model(
        model.apply,
        model_params,
        _EVAL_FNS,
        iter(eval_data),
        rng,
        epoch,
        metrics_managers=metrics_managers,
        prefix='eval',
    )

  # End of training; test the model.
  evaluate_model(
      model.apply,
      model_params,
      _EVAL_FNS,
      iter(test_data),
      rng,
      epoch,
      metrics_managers=metrics_managers,
      prefix='test',
  )
