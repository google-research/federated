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
"""Train the model with auditing.

Insert canaries as necessary and train the model, while logging the
test statistics for auditing along with evaluation metrics.
"""

import time

from absl import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from lidp_auditing import auditing_eval
from lidp_auditing import constants
from lidp_auditing import utils


def run_training_with_canaries(
    datasets: constants.DatasetTupleType,
    model: tf.keras.Model,
    canary_type: str,
    num_epochs: int,
    batch_size: int,
    batch_size_test: int,
    learning_rate: float,
    l2_norm_clip: float,
    noise_multiplier: float,
    global_seed: int,
    savefilename: str,
) -> pd.DataFrame:
  """Run the main training loop with canaries."""
  # Logging
  del savefilename, global_seed
  loss_logs = {}
  grad_norm_logs = {}
  accuracy_logs = {}
  train_canary_outputs = {}
  test_canary_outputs = {}
  evaluate_fn = auditing_eval.get_evaluate_fn()

  # Datasets
  train_dataset, test_dataset = datasets[:2]
  canary_dataset_train, canary_dataset_test = datasets[2:]

  # Optimizer: we handle DP utilities separately, use a regular one here
  optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

  # We assume classification models *without* a softmax layer
  vector_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction=tf.keras.losses.Reduction.NONE
  )
  metric = tf.keras.metrics.SparseCategoricalAccuracy()
  logs_df = pd.DataFrame()  # Intialize for correct return type

  for epoch in range(1, 1 + num_epochs):
    logging.warning('Starting epoch %d', epoch)
    avg_loss_per_epoch = 0.0
    l2_norm_per_parameter_epoch = np.zeros(
        len(tf.nest.flatten(model.trainable_variables)), dtype=np.float32
    )
    epoch_start_time = time.time()
    for i, batch in enumerate(train_dataset.shuffle(60000).batch(batch_size)):
      # Process a batch, add canaries, get clipped + noised gradients
      loss, grads_and_vars, l2_norm_per_parameter = process_one_batch(
          batch,
          model,
          vector_loss_fn,
          canary_type,
          l2_norm_clip,
          noise_multiplier,
      )
      # Apply gradients and running loss
      optimizer.apply_gradients(grads_and_vars)
      avg_loss_per_epoch = (i * avg_loss_per_epoch + loss.numpy()) / (i + 1)
      l2_norm_per_parameter_epoch = (
          i * l2_norm_per_parameter_epoch + l2_norm_per_parameter
      ) / (i + 1)

    # End-of-epoch evaluation
    if np.isnan(avg_loss_per_epoch):
      logging.warning('NaNs encountered in epoch %d. Quitting', epoch)
      break
    epoch_end_time = time.time()
    loss_logs[epoch] = avg_loss_per_epoch
    grad_norm_logs[epoch] = l2_norm_per_parameter_epoch
    accuracy_logs[epoch] = evaluate_fn(
        test_dataset.batch(batch_size_test), model, metric
    ).numpy()
    accuracy_end_time = time.time()
    # Compute the canary scores
    train_canary_outputs[epoch] = auditing_eval.evaluate_canary_dataset(
        canary_type,
        canary_dataset_train,
        model,
        vector_loss_fn,
        batch_size_test,
    )
    test_canary_outputs[epoch] = auditing_eval.evaluate_canary_dataset(
        canary_type, canary_dataset_test, model, vector_loss_fn, batch_size_test
    )
    canary_end_time = time.time()
    logging.warning(
        'Epoch %d completed. Accuracy = %.2f percent. '
        'Time %.2f s = (%.2f training + %.2f accuracy + %.2f canary)',
        epoch,
        accuracy_logs[epoch] * 100,
        canary_end_time - epoch_start_time,
        epoch_end_time - epoch_start_time,
        accuracy_end_time - epoch_end_time,
        canary_end_time - accuracy_end_time,
    )
    logs_df = _get_logs_dataframe(
        loss_logs,
        accuracy_logs,
        grad_norm_logs,
        train_canary_outputs,
        test_canary_outputs,
    )
    # logs_df.to_csv(savefilename)
  return logs_df


def _get_logs_dataframe(
    loss_logs,
    accuracy_logs,
    grad_norm_logs,
    train_canary_outputs,
    test_canary_outputs,
):
  """Convert logs to a save format."""
  return pd.DataFrame({
      'Train_loss': loss_logs,
      'Test_accuracy': accuracy_logs,
      'Grad_norm': grad_norm_logs,
      'Train_canary': train_canary_outputs,
      'Test_canary': test_canary_outputs,
  })


def process_one_batch(
    batch: tuple[tf.Tensor, tf.Tensor, tf.Tensor],
    model: tf.keras.Model,
    vector_loss_fn: tf.keras.losses.Loss,
    canary_type: str,
    l2_norm_clip: float,
    noise_multiplier: float,
    return_gradients_before_reduce: bool = False,
) -> tuple[tf.Tensor, list[tuple[tf.Tensor, tf.Tensor]], np.ndarray]:
  """Process one batch (add canaries if necessary) and take a step."""

  var_list = tf.nest.flatten(model.trainable_variables)
  x, y, z = batch  # x: input, y: output, z: canary_info
  batch_size = x.shape[0]
  # we do not use microbatching:
  num_microbatches = tf.constant(batch_size, dtype=tf.float32)

  # Get the per-sample gradient
  out = get_clipped_jacobian_from_predictions(
      x, y, model, vector_loss_fn, l2_norm_clip
  )
  mean_loss, clipped_gradients, l2_norm_per_parameter = out
  # List of same length as var_list
  # clipped_gradients[i] is of shape (batch_size, *var_list[i].shape)

  if canary_type == constants.RANDOM_GRADIENT_CANARY:
    # Replace some gradients with the random noise
    # clipped_gradients = invoke_random_gradient_canary(
    #     clipped_gradients, var_list, z, l2_norm_clip
    # )
    clipped_gradients = invoke_random_gradient_canary_batched(
        clipped_gradients, var_list, z, l2_norm_clip
    )

  if return_gradients_before_reduce:
    # Return per-example clipped gradients
    grads_and_vars = list(zip(clipped_gradients, var_list))
    return mean_loss, grads_and_vars, np.array(l2_norm_per_parameter)

  # Add DP noise
  final_gradients = add_dp_noise_to_gradients(
      clipped_gradients, l2_norm_clip, noise_multiplier, num_microbatches
  )

  grads_and_vars = list(zip(final_gradients, var_list))
  return mean_loss, grads_and_vars, np.array(l2_norm_per_parameter)


@tf.function
def get_clipped_jacobian_from_predictions(
    x: tf.Tensor,
    y: tf.Tensor,
    model: tf.keras.Model,
    vector_loss_fn: tf.keras.losses.Loss,
    l2_norm_clip: float,
) -> tuple[tf.Tensor, list[tf.Tensor], list[tf.Tensor]]:
  """Get the predictions & loss, return clipped per-sample gradients."""
  # Writing this as a tf.function cuts the running time by ~10x:
  # 2X speed up for the jacobian and 5X speed up for the clipping
  logging.warning('******** Tracing jacobian function **********')

  var_list = tf.nest.flatten(model.trainable_variables)  # flat list of tensors
  with tf.GradientTape() as tape:
    prediction = model(x, training=True)
    loss_vector = vector_loss_fn(y, prediction)  # (batch_size,)
  jacobian = tape.jacobian(
      loss_vector, var_list, unconnected_gradients=tf.UnconnectedGradients.NONE
  )
  # List of same length as model.trainable_variables:
  # jacobian[i] is of shape (batch_size, *var_list[i].shape).
  clipped_gradients, l2_norm_per_parameter = tf.vectorized_map(
      lambda g: utils.clip_gradients_vmap(g, l2_norm_clip), jacobian
  )
  # clipped_gradients: same structure as jacobian
  # l2_norm_per_parameter: list of same length as model.trainable_variables:
  # l2_norm_per_parameter[i] is of shape (batch_size,)
  l2_norm_per_parameter = tf.nest.map_structure(
      tf.reduce_mean, l2_norm_per_parameter
  )
  # list of scalars, mean norm of parameter over the batch
  return tf.reduce_mean(loss_vector), clipped_gradients, l2_norm_per_parameter


def add_dp_noise_to_gradients(
    clipped_gradients: list[tf.Tensor],
    l2_norm_clip: float,
    noise_multiplier: float,
    num_microbatches: tf.Tensor,
):
  """Add noise for DP to clipped gradients."""

  def reduce_noise_normalize_batch(g):
    # Sum gradients over all microbatches.
    summed_gradient = tf.reduce_sum(g, axis=0)

    # Add noise to summed gradients.
    noise_stddev = l2_norm_clip * noise_multiplier
    noise = tf.random.normal(
        tf.shape(input=summed_gradient), stddev=noise_stddev
    )
    noised_gradient = tf.add(summed_gradient, noise)

    # Normalize by number of microbatches and return.
    return tf.truediv(noised_gradient, tf.cast(num_microbatches, tf.float32))

  noised_gradients = tf.nest.map_structure(
      reduce_noise_normalize_batch, clipped_gradients
  )
  return noised_gradients


def invoke_random_gradient_canary(
    jacobian: list[tf.Tensor],
    var_list: list[tf.Tensor],
    z: tf.Tensor,
    l2_norm_clip: float,
) -> list[tf.Tensor]:
  """Replace the jacobian corresponding to z != -1 with random normals."""
  if z[z >= 0].shape[0] == 0:
    # No canaries in this batch
    return jacobian
  idx_seed_pairs = [(idx, s) for idx, s in enumerate(z) if s >= 0]
  # logging.warning('Using %d canaries in this round', len(idx_seed_pairs))
  for idx, seed in idx_seed_pairs:
    # Get the noise
    noise = utils.get_random_normal_like(
        var_list, seed, flat_l2_norm=l2_norm_clip
    )
    for i in range(len(jacobian)):  # replace gradient with noise
      # TF version of the in-place update:
      # jacobian[i][idx] = noise[i]
      jacobian[i] = tf.tensor_scatter_nd_update(
          jacobian[i], [[idx]], tf.expand_dims(noise[i], 0)
      )
  return jacobian


def invoke_random_gradient_canary_batched(
    jacobian: list[tf.Tensor],
    var_list: list[tf.Tensor],
    z: tf.Tensor,
    l2_norm_clip: float,
) -> list[tf.Tensor]:
  """Replace the jacobian for z != -1 with random normals in a batch."""
  # Leads to a ~5x improvement in running time.
  # Has to re-trace the function each time a different batch size is given.
  indices = tf.where(z >= 0)  # (num_seeds, 1)
  if indices.shape[0] == 0:
    # no canaries in this batch
    return jacobian
  # logging.warning('Using %d canaries in this round', indices.shape[0])
  # Get the noise
  seeds = z[z >= 0]  # (num_seeds,)
  batched_noise = utils.get_batched_random_normal_like(
      var_list, seeds, tf.constant(l2_norm_clip)
  )  # List of shape [(num_seeds, *var_list[i].shape)]
  for i in range(len(jacobian)):
    # TF version of the in-place update:
    # jacobian[i][indices] = batched_noise[i]
    jacobian[i] = tf.tensor_scatter_nd_update(
        jacobian[i], indices, batched_noise[i]
    )
  return jacobian
