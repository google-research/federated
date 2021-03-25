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
"""Shared utilities for federated reconstruction training."""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from reconstruction import reconstruction_utils

# Dataset split strategies.

# The first 1/`split_dataset_proportion` proportion of words is used for
# reconstruction.
SPLIT_STRATEGY_AGGREGATED = 'aggregated'

# Every `split_dataset_proportion` word is used for reconstruction.
SPLIT_STRATEGY_SKIP = 'skip'


def build_list_sample_fn(a: Union[Sequence[Any], int],
                         size: int,
                         replace: bool = False) -> Callable[[int], List[Any]]:
  """Builds the function for sampling from the input iterator at each round.

  Note that round number is ignored here. For more sophosticated (seeded)
  sampling that remains consistently seeded across rounds, see
  `tff.simulation.build_uniform_sampling_fn`.

  Args:
    a: A 1-D array-like sequence or int that satisfies np.random.choice.
    size: The number of samples to return each round.
    replace: A boolean indicating whether the sampling is done with replacement
      (True) or without replacement (False).

  Returns:
    A function which returns a List of elements from the input iterator at a
    given round round_num.
  """

  def sample_fn(_):
    return np.random.choice(a, size=size, replace=replace).tolist()

  return sample_fn


def build_eval_fn(
    evaluation_computation: tff.Computation, client_datasets_fn: Callable[[int],
                                                                          Any],
    get_model: Callable[[Any], tff.learning.ModelWeights]
) -> Callable[[tff.learning.ModelWeights, int], Dict[str, float]]:
  """Creates an evaluation function for use with `training_loop.run`.

  Args:
    evaluation_computation: A `tff.Computation` performing evaluation.
    client_datasets_fn: A function taking in an integer round number and
      returning the expected input of `evaluation_computation`. See
      `tff.simulation.build_uniform_client_sampling_fn` for an example. For
      evaluation, the round number passed is always 0, so this function should
      typically return a different result each time it is called with the same
      argument, e.g. if it is sampling a subset of users from the evaluation
      set.
    get_model: A callable accepting the current server state, and returning a
      `tff.learning.ModelWeights` to be used for evaluation.

  Returns:
    An evaluation function accepting as input a `tff.learning.ModelWeights` and
    an integer `round_num`, and returning a dictionary of evaluation metrics.
  """

  def eval_fn(state: Any, round_num: int) -> Dict[str, float]:
    model = get_model(state)
    sampled_data = client_datasets_fn(round_num)
    return evaluation_computation(model, sampled_data)

  return eval_fn


def build_dataset_split_fn(
    *,  # remaining arguments are keyword-only.
    recon_epochs_max: int,
    recon_epochs_constant: bool,
    recon_steps_max: Optional[int],
    post_recon_epochs: int,
    post_recon_steps_max: Optional[int],
    split_dataset: bool,
    split_dataset_strategy: str,
    split_dataset_proportion: int,
) -> reconstruction_utils.DatasetSplitFn:
  """Builds a `DatasetSplitFn` for Federated Reconstruction training/eval.

  Returned `DatasetSplitFn` parameterizes training and evaluation computations
  and enables reconstruction for multiple local epochs (potentially as a
  function of the server round number), multiple epochs of post-reconstruction
  training, limiting the number of steps for both stages, and splitting client
  datasets into disjoint halves for each stage.

  Note that the returned function is used during both training and evaluation:
  during training, `post-reconstruction` refers to training of global
  variables (possibly jointly with local variables), and during evaluation,
  it refers to calculation of metrics using reconstructed local variables and
  fixed global variables.

  Args:
    recon_epochs_max: The integer maximum number of iterations over the dataset
      to make during reconstruction.
    recon_epochs_constant: If True, use `recon_epochs_max` as the constant
      number of iterations to make during reconstruction. If False, the number
      of iterations is min(round_num, recon_epochs_max).
    recon_steps_max: If not None, the integer maximum number of steps (batches)
      to iterate through during reconstruction. This maximum number of steps is
      across all reconstruction iterations, i.e. it is applied after
      `recon_epochs_max` and `recon_epochs_constant`. If None, this has no
      effect.
    post_recon_epochs: The integer constant number of iterations to make over
      client data after reconstruction.
    post_recon_steps_max: If not None, the integer maximum number of steps
      (batches) to iterate through after reconstruction. This maximum number of
      steps is across all reconstruction iterations, i.e. it is applied after
      `recon_epochs_max` and `recon_epochs_constant`. If None, this has no
      effect.
    split_dataset: If True, splits `client_dataset` for each user, using some
      entries for reconstruction, others after reconstruction, as specified by
      the `split_dataset_strategy`. If False, `client_dataset` is used for both
      reconstruction and post-reconstruction, with the above arguments applied.
      If True, splitting requires that mupltiple iterations through the dataset
      yield the same ordering. For example if
      `client_dataset.shuffle(reshuffle_each_iteration=True)` has been called,
      then the split datasets may have overlap. If True, note that the dataset
      should have more than one batch for reasonable results, since the
      splitting does not occur within batches.
    split_dataset_strategy: The method to use to split the data. Must be one of
      - `skip`, in which case every `split_dataset_proportion` example is used
      for reconstruction. - `aggregated`, when the first
      1/`split_dataset_proportion` proportion of the examples is used for
      reconstruction. This argument is ignored if `split_dataset` is set to
      False.
    split_dataset_proportion: Parameter controlling how much of the data is used
      for reconstruction. If `split_dataset_proportion` is n, then 1 / n of the
      data is used for reconstruction. This argument is ignored if
      `split_dataset` is set to False.

  Returns:
    A `SplitDatasetFn`.
  """

  @tf.function
  def dataset_split_fn(
      client_dataset: tf.data.Dataset,
      round_num: tf.Tensor) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """A `DatasetSplitFn` built with the given arguments.

    Args:
      client_dataset: `tf.data.Dataset` representing client data.
      round_num: Scalar tf.int64 tensor representing the 1-indexed round number
        during training. During evaluation, this is 0.

    Returns:
      A tuple of two `tf.data.Dataset`s, the first to be used for
      reconstruction, the second to be used post-reconstruction.
    """
    get_entry = lambda i, entry: entry
    if split_dataset:
      if split_dataset_strategy == SPLIT_STRATEGY_SKIP:

        def recon_condition(i, _):
          return tf.equal(tf.math.floormod(i, split_dataset_proportion), 0)

        def post_recon_condition(i, _):
          return tf.greater(tf.math.floormod(i, split_dataset_proportion), 0)

      elif split_dataset_strategy == SPLIT_STRATEGY_AGGREGATED:
        num_elements = client_dataset.reduce(
            tf.constant(0.0, dtype=tf.float32), lambda x, _: x + 1)

        def recon_condition(i, _):
          return i <= tf.cast(
              num_elements / split_dataset_proportion, dtype=tf.int64)

        def post_recon_condition(i, _):
          return i > tf.cast(
              num_elements / split_dataset_proportion, dtype=tf.int64)

      else:
        raise ValueError(
            'Unimplemented `split_dataset_strategy`: Must be one of '
            '`{}`, or `{}`. Found {}'.format(SPLIT_STRATEGY_SKIP,
                                             SPLIT_STRATEGY_AGGREGATED,
                                             split_dataset_strategy))
    # split_dataset=False.
    else:
      recon_condition = lambda i, _: True
      post_recon_condition = lambda i, _: True

    recon_dataset = client_dataset.enumerate().filter(recon_condition).map(
        get_entry)
    post_recon_dataset = client_dataset.enumerate().filter(
        post_recon_condition).map(get_entry)

    # Number of reconstruction epochs is exactly recon_epochs_max if
    # recon_epochs_constant is True, and min(round_num, recon_epochs_max) if
    # not.
    num_recon_epochs = recon_epochs_max
    if not recon_epochs_constant:
      num_recon_epochs = tf.math.minimum(round_num, recon_epochs_max)

    # Apply `num_recon_epochs` before limiting to a maximum number of batches
    # if needed.
    recon_dataset = recon_dataset.repeat(num_recon_epochs)
    if recon_steps_max is not None:
      recon_dataset = recon_dataset.take(recon_steps_max)

    # Do the same for post-reconstruction.
    post_recon_dataset = post_recon_dataset.repeat(post_recon_epochs)
    if post_recon_steps_max is not None:
      post_recon_dataset = post_recon_dataset.take(post_recon_steps_max)

    return recon_dataset, post_recon_dataset

  return dataset_split_fn
