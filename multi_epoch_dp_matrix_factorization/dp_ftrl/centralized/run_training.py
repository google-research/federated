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
"""Training binary for example-level DPMatFac experiments."""

import collections
from collections.abc import Sequence
import random

from absl import app
from absl import flags
from absl import logging
import jax
import optax

from multi_epoch_dp_matrix_factorization.dp_ftrl.centralized import data_loaders
from multi_epoch_dp_matrix_factorization.dp_ftrl.centralized import grad_processor_builders
from multi_epoch_dp_matrix_factorization.dp_ftrl.centralized import models
from multi_epoch_dp_matrix_factorization.dp_ftrl.centralized import training_loop

_PREEXISTING_FLAGS = frozenset(iter(flags.FLAGS))

_RUN_NAME = flags.DEFINE_string(
    'run_name',
    'cifar10_example_dpftrl',
    (
        'The name of this run (work unit). Will be'
        'append to  --root_output_dir to separate experiment results.'
    ),
)
RUN_TAGS = flags.DEFINE_string(
    'run_tags',
    '',
    (
        'Unused "hyperparameter" that can be used to tag runs with additional '
        'strings that are useful in filtering and organizing results.'
    ),
)

_DATASET = flags.DEFINE_enum(
    'dataset',
    'cifar10',
    [
        'mnist',
        'fashion_mnist',
        'cifar10',
        'emnist_class',
        'emnist_merge',
        'cifar10_like_synthetic',
    ],
    'Dataset to use.',
)

# Parameters determining the privacy properties of training, and the mechanism
# used. This set of parameters must be sufficient to compute the privacy
# guarantee for the training procedure.
_L2_SENSITIVITY = flags.DEFINE_float(
    'l2_sensitivity',
    1.0,
    (
        'The (global) l2_sensitivity of the mechanism used, under '
        'the participation pattern and number of epochs trained. Assumed to be '
        'known by the caller of this binary.'
    ),
)
_NOISE_MULTIPLIER = flags.DEFINE_float(
    'noise_multiplier',
    0.0,
    (
        'The noise multiplier (that is, multiplier for the l2 sensitivity flag)'
        ' to use while constructing a mechanism. This value will be multiplied'
        ' with the l2_sensitivity to produce the noise multiplier passed down'
        ' to Gaussian mechanism construction.'
    ),
)
_L2_CLIP = flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm.')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 500, 'Batch size.')
_EPOCHS = flags.DEFINE_integer('epochs', 20, 'Number of epochs.')
# Notice that this may fail, since I think we may have been factorizing the
# momentum matrix here. Can check it out later.
_MECHANISM = flags.DEFINE_enum_class(
    'mechanism',
    grad_processor_builders.GradProcessorSpec.NO_PRIVACY,
    grad_processor_builders.GradProcessorSpec,
    'Specification of mechanism to train.',
)

# Optimization parameters.
_MOMENTUM = flags.DEFINE_float('momentum', 0.9, 'Momentum for training')
_LEARNING_RATE = flags.DEFINE_float(
    'learning_rate', 0.05, 'Learning rate for training'
)
_COOLDOWN = flags.DEFINE_bool(
    'cooldown',
    False,
    (
        'Whether to use LR '
        'cooldown for mechanisms for which cooldown is '
        'not baked into the optimized matrix.'
    ),
)

# Operational parameters.
_MODEL_SEED = flags.DEFINE_integer(
    'model_seed', 0, 'Seed used for model generation.'
)
_BATCH_ORDER_SEED = flags.DEFINE_integer(
    'batch_order_seed', 0, 'Seed used for batch order generation.'
)
_NOISE_SEED = flags.DEFINE_integer(
    'noise_seed', 0, 'Seed to use for noise generation.'
)
_ROOT_OUTPUT_DIR = flags.DEFINE_string(
    'root_output_dir',
    '/tmp/dp_matfact',
    'Root output directory',
)
# Note that we don't parameterize by model for now, though conceivably this
# would be desired. Presumably we would prefer, however, to infer from dataset.
# Also notice that we *don't* have a notion of restarts--this will be captured
# by the mechanisms themselves.

# We store the hyperparameter flags in a data structure to be passed as a
# dictionary to the training loop, for writing to storage.
_HPARAM_FLAGS = [f for f in flags.FLAGS if f not in _PREEXISTING_FLAGS]
FLAGS = flags.FLAGS


def _create_optimizer(
    mechanism: grad_processor_builders.GradProcessorSpec,
    lr: float,
    momentum: float,
    cooldown: float,
    num_steps: int,
) -> optax.GradientTransformation:
  """Builds Optax optimizer which interoperates with the specified mechanism."""

  def _mechanism_has_momentum(key: str):
    return 'momentum' in key

  def _mechanism_has_cooldown(key: str):
    return 'cooldown' in key

  if cooldown and not _mechanism_has_cooldown(mechanism.value):
    # An optax implementation of the get_lr_schedule function in
    # ../../../multiple_participations/factorize_multi_epoch_prefix_sum.py.
    transition_steps = num_steps // 4
    cooldown_target = _LEARNING_RATE.value * 0.05
    transition_begin = num_steps - transition_steps + 1
    learning_rate_for_optimizer = optax.linear_schedule(
        init_value=lr,
        end_value=cooldown_target,
        transition_steps=transition_steps,
        transition_begin=transition_begin,
    )
  else:
    learning_rate_for_optimizer = lr

  if _mechanism_has_momentum(mechanism.value):
    momentum_for_optimizer = None
  else:
    momentum_for_optimizer = momentum

  return optax.sgd(
      learning_rate=learning_rate_for_optimizer, momentum=momentum_for_optimizer
  )


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  random.seed(_BATCH_ORDER_SEED.value)
  rng = jax.random.PRNGKey(_MODEL_SEED.value)

  train_data, val_data, test_data, n_classes = data_loaders.get_tfds_data(
      _DATASET.value, _BATCH_SIZE.value
  )
  train_data = data_loaders.shuffle_data_stream(train_data)

  num_steps = len(train_data) * _EPOCHS.value
  optax_optimizer = _create_optimizer(
      _MECHANISM.value,
      _LEARNING_RATE.value,
      _MOMENTUM.value,
      _COOLDOWN.value,
      num_steps,
  )
  hk_model = models.build_vgg_model(
      n_classes=n_classes, dense_size=128, activation_fn=jax.nn.tanh
  )
  hk_model_params = hk_model.init(rng, next(iter(train_data))[0])
  # We have integer labels, so this is crucial :)
  loss_fn = optax.softmax_cross_entropy_with_integer_labels
  dp_grad_processor = grad_processor_builders.build_grad_processor(
      model_params=hk_model_params,
      spec=_MECHANISM.value,
      l2_norm_clip=_L2_CLIP.value,
      l2_clip_noise_multiplier=_L2_SENSITIVITY.value * _NOISE_MULTIPLIER.value,
      steps_per_epoch=len(train_data),
      num_epochs=_EPOCHS.value,
      noise_seed=_NOISE_SEED.value,
      momentum=_MOMENTUM.value,
  )
  logging.info(
      'Beginning training with: %d epochs and batch size: %d',
      _EPOCHS.value,
      _BATCH_SIZE.value,
  )
  training_loop.train(
      train_data=train_data,
      eval_data=val_data,
      test_data=test_data,
      model=hk_model,
      loss_fn=loss_fn,
      batch_grad_processor=dp_grad_processor,
      post_dp_optimizer=optax_optimizer,
      num_epochs=_EPOCHS.value,
      rng=rng,
      root_output_dir=_ROOT_OUTPUT_DIR.value,
      run_name=_RUN_NAME.value,
      hparams_dict=collections.OrderedDict(
          [(name, FLAGS[name].value) for name in _HPARAM_FLAGS]
      ),
  )


if __name__ == '__main__':
  app.run(main)
