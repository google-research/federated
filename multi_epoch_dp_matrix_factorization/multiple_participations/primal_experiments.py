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
"""Experiments on the new primal optimization routine.

Unlike factorize_multi_epoch_prefix_sum (which uses a program_state_manager),
this binary is not currently fault tolerant.  If this job fails before
completion, it will not save the optimal factorization to disk, although
xManager metrics will still be logged as long as it was running.  In general,
failed experiments will have to be started over from scratch.  For reasonable
settings of command line arguments, we have not observed any failures.  If
iterations is set too large, out-of-memory errors are possible.
"""
from collections.abc import Callable
import time

from absl import app
from absl import flags
from jax.config import config
import jax.numpy as jnp

from multi_epoch_dp_matrix_factorization import loops
from multi_epoch_dp_matrix_factorization import matrix_constructors
from multi_epoch_dp_matrix_factorization import matrix_io
from multi_epoch_dp_matrix_factorization.multiple_participations import contrib_matrix_builders
from multi_epoch_dp_matrix_factorization.multiple_participations import lagrange_terms
from multi_epoch_dp_matrix_factorization.multiple_participations import optimization
from multi_epoch_dp_matrix_factorization.multiple_participations import primal_optimization

config.update('jax_enable_x64', True)

_WORKLOAD = flags.DEFINE_enum(
    'workload', 'prefix', ['prefix', 'momentum'], 'workload to factorize'
)
_ITERATIONS = flags.DEFINE_integer('iterations', 128, 'Iterations')
_NUM_EPOCHS = flags.DEFINE_integer('num_epochs', 4, 'Number of epochs.')
_DUAL = flags.DEFINE_bool('dual', False, 'Use dual optimizer.')


def momentum_cooldown(n: int) -> jnp.ndarray:
  # Hard-coded for now based on previous experiments
  cooldown_period = n // 4
  cooldown_workload = 0.05
  lr = jnp.ones(n)
  lr = lr.at[-cooldown_period:].set(
      jnp.linspace(1.0, cooldown_workload, num=cooldown_period)
  )
  return matrix_constructors.momentum_sgd_matrix(n, 0.95, lr)  # pytype: disable=bad-return-type  # jnp-array


def dual_optimization(
    workload: jnp.ndarray,
    n: int,
    num_epochs: int,
    metric_callback: Callable[[int, dict[str, float]], None] = print,
) -> jnp.ndarray:
  """Runs the dual optimization algorithm and project to satisfy non-negativity."""
  contrib_matrix = contrib_matrix_builders.epoch_participation_matrix(
      n, num_epochs
  )
  lt = lagrange_terms.init_lagrange_terms(contrib_matrix)
  results = optimization.solve_lagrange_dual_problem(
      s_matrix=workload,
      lt=lt,
      update_langrange_terms_fn=optimization.OptaxUpdate(
          lt=lt, multiplicative_update=True
      ),
      max_iterations=1500,
      iters_per_eval=1,
      target_relative_duality_gap=1e-3,
  )
  encoder_gram = jnp.array(results['x_matrix'])
  encoder_gram = encoder_gram * (encoder_gram > 0)

  sens, _ = optimization.max_min_sensitivity_squared_for_x(encoder_gram, lt)
  encoder_gram = encoder_gram / jnp.sqrt(sens)

  # TODO(b/260247553): perform logging directly in solve_lagrange_dual_problem,
  # e.g., by defining a XmanagerMetricReleaseManager and passing that in.
  for step, (loss, dual) in enumerate(
      zip(results['losses'], results['dual_obj_vals'])
  ):
    metric_callback(step, {'loss': loss, 'dual': dual})

  return encoder_gram


def main(_) -> None:
  n = _ITERATIONS.value
  num_epochs = _NUM_EPOCHS.value
  dual = _DUAL.value
  metric_callback = print

  if _WORKLOAD.value == 'momentum':
    workload = momentum_cooldown(n)
  else:
    workload = jnp.tri(n, dtype=jnp.float64)
  t0 = time.time()
  if dual:
    gram_encoder = dual_optimization(
        workload, n, num_epochs, metric_callback=metric_callback
    )
  else:
    opt = primal_optimization.MatrixFactorizer(workload, epochs=num_epochs)
    gram_encoder = opt.optimize(10000, metric_callback=metric_callback)
  t1 = time.time()

  # This calculation assumes sensitivity(x_matrix) = 1
  # TODO(b/260247553): Consider
  # a) Having optimization return sensitivity of X
  # b) Making helper function that does this
  # c) Asserting that sensitivity is indeed 1
  #   (no *efficient* utility for computing multi-epoch sensitivity, though)
  tse = jnp.trace(workload.T @ workload @ jnp.linalg.inv(gram_encoder))
  metric_callback(-1, {'Expected TSE': float(tse), 'time': t1 - t0})
  epochs_str = '%depochs' % num_epochs if num_epochs else None
  dual_str = 'dual' if dual else None
  mech_str = '_'.join(filter(None, [epochs_str, dual_str]))
  key = 'mech=%s,target=%s' % (mech_str, _WORKLOAD.value)
  path = matrix_io.get_matrix_path(n, key)

  decoder, encoder = loops.w_and_h_from_x_and_s(gram_encoder, workload)
  matrix_io.verify_and_write(  # pytype: disable=wrong-arg-types  # jnp-array
      w_matrix=decoder, h_matrix=encoder, s_matrix=workload, output_dir=path
  )


if __name__ == '__main__':
  app.run(main)
