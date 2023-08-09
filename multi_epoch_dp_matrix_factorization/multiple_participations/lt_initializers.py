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
"""Library of initializers for the LagrangeTerms data structure."""

import os

from jax import numpy as jnp
import tensorflow as tf

from multi_epoch_dp_matrix_factorization.multiple_participations import contrib_matrix_builders
from multi_epoch_dp_matrix_factorization.multiple_participations import lagrange_terms

#

LM_MATRIX_STRING = 'lagrange_multiplier_vector_tensor_pb'
NN_LM_MATRIX_STRING = 'nonneg_multiplier_matrix_tensor_pb'


def init_nonnegative_lagrange_terms_from_path(
    init_matrix_dir, num_epochs, steps_per_epoch
) -> lagrange_terms.LagrangeTerms:
  """Loads initial LangrangeTerms."""
  n = num_epochs * steps_per_epoch
  epoch_contrib_matrix = (
      contrib_matrix_builders.epoch_participation_matrix_all_positive(
          n, num_epochs
      )
  )
  lagrange_multiplier = tf.io.parse_tensor(
      tf.io.read_file(os.path.join(init_matrix_dir, LM_MATRIX_STRING)),
      tf.float64,
  ).numpy()
  nn_mult = tf.io.parse_tensor(
      tf.io.read_file(os.path.join(init_matrix_dir, NN_LM_MATRIX_STRING)),
      tf.float64,
  ).numpy()
  return lagrange_terms.LagrangeTerms(  # pytype: disable=wrong-arg-types  # jnp-array
      lagrange_multiplier=jnp.array(lagrange_multiplier),
      contrib_matrix=epoch_contrib_matrix,
      nonneg_multiplier=nn_mult,
  )


def init_nonnegative_lagrange_terms(
    num_epochs, steps_per_epoch
) -> lagrange_terms.LagrangeTerms:
  """Returns a dual-feasible initialization of the LagrangeTerms."""
  n = num_epochs * steps_per_epoch
  epoch_contrib_matrix = (
      contrib_matrix_builders.epoch_participation_matrix_all_positive(
          n, num_epochs
      )
  )
  # We need to carefully initialize the LagrangeTerms so that the initial
  # u_total() is PD. Further, if using a multiplicative update for the
  # the nonnegative weights (which seems to work well), we need to be
  # completely sure we only assign zeros where we know the optimal multiplier
  # for the non-negativity constraint is zero. Fortunately, the structure of the
  # epoch participation structure means we can do this: we only need to worry
  # about negativity on X[i, j] when i, j are separated by exactly
  # steps_per_epoch, that is, if the same user might participate on both
  # steps i and j. It appears these entries are always zero in the optimal X,
  # with the non-negativity constraint being tight.
  #
  # This results in the initial u_total() being the identity matrix.
  return lagrange_terms.LagrangeTerms(  # pytype: disable=wrong-arg-types  # jnp-array
      lagrange_multiplier=jnp.ones(epoch_contrib_matrix.shape[1]),
      contrib_matrix=epoch_contrib_matrix,
      nonneg_multiplier=(
          epoch_contrib_matrix @ epoch_contrib_matrix.T
          - jnp.eye(epoch_contrib_matrix.shape[0])
      ),
  )
