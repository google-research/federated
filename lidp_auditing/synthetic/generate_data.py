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
"""Generate synthetic Gaussian data.

The canaries `c_j`'s are drawn uniformly at random from a
d-dimensional sphere of radius one.
The DP noise is modeled by `z ~ N(0, sigma^2 I_d)`.

Given a vector `v` as the output of the Gaussian mechanism,
we will track the `k`-dimensional vector `(v . c1, ..., v . c_k)`.
Given a threshold `tau`, our test statistic is
x = ( I(v . c1 >= tau), ..., I(v . c_k >= tau) )

We can vary n, k, d, sigma^2.
"""

import numba
import numpy as np


@numba.njit(parallel=True)
def project_on_unit_sphere_numba(u):
  for i in range(u.shape[0]):
    norm = np.linalg.norm(u[i, :])
    u[i, :] /= norm
  return u


@numba.njit(parallel=True)
def generate_data_numba(
    n: int,
    k: int,
    dim: int,
    seed: int,  # pylint: disable=unused-argument
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Generate synthetic Gaussian data.

  Args:
    n: Number of samples
    k: Number of canaries
    dim: dimension
    seed: random seed

  Returns:
    Tuple of
      - dot product statistic with sum of canaries (TPR)
      - dot product statistic with sum of canaries (FPR)
      - dot product statistic with DP noise z (TPR)
      - dot product statistic with DP noise z (FPR)
  """
  # Random numbers are not generated in a deterministic manner
  # with parallel=True. In fact, it is not clear how thread-safe the seeding
  # process is. I'm not sure but it may be possible that different threads
  # may produce dependent random numbers since they are seeded together
  # initially. Since the process is not deterministic anyway, let us not
  # use the seeds.
  # np.random.seed(seed)
  m = k  # Number of canaries for FPR

  # dot_prod_tpr[i, j] = <v[H1] - z, u_j> for the i^th trial
  dot_prod_tpr = np.zeros((n, k))
  # dot_prod_fpr[i, j] = <v[H0] - z, u_j'> for the i^th trial
  # where u_j' is a canary that was not included during training
  dot_prod_fpr = np.zeros((n, m))

  # Maintain dot product w/ z separately so we can simulate the effect of noise
  # dot_prod_z_tpr[i, j] = <z, u_j> for the i^th trial
  dot_prod_z_tpr = np.zeros((n, k))
  # dot_prod_z_tpr[i, j] = <z, u_j'> for the i^th trial
  dot_prod_z_fpr = np.zeros((n, m))

  # Generate the canaries
  for i in range(n):
    # Uniform on the unit sphere
    u = project_on_unit_sphere_numba(np.random.randn(k, dim))  # (k, dim)
    u1 = project_on_unit_sphere_numba(np.random.randn(m, dim))  # (m, dim)

    # Compute dot products w/ canaries
    v_tpr_minus_z = u.sum(axis=0)  # (dim,) = u_1 + ... + u_k
    dot_prod_tpr[i, :] = np.dot(u, v_tpr_minus_z)  # (k, dim) x (dim,) = (k,)
    v_fpr_minus_z = u[1:].sum(axis=0)  # (dim,) = u_2 + ... + u_k
    dot_prod_fpr[i, :] = np.dot(u1, v_fpr_minus_z)  # (m, dim) x (dim,) = (m,)

    # Compute dot products w/ DP noise
    z = np.random.randn(dim)  # with std = 1
    # NOTE: older code used rand instead of randn
    dot_prod_z_tpr[i] = np.dot(u, z)  # (k,)
    dot_prod_z_fpr[i] = np.dot(u1, z)  # (m,)
  return dot_prod_tpr, dot_prod_fpr, dot_prod_z_tpr, dot_prod_z_fpr
