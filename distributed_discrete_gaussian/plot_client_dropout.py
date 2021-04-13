# Copyright 2021, Google LLC. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generate the plot for privacy degradation."""
import math
import random

from accounting_utils import get_ddgauss_epsilon
from accounting_utils import get_ddgauss_gamma
from accounting_utils import get_ddgauss_noise_stddev
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
  initial_n = 100
  fixed_q = 1.0

  bits = 32
  beta = math.exp(-0.5)
  delta = 1e-5
  dim = 2**20
  k_stddevs = 4
  l2_clip_norm = 1.0
  steps = 1500

  initial_epsilons = [1, 2, 4, 6, 8]
  drop_rates = np.arange(0.0, 0.61, 0.02)

  save_fig = False
  legend_out = True
  width = 4.3

  fig, ax = plt.subplots(1, 1, figsize=(width, 2.3), constrained_layout=True)

  #### Curves within a plot ####
  for initial_eps in initial_epsilons:
    # Get the parameters as if no clients are dropping out.
    initial_gamma = get_ddgauss_gamma(
        q=fixed_q,
        epsilon=initial_eps,
        l2_clip_norm=l2_clip_norm,
        bits=bits,
        num_clients=initial_n,
        dimension=dim,
        delta=delta,
        beta=beta,
        steps=steps,
        k=k_stddevs)

    initial_local_stddev = get_ddgauss_noise_stddev(
        q=fixed_q,
        epsilon=initial_eps,
        l2_clip_norm=l2_clip_norm,
        gamma=initial_gamma,
        beta=beta,
        steps=steps,
        num_clients=initial_n,
        dimension=dim,
        delta=delta)

    print(f'initial_eps = {initial_eps}')
    print(f'initial_gamma = {initial_gamma}')
    print(f'initial_local_stddev = {initial_local_stddev}')

    actual_eps = []
    for drop in drop_rates:
      # Update `n` with client dropouts and recalculate the server epsilon.
      cur_n = int(round(initial_n * (1.0 - drop)))
      eps_after_drop = get_ddgauss_epsilon(
          q=fixed_q,
          noise_stddev=initial_local_stddev,
          l2_clip_norm=l2_clip_norm,
          gamma=initial_gamma,
          beta=beta,
          steps=steps,
          num_clients=cur_n,
          dimension=dim,
          delta=delta)

      print(f'initial_eps={initial_eps}, drop rate = {drop:.3f}, T = {steps}, '
            f'cur_n = {cur_n}: eps={eps_after_drop}')
      actual_eps.append(eps_after_drop)

    ax.plot([r * 100 for r in drop_rates],
            actual_eps,
            linewidth=1.5,
            label=rf'Initial $\varepsilon={initial_eps}$')

  ax.set_title('Server Observed Privacy', fontsize=12)
  if legend_out:
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 11})
  else:
    ax.legend(loc='best')
  ax.grid(True)

  ax.set_xlabel(r'Client dropout rate (\%)', size=12)
  ax.set_xlim(0, max(drop_rates) * 100)  # Percentage
  ax.set_xticks([0, 10, 20, 30, 40, 50, 60])

  ax.set_ylabel(r'$\varepsilon$', size=16)
  ax.set_yticks(list(range(0, 2 * max(initial_epsilons), 4)))

  if save_fig:
    plot_id = random.randint(0, 10000)
    filename = f'dropout-{plot_id:03}.pdf'
    fig.savefig(filename, bbox_inches='tight', pad_inches=0)
    print('Figure saved to', filename)

  plt.show()
