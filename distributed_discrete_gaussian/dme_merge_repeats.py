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
"""Helper script for merging and plotting parallel DME results."""
import os
import pprint
import sys

import matplotlib.pyplot as plt
import numpy as np
from numpy import array  # pylint: disable=unused-import.
from numpy import float32  # pylint: disable=unused-import.
import scipy.stats


def mean_confidence_interval(data, confidence=0.95):
  """Compute confidence interval for data with shape (repeat, len(x-axis))."""
  n = len(data)
  m, se = np.mean(data, axis=0), scipy.stats.sem(data, axis=0)
  h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
  return m, m - h, m + h


def plot_curve(subplot, epsilons, data, label, linewidth=1, color=None):
  assert len(data.shape) == 2, 'data should be (repeat, len(x-axis))'
  means, lower, upper = mean_confidence_interval(data)
  subplot.plot(epsilons, means, label=label, linewidth=linewidth, color=color)
  subplot.fill_between(
      epsilons, lower, upper, alpha=0.2, edgecolor='face', color=color)


def plot_results(merged_results):
  """Plot the specified DME result dict."""
  use_log = True
  linewidth = 1
  bit2color = {
      8: '#00B945',
      10: '#FF9500',
      12: '#FF2C00',
      14: '#845B97',
      16: '#474747',
      18: '#17becf',
      20: '#FF9500'
  }
  numplots = len(merged_results)
  numplots2width = {1: 3.5, 2: 7, 3: 10}
  _, ax = plt.subplots(
      1,
      numplots,
      figsize=(numplots2width[numplots], 2.3),
      constrained_layout=True)
  ax = np.atleast_1d(ax)

  # Plot results.
  for res, subplot in zip(merged_results, ax):
    n, d = res['n'], res['d']
    epsilons, bits = res['epsilons'], res['bits']
    k_stddevs = res['k_stddevs']
    gauss, distributed_dgauss = res['gauss'], res['distributed_dgauss']

    # Gaussian.
    plot_curve(
        subplot, epsilons, gauss, 'Continuous Gaussian', linewidth=linewidth)

    # DDGauss.
    for bit_index, b in enumerate(bits):
      plot_curve(
          subplot,
          epsilons,
          distributed_dgauss[:, bit_index],
          rf'DDGauss ($B={b}$)',
          linewidth=linewidth,
          color=bit2color[int(b)])

    subplot.set_xlabel(r'$\varepsilon$', size=16)
    subplot.set_xlim(epsilons[0], epsilons[-1])
    subplot.set_ylabel(r'MSE', size=12)

    print('ylim:', subplot.get_ylim())

    subplot.set_yscale('log' if use_log else 'linear')
    subplot.set_title(rf'$n={n},\quad d={d},\quad k={k_stddevs}$', fontsize=14)
    subplot.tick_params(axis='both', labelsize=12)
    subplot.grid(True)
    subplot.legend(loc='best', prop={'size': 7})

  plt.show()


def main():
  args = sys.argv
  prefix = args[1]
  print('Globbing prefix:')
  dirname, fname_prefix = os.path.split(prefix)
  print('Folder:', dirname)
  files = [fn for fn in os.listdir(dirname) if fn.startswith(fname_prefix)]
  print('Parsing the following files:')
  pprint.pprint(files)
  print()

  result_repeated = []
  for fname in files:
    with open(os.path.join(dirname, fname), 'r') as f:
      results = f.read()
      results = eval(results)  # pylint: disable=eval-used.
      result_repeated.append(results)

  # Take any of the result repeated (n, d) list to merge.
  merged_results = result_repeated[0]

  # Looping through (n, d) pairs and concat along the repeat axis.
  for nd_result_index in range(len(merged_results)):
    gauss_results = [
        result_repeated[i][nd_result_index]['gauss']
        for i in range(len(result_repeated))
    ]
    merged_results[nd_result_index]['gauss'] = np.concatenate(
        gauss_results, axis=0)

    ddgauss_results = [
        result_repeated[i][nd_result_index]['distributed_dgauss']
        for i in range(len(result_repeated))
    ]
    merged_results[nd_result_index]['distributed_dgauss'] = np.concatenate(
        ddgauss_results, axis=0)

  results_str = pprint.pformat(merged_results)
  print('Merged results:\n')
  print(results_str)

  print('Plotting results...')
  plot_results(merged_results)


if __name__ == '__main__':
  main()
