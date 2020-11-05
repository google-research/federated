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
"""Main file to run experiments.

This algorithm implements an interactive protocol for density estimation over
some location. It uses levels to iteratively zoom in on the regions that pass
selected threshold. This implementation simulates SecAgg rounds and applies to
each contribution differentially private noise. After each level the result is
plotted and the algorithm computes the metrics comparing to the original density
image.
"""

import random
from typing import List

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import tqdm

from analytics.location_heatmaps import geo_utils
from analytics.location_heatmaps import mechanisms
from analytics.location_heatmaps import metrics
from analytics.location_heatmaps import plotting

TOPK = 1000
TOTAL_SIZE = 1024


def get_data(path, crop_tuple=(512, 100, 1536, 1124), total_size=1024):
  """Download the map image.

  Downloads the image from a given path, crops it and transforms into a list
  of individual locations treating each pixel luminosity value as a single
  contribution from the user. Iterates over the image and saves a tuple of
  coordinates (x,y) into a list. Returns an image converted into numpy array and
  a shuffled list of individual coordinates.
  Args:
    path: location of the map, expects a png file.
    crop_tuple: cropping bounds of the image, excpects 4 values.
    total_size: dimension of the image after cropping.

  Returns:
    np.array of the image and a shuffled list of coordinates.
  """

  with open(path) as f:
    image = Image.open(f).convert('L')
  image = image.crop(crop_tuple)
  true_image = np.asarray(image)
  dataset = list()

  for i in tqdm.tqdm(range(total_size), total=total_size):
    for j in range(total_size):
      for _ in range(int(true_image[i, j])):
        dataset.append([i, j])

  random.shuffle(dataset)
  return true_image, dataset


def print_output(text, flag):
  """Simple flag to suppress output."""

  if flag:
    print(text)


def run_experiment(true_image,
                   dataset,
                   level_sample_size=10000,
                   secagg_round_size=10000,
                   threshold=0,
                   collapse_threshold=None,
                   eps_func=lambda x: 1,
                   total_epsilon_budget=None,
                   top_k=TOPK,
                   partial=100,
                   max_levels=10,
                   threshold_func=None,
                   collapse_func=None,
                   total_size=TOTAL_SIZE,
                   min_dp_size=None,
                   dropout_rate=None,
                   output_flag=True,
                   quantize=None,
                   noise_class=mechanisms.GeometricNoise,
                   save_gif=False) -> List[geo_utils.AlgResult]:
  """The main method to run an experiment using TrieHH.

  Args:
      true_image: original image for comparison
      dataset: dataset of user contributions, i.e. coordinates (x,y).
      level_sample_size: Sample size to run at every level for the algorithm.
      secagg_round_size: SecAgg round size to use for the noise generator.
      threshold: Threshold to split the tree leaf into 4 subregions.
      collapse_threshold: collapse node threshold.
      eps_func: function that produces epsilon value for each level, takes
        round_num and count of the tree leafs.
      total_epsilon_budget: Total epsilon-user budget. If the budget is set the
        last round will consume the remaining budget.
      top_k: a parameter to estimate the hotspot percentile, defaults to `TOPK`.
      partial: uses sub-arrays to prevent OOM.
      max_levels: max number of levels to run deep.
      threshold_func: a function to determine threshold takes eps and tree size.
      collapse_func: a function to determine collapse threshold.
      total_size: size of the location area (e.g. 1024).
      min_dp_size: minimim size to reach DP, defaults to `secagg_round_size`.
      dropout_rate: rate of the dropout from SecAgg round.
      output_flag: whether to plot and print or suppress all output.
      quantize: apply quantization to the vectors.
      noise_class: use specific noise, defaults to GeometricNoise.
      save_gif: saves all images as a gif.

  Returns:
      A list of per level geo_utls.AlgResult objects.
  """

  tree, tree_prefix_list = geo_utils.init_tree()
  per_level_results = list()
  finished = False
  sum_vector = None

  spent_budget = 0
  if level_sample_size % secagg_round_size != 0:
    raise ValueError('Sample size cannot be split into SecAgg')
  else:
    print_output(f'Total of {level_sample_size/ secagg_round_size} ' +\
            'SecAgg rounds per level', output_flag)
  # define DP round size
  dp_round_size = min_dp_size if min_dp_size else secagg_round_size
  if threshold and threshold_func:
    raise ValueError('Specify either `threshold` or `threshold_func`.')
  if collapse_threshold and collapse_func:
    raise ValueError('Specify either `collapse_threshold` or `collapse_func`.')

  for i in range(max_levels):
    samples = random.sample(dataset, level_sample_size)
    samples_len = len(samples)
    prefix_len = len(tree_prefix_list)
    # create an image from the sampled data.
    image_sampled = geo_utils.build_from_sample(samples, total_size=total_size)

    if total_epsilon_budget:
      remaining_budget = total_epsilon_budget - spent_budget
      # check budget
      if remaining_budget <= 0.001:
        break
    else:
      remaining_budget = None

    # create candidate eps
    eps = eps_func(i, prefix_len)

    if eps is None:
      noiser = mechanisms.ZeroNoise()
    else:
      # prevent spilling over the budget
      if remaining_budget:
        # last round, no progress in tree, or cannot run at least two rounds.
        if i == max_levels or finished \
            or remaining_budget < 2 * eps * samples_len:
          print_output('Last round. Spending remaining epsilon budget: ' +\
                  f'{remaining_budget}', output_flag)
          eps = remaining_budget / samples_len

      noiser = noise_class(dp_round_size, 1, eps)
    spent_budget += eps * samples_len

    if threshold_func:
      threshold = threshold_func(
          i, prefix_len, eps,
          eps + (total_epsilon_budget - spent_budget) / samples_len)
    if collapse_func:
      collapse_threshold = collapse_func(threshold)
    print_output(
        f'Level: {i}. Threshold: {threshold:.2f}. ' +
        f'Collapse threshold: {collapse_threshold:.2f}', output_flag)

    # to prevent OOM errors we use vectors of size partial.
    round_vector = np.zeros([partial, prefix_len])
    sum_vector = np.zeros(prefix_len)
    for j, sample in enumerate(tqdm.tqdm(samples)):
      if dropout_rate and random.random() <= dropout_rate:
        continue
      round_vector[j % partial] = geo_utils.report_coordinate_to_vector(
          sample, tree, tree_prefix_list)
      if j % partial == 0 or j == samples_len - 1:
        round_vector = noiser.apply_noise(round_vector)
        if quantize is not None:

          round_vector = geo_utils.quantize_vector(round_vector,
                                                   -2**(quantize - 1),
                                                   2**(quantize - 1))
          sum_vector += geo_utils.quantize_vector(
              round_vector.sum(axis=0), -2**(quantize - 1), 2**(quantize - 1))
        else:
          sum_vector += round_vector.sum(axis=0)

        round_vector = np.zeros([partial, prefix_len])
    del round_vector
    rebuilder = np.copy(sum_vector)
    test_image = geo_utils.rebuild_from_vector(
        rebuilder, tree, image_size=total_size, threshold=threshold)
    grid_contour = geo_utils.rebuild_from_vector(
        sum_vector,
        tree,
        image_size=total_size,
        contour=True,
        threshold=threshold)
    result = geo_utils.AlgResult(
        image=test_image,
        sum_vector=sum_vector,
        tree=tree,
        tree_prefix_list=tree_prefix_list,
        threshold=threshold,
        grid_contour=grid_contour,
        eps=eps)

    per_level_results.append(result)

    # compare to true image without sampling error
    if output_flag:
      metric = metrics.get_metrics(
          result.image,
          true_image=image_sampled,
          top_k=top_k,
          total_size=total_size)
      print(f'Level: {i}. MSE without sampling error: {metric.mse:.2e}')

    tree, tree_prefix_list, finished = geo_utils.split_regions(
        tree_prefix_list, sum_vector, threshold, collapse_threshold)
  if output_flag:
    print(f'Total epsilon-users: {spent_budget:.2f} with ' +\
          f'{spent_budget/level_sample_size:.2f} eps per person. ')
    _, ax = plt.subplots(
        1, len(per_level_results), figsize=(len(per_level_results) * 10, 10))
    _, ax_contour = plt.subplots(
        1, len(per_level_results), figsize=(len(per_level_results) * 10, 10))

    for i in range(len(per_level_results)):
      axis = ax[i] if len(per_level_results) > 1 else ax
      result = per_level_results[i]
      metric = metrics.get_metrics(
          test_image=result.image,
          true_image=true_image,
          top_k=top_k,
          total_size=total_size)
      plotting.plot_it(
          ax=axis,
          test_image=result.image,
          eps=result.eps,
          total_regions=len(result.tree_prefix_list),
          metric=metric)
      ax_contour[i].axes.xaxis.set_visible(False)
      ax_contour[i].axes.yaxis.set_visible(False)
      ax_contour[i].imshow(grid_contour)
    if save_gif:
      images = [result.image for result in per_level_results]
      plotting.save_gif(images, path='/gif_image/')

  return per_level_results
