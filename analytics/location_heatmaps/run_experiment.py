# Copyright 2021, Google LLC.
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

import os
from typing import List

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

import geo_utils
import mechanisms
import metrics
import plotting
from sketches import get_count_min_sketch
from config import Config

TOPK = 1000
TOTAL_SIZE = 1024


def get_data(path, crop_tuple=(512, 100, 1536, 1124),
             total_size=1024, save=True):
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

  with open(path, 'rb') as f:
    image = Image.open(f).convert('L')
  image = image.crop(crop_tuple)
  true_image = np.asarray(image)
  if os.path.isfile('dataset.npy'):
    dataset = np.load('dataset.npy')
  else:
    dataset = geo_utils.convert_to_dataset(true_image, total_size)
    if save:
      np.save('dataset', dataset)

  return true_image, dataset


def get_split_data(path):
  dataset = np.load(path)


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
                   eps_func=lambda x, y: 1,
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
                   save_gif=False,
                   positivity=False,
                   start_with_level=0,
                   ignore_start_eps=False,
                   last_result_ci=None,
                   count_min=False) -> List[geo_utils.AlgResult]:
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
      count_min: use count-min sketch.

  Returns:
      A list of per level geo_utls.AlgResult objects.
  """
  config = Config(dataset=dataset,
                  image=true_image,
                  level_sample_size=level_sample_size,
                  secagg_round_size=secagg_round_size,
                  threshold=threshold,
                  collapse_threshold=collapse_threshold,
                  eps_func=eps_func,
                  total_epsilon_budget=total_epsilon_budget,
                  top_k=top_k,
                  partial=partial,
                  max_levels=max_levels,
                  threshold_func=threshold_func,
                  collapse_func=collapse_func,
                  total_size=total_size,
                  min_dp_size=min_dp_size,
                  dropout_rate=dropout_rate,
                  output_flag=output_flag,
                  quantize=quantize,
                  noise_class=noise_class,
                  save_gif=save_gif,
                  positivity=positivity,
                  start_with_level=start_with_level)

  tree, tree_prefix_list = geo_utils.init_tree(positivity)
  per_level_results = list()
  per_level_grid = list()
  finished = False
  sum_vector = None
  print_output(f'positivity: {positivity}', output_flag)

  spent_budget = 0
  remaining_budget = total_epsilon_budget
  if level_sample_size % secagg_round_size != 0:
    raise ValueError('Sample size cannot be split into SecAgg')
  else:
    print_output(f'Total of {level_sample_size / secagg_round_size} ' + \
                 'SecAgg rounds per level', output_flag)
  # define DP round size
  dp_round_size = min_dp_size if min_dp_size else secagg_round_size
  if threshold and threshold_func:
    raise ValueError('Specify either `threshold` or `threshold_func`.')
  if collapse_threshold and collapse_func:
    raise ValueError(
      'Specify either `collapse_threshold` or `collapse_func`.')
  samples = np.random.choice(dataset, level_sample_size, replace=False)
  if count_min:
    count_min_sketch = get_count_min_sketch(depth=20, width=2000)
    sensitivity = 20
  else:
    count_min_sketch = None
    sensitivity = 1

  for i in range(max_levels):

    samples_len = len(samples)
    prefix_len = len(tree_prefix_list)
    # create an image from the sampled data.
    image_sampled = geo_utils.build_from_sample(samples,
                                                total_size=total_size)

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
        if i == max_levels - 1 or finished \
            or remaining_budget < 2 * eps * samples_len:
          print_output(
            'Last round. Spending remaining epsilon budget: ' + \
            f'{remaining_budget}', output_flag)
          eps = remaining_budget / samples_len

      noiser = noise_class(dp_round_size, sensitivity, eps)
      if ignore_start_eps and start_with_level <= i:
        print_output('Ignoring eps spent', flag=output_flag)
      else:
        spent_budget += eps * samples_len

    if threshold_func:
      threshold = threshold_func(
        i, prefix_len, eps,
        (total_epsilon_budget - spent_budget) / samples_len)
    if collapse_func:
      collapse_threshold = collapse_func(threshold)
    print_output(
      f'Level: {i}. Eps: {eps}. Threshold: {threshold:.2f}. Remaining: {remaining_budget / samples_len if remaining_budget is not None else 0:.2f}',
      output_flag)

    # to prevent OOM errors we use vectors of size partial.
    if start_with_level > i:
      tree, tree_prefix_list, finished = geo_utils.split_regions(
        tree_prefix_list=tree_prefix_list,
        vector_counts=None,
        threshold=threshold, image_bit_level=10,
        collapse_threshold=collapse_threshold, positivity=positivity,
        expand_all=True, count_min=count_min)
      print_output(f"Expanding all at the level: {i}.", output_flag)
      continue

    result, grid_contour = geo_utils.make_step(samples, eps, threshold,
                                               partial,
                                               prefix_len, dropout_rate,
                                               tree, tree_prefix_list,
                                               noiser, quantize, total_size,
                                               positivity, count_min=count_min_sketch)

    per_level_results.append(result)
    per_level_grid.append(grid_contour)

    # compare to true image without sampling error
    if positivity:
      im = result.pos_image
    else:
      im = result.image

    metric = metrics.get_metrics(
      im,
      true_image=image_sampled,
      top_k=top_k,
      total_size=total_size)
    result.sampled_metric = metric
    metric = metrics.get_metrics(
      im,
      true_image=true_image,
      top_k=top_k,
      total_size=total_size)
    result.metric = metric
    print_output(
      f'Level: {i}. MSE: {result.sampled_metric.mse:.2e}, without sampling error: {metric.mse:.2e}.',
      output_flag)
    if i == 0 or not last_result_ci:
      last_result = None
    else:
      last_result = per_level_results[i - 1]
    tree, tree_prefix_list, finished = geo_utils.split_regions(
      tree_prefix_list=result.tree_prefix_list, vector_counts=result.sum_vector,
      threshold=threshold, image_bit_level=10,
      collapse_threshold=collapse_threshold, positivity=positivity,
      last_result=last_result)
    if finished:
      break
  if output_flag:
    print(f'Total epsilon-users: {spent_budget:.2f} with ' + \
          f'{spent_budget / level_sample_size:.2f} eps per person. ')
    fig, ax = plt.subplots(
      1, len(per_level_results),
      figsize=(len(per_level_results) * 10, 10))
    _, ax_contour = plt.subplots(
      1, len(per_level_results),
      figsize=(len(per_level_results) * 10, 10))

    for i in range(len(per_level_results)):
      axis = ax[i] if len(per_level_results) > 1 else ax
      axis_contour = ax_contour[i] if len(per_level_results) > 1 else ax_contour
      result = per_level_results[i]
      plotting.plot_it(
        ax=axis,
        test_image=result.pos_image if positivity else result.image,
        eps=result.eps,
        total_regions=len(result.tree_prefix_list),
        metric=result.metric)

      axis_contour.axes.xaxis.set_visible(False)
      axis_contour.axes.yaxis.set_visible(False)
      axis_contour.imshow(per_level_grid[i])
    fig.savefig('results.pdf')
    if save_gif:
      images = [result.image for result in per_level_results]
      plotting.save_gif(images, path='/gif_image/')

  return per_level_results
