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
"""Utilities to plot results.

All the functions expect an already transformed location vector from the
prefix tree into a uniform image. Additionally, this module allows to save
images or gifs to a path and visualize 3d versions of the result.
"""
from typing import List

import imageio
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib import ticker
import numpy as np
from sklearn import metrics as mt

from analytics.location_heatmaps import geo_utils
from analytics.location_heatmaps import metrics


def copy_tmp_file_to_path(filename, path):
  """Uploads the file from local /tmp to a particular path."""

  with open(f'/tmp/{filename}', 'rb') as f:
    with open(f'{path}/{filename}', 'wb') as g:
      g.write(f.read())
  print(f'Saved into: {path}/{filename}')
  return


def plot_it(ax, test_image, eps, total_regions, metric: metrics.Metrics):
  """Main plotting function.

  Computes the statistics from the metrics module and sets them as a
  title over the image in a matplotlib Axes object.

  Args:
    ax: matplotlib axis.
    test_image: transformed location data into a uniform-depth array.
    eps: current round budget (for stats).
    total_regions: total number of leafs in the tree.
    metric: coomputed metric on the image.

  Returns:
    None
  """

  # remove axis labels and ticks
  ax.axes.xaxis.set_visible(False)
  ax.axes.yaxis.set_visible(False)
  eps = f'{eps:.4f}' if eps else 'None'
  ax.set_title(f'eps: {eps}. Regions: {total_regions}' + '\n' +
               f'L1: {metric.l1_distance:.2e}. L2: {metric.l2_distance:.2e}.' +
               '\n' + f'HS: {metric.hotspots_count}. f1: {metric.f1:.3f}' +
               '\n' + f'WS: {metric.wasserstein:.2e}. ' +
               f'Mutual: {metric.mutual_info:.3f}.' + '\n' +
               f'MSE: {metric.mse:.2e}')
  ax.imshow(test_image)
  return


def plot_f1_line(test_image, true_image, total_size, k=2):
  """Plots f1 line for every k percentiles from 0 to 100."""

  test_image = metrics.normalize(metrics.rescale_image(test_image, total_size))
  x_list = list()
  y_list = list()

  for i in range(0, 100, k):
    top_k_test = (test_image > np.percentile(test_image, i))
    top_k_true = (true_image > np.percentile(true_image, i))

    f1 = mt.f1_score(top_k_true.reshape(-1), top_k_test.reshape(-1))
    x_list.append(i)
    y_list.append(f1)

  _, ax = plt.subplots()
  ax.plot(x_list, y_list)
  ax.set_title('F1-line')

  return


def save_gif(images: List[np.ndarray], path, gif_name='gif_map'):
  """Saves gif to path.

  Args:
    images: list of np.arrays to put in gif.
    path: path to upload the result.
    gif_name: name of gif.

  Returns:
    None
  """
  gif_name = f'{gif_name}.gif'

  with open(f'/tmp/{gif_name}', 'w') as f:
    imageio.mimsave(
        f, [image_prepare(image) for image in images],
        format='gif', duration=1.0)
  copy_tmp_file_to_path(gif_name, path=path)


def image_prepare(image):
  """Clip negative values to 0 to reduce noise and use 0 to 255 integers."""
  image[image < 0] = 0
  image *= 255 / image.sum()
  return image.astype(int)


def plot_3d_single(fig, pos, test_image):
  """Convert the image into a 3d density model.

  Takes the figure object and position and plots a 3d version of the image.
  Args:
    fig: figure created using plt.figure() call.
    pos: a position in the matplotlib matrix, e.g. 111 for a single image.
    test_image: produced test_image data.

  Returns:
    None
  """

  vmax = np.max(test_image)
  ax = fig.add_subplot(pos, projection='3d')
  # Make data.
  x_arr = np.arange(0, 1024, 1)
  y_arr = np.arange(0, 1024, 1)
  x_arr, y_arr = np.meshgrid(x_arr, y_arr)

  z_arr = np.copy(test_image)

  surf = ax.plot_surface(
      x_arr,
      y_arr,
      z_arr,
      cmap=cm.get_cmap('Spectral'),
      linewidth=0,
      antialiased=False)

  # Customize the z axis.
  ax.set_zlim(-0.01, vmax)
  ax.zaxis.set_major_locator(ticker.LinearLocator(5))
  ax.zaxis.set_major_formatter(ticker.FormatStrFormatter('%.02f'))

  # Add a color bar which maps values to colors.
  fig.colorbar(surf, shrink=0.5, aspect=5)
  ax.invert_yaxis()
  ax.view_init(70)


def plot_all_3d(per_level_results: List[geo_utils.AlgResult]):
  """Plot 3d versions of the images produced by the algorithm.

  Args:
    per_level_results: a list of per level algorithm results.
  Returns:
    None
  """
  level_count = len(per_level_results)
  fig = plt.figure(figsize=(level_count * 5, 5))
  for i in range(level_count):
    image = per_level_results[i].image
    plot_3d_single(fig, f'1{level_count}{i+1}', image)
  plt.show()
