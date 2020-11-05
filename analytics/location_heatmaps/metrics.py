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
"""Utilities to get metrics.

This module computes F1, MSE, L1, L2 metrics for the image by transforming
the result to be comparable with the original image.
"""

import dataclasses
import numpy as np
from scipy import stats
from sklearn import metrics as mt


@dataclasses.dataclass
class Metrics:
  """Stores obtained metrics.

  Attributes:
    mse: mean squared error.
    l1_distance: L1 distance.
    l2_distance: L2 distance.
    wasserstein: Wasserstein distance (e.g. earth's movers distance).
    hotspots_count: count of the current hotspots.
    f1: f1 score on the discovered hot spots.
    mutual_info: mutual information metric.
  """
  mse: float
  l1_distance: float
  l2_distance: float
  wasserstein: float
  hotspots_count: int
  f1: float
  mutual_info: float


def rescale_image(image: np.ndarray, total_size: int):
  """Scale up the image to a certain size.

  Naive scaling method for a provided image with two dimensions of some
  size perform scaling such that the final image has levelxlevel size. This
  method simply duplicates values into the larger pixels.

  Args:
     image: initial 'unscaled' square-size image (np.array)
     total_size: desired dimension, power of 2, divisible by the image size.

  Returns:
     scaled image array of size total_size x total_size.
  """
  if total_size % image.shape[0] != 0:
    raise ValueError('Provided scale size has to be divisible by image size.')
  if image.shape[0] != image.shape[1]:
    raise ValueError('Provided image needs to have a squared size.')
  scale = int(total_size / image.shape[0])
  new_image = np.zeros([total_size, total_size])
  for i in range(scale):
    for j in range(scale):
      new_image[i::scale, j::scale] = image
  return new_image


def normalize(vector: np.ndarray):
  """Normalizes the np.array to sum up to one and clips negative values to 0."""

  arr = np.copy(vector)
  arr[arr < 0] = 0
  arr = arr / np.sum(arr)
  return arr


def largest_indices(array: np.ndarray, top_k: int):
  """Compute top-k coordinates of the provided array.

  Takes an image as np.array, computes indices of the largest elements, and
  returns the list of the coordinates and an image with the largest elements
  having value 1 and the rest of the image is 0.
  Args:
    array: data array
    top_k: number of elements to select

  Returns:
     list of top k coordinates, zero array except top-k coordinates set to 1.
  """
  flat = array.flatten()
  # find the top-k elements (unsorted) in the flattened array
  indices = np.argpartition(flat, -top_k)[-top_k:]
  # unravel the flattened indices into the image shape
  unraveled = np.unravel_index(indices, array.shape)

  # create a set of coordinates with top-k elements and create an image.
  tuples = set()
  top_k_arr = np.zeros_like(array)
  for i in range(top_k):
    x_coord = unraveled[0][i]
    y_coord = unraveled[1][i]
    tuples.add((x_coord, y_coord))
    top_k_arr[x_coord, y_coord] = 1

  return tuples, top_k_arr


def get_metrics(test_image, true_image, top_k, total_size):
  """Computes multiple different metrics between two images.

  We compute a variety of metrics on the input image: we output L1 and L2
  distances, Wasserstein (earth movers) distance, hotspot count and f1 score for
  the provided TOP-K parameter, and an MSE error. For the correct comparison the
  images are scaled to the same size first,and then compared per coordinate.


  Args:
    test_image: obtained image to obtain the metrics
    true_image: original image to compare against the test_image.
    top_k: parameter to compute top-k hot spots.
    total_size: the size to scale the images to.

  Returns:
    l2 dist, hot spot counts, movers distance, f1-score, l1 dist, mutual info,
    MSE.
  """

  # normalize the input images
  test_image = normalize(rescale_image(test_image, total_size))
  true_image = normalize(rescale_image(true_image, total_size))

  top_k_test, top_k_test_arr = largest_indices(test_image, top_k)
  top_k_true, top_k_true_arr = largest_indices(true_image, top_k)

  l1_distance = np.linalg.norm(true_image - test_image, ord=1)
  l2_distance = np.linalg.norm(true_image - test_image, ord=2)

  mse = mt.mean_squared_error(test_image, true_image)
  top_k_diff = len(top_k_true.intersection(top_k_test))
  wasserstein = stats.wasserstein_distance(
      test_image.reshape(-1), true_image.reshape(-1))
  f1 = mt.f1_score(top_k_true_arr.reshape(-1), top_k_test_arr.reshape(-1))

  mutual = mt.mutual_info_score(true_image.reshape(-1), test_image.reshape(-1))

  metrics = Metrics(l1_distance=l1_distance, l2_distance=l2_distance,
                    mse=mse, f1=f1, wasserstein=wasserstein,
                    hotspots_count=top_k_diff, mutual_info=mutual)

  return metrics
