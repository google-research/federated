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
"""Methods for operating the geo classes.

We store the prefix tree using pygtrie objects. Initially we consider user's
coordinate as an (x,y) tuple. We then compute a binary version of this tuple,
e.g. (x=12, y=5) => (1100, 0101) creates a prefix: ‘10/11/00/01’. We keep the
counts using vectors with positions corresponding to the ids of the leafs in the
tree. For each leaf we implement a conversion process into either the coordinate
on some level or a region on the lowest level.
"""

import dataclasses
import random
from typing import List, Any
from scipy.stats import norm

import numpy as np
import pygtrie
from tqdm import tqdm

DEFAULT_CHILDREN = ['00', '01', '10', '11']


def get_default_children(has_aux_bit=False, split=None):
  """Returns a quad tree first 4 nodes. If has_aux_bit (boolean) provided expands
  to 2 more bits or a specific pos/neg nodes.
  Args:
    has_aux_bit: a boolean to use additional bit for data, e.g. pos/neg.
    split: specific subset of has_aux_bit (pos/neg).

  Returns:
    A list of nodes to initialize the tree.
  """
  if has_aux_bit:
    if split == 'pos':
      return ['001', '011', '101', '111']
    elif split == 'neg':
      return ['000', '010', '100', '110']
    else:
      return ['000', '001', '010', '011',
              '100', '101', '110', '111']
  else:
    return ['00', '01', '10', '11']


@dataclasses.dataclass
class AlgResult:
  """Main result object.

  Attributes:
    image: resulting reassembled image
    sum_vector: a vector of reports on the tree leaves.
    tree: a prefix trie used to convert the sum_vector into image.
    tree_prefix_list: a reverse prefix matching vector coordinates to the trie.
    split_threshold: threshold parameter used to obtain the current tree.
    grid_contour: image showing the tree leafs locations on the map.
    eps: current value of the epsilon in SecAgg round.
  """
  image: np.ndarray
  sum_vector: np.ndarray
  tree: pygtrie.StringTrie
  tree_prefix_list: List[str]
  split_threshold: float
  grid_contour: np.ndarray
  eps: float
  pos_image: np.ndarray = None
  neg_image: np.ndarray = None
  metric: Any = None
  sampled_metric: Any = None
  level_animation_list: List = None


def coordinates_to_binary_path(xy_tuple, depth=10):
  """Transform a coordinate tuple into a binary vector.

  We compute a binary version of the provided coordinate tuple,
  e.g. (x=12, y=5) => (1100, 0101) creates a prefix: ‘10/11/00/01’.

  Args:
    xy_tuple: a tuple of (x,y) coordinates of the user location.
    depth: desired length of the binary vector, e.g. max depth of the tree.

  Returns:
    binary version of the coordinate.
  """
  has_aux_bit = ''
  if len(xy_tuple) == 2:
    x_coord, y_coord = xy_tuple
  else:
    x_coord, y_coord, has_aux_bit = xy_tuple
  path = ''
  for j in reversed(range(depth)):
    path += f'{(x_coord >> j) & 1}{(y_coord >> j) & 1}{has_aux_bit}/'
  path = path[:-1]
  
  return path


def binary_path_to_coordinates(path):
  """Using tree path to the leaf node retrieve (x, y) coordinates.

  Reassembles the path into coordinates. Note that if the path is shorter,
  e.g. for leafs closer to the tree root, the (x, y) coordinates would be
  w.r.t. to the image of the size 2^b x 2^b, where b = `path coordinate bits`.
  Args:
    path: binary path of the location ('00/01')

  Returns:
    x coordinate, y coordinate, total bit level, pos
  """

  x = 0
  y = 0
  pos = None
  splitted_path = path.split('/')
  for xy in splitted_path:
    x = x << 1
    y = y << 1
    x += int(xy[0])
    y += int(xy[1])
    if len(xy) > 2:
      pos = int(xy[2])
  return x, y, len(splitted_path), pos


def report_coordinate_to_vector(xy, tree, tree_prefix_list, count_min):
  """Converts a coordinate tuple into a one-hot vector using tree."""
  path = coordinates_to_binary_path(xy)
  (sub_path, value) = tree.longest_prefix(path)
  if count_min:
    count_min.add(sub_path)
    # print(sub_path, sketch.query(sub_path))
    vector = count_min.get_matrix()
  else:
    vector = np.zeros([len(tree_prefix_list)])

    vector[value] += 1
  return vector


def init_tree(has_aux_bit=False):
  """Initializes tree to have four leaf nodes.

  Creates pgtrie with leafs from `DEFAULT_CHILDREN` and assigns each node
  a positional identifier using positions from the `DEFAULT_CHILDREN`.

  Args:
    has_aux_bit: Whether to account for pos and neg users.

  Returns:
    constructed pygtrie, reverse prefix of the trie.
  """

  new_tree = pygtrie.StringTrie()

  for i, z in enumerate(get_default_children(has_aux_bit)):
    new_tree[z] = i
  return new_tree, list(get_default_children(has_aux_bit))


def transform_region_to_coordinates(x_coord,
                                    y_coord,
                                    prefix_len,
                                    image_bit_level=10):
  """Transforms (x,y)-bit region into a square for a final level.

  This method converts a leaf on some level `prefix_len` to a square region at
  the final level `2^image_bit_level`. For example, a first leaf on the
  smallest prefix 2x2 will occupy (0:512, 0:512) region of the 10-bit image.

  Args:
    x_coord:
    y_coord:
    prefix_len:
    image_bit_level:

  Returns:
    A square region coordinates.
  """

  shift = image_bit_level - prefix_len
  x_bot = x_coord << shift
  x_top = ((x_coord + 1) << shift) - 1
  y_bot = y_coord << shift
  y_top = ((y_coord + 1) << shift) - 1
  return (x_bot, x_top, y_bot, y_top)


def rebuild_from_vector(vector, tree, image_size, contour=False, split_threshold=0,
                        has_aux_bit=False, count_min=None):
  """Using coordinate vector and the tree produce a resulting image.

  For each value in the vector it finds the corresponding prefix and plots the
  value of the vector on a square region of the final image.

  Args:
    vector: data vector from the accumulated responses.
    tree: current tree object
    image_size: desired final resolution of the image.
    contour: release only the contours of the grid (for debugging)
    split_threshold: reduces noise by setting values below threshold to 0.
    has_aux_bit: produce two images with positive and negative cases.
    count_min: use count min sketch.

  Returns:
    image of the size `image_size x image_size`
  """
  image_bit_level = int(np.log2(image_size))
  current_image = np.zeros([image_size, image_size])
  pos_image, neg_image = None, None
  if has_aux_bit:
    pos_image = np.zeros([image_size, image_size])
    neg_image = np.zeros([image_size, image_size])
  for path in sorted(tree):
    if count_min:
      value = count_min.query(path)
    else:
      value = vector[tree[path]]
    (x, y, prefix_len, pos) = binary_path_to_coordinates(path)
    (x_bot, x_top, y_bot,
     y_top) = transform_region_to_coordinates(x, y, prefix_len,
                                              image_bit_level)

    if value < split_threshold:
      value = 0
    count = value / 2 ** (1 * (image_bit_level - prefix_len))

    # Build a grid image without filling the regions.
    if contour:
      current_image[x_bot:x_top + 1,
      y_bot - max(1, 5 // prefix_len):y_bot + max(1, 5 // prefix_len)] = 1
      current_image[x_bot:x_top + 1,
      y_top - max(1, 5 // prefix_len):y_top + 10 // prefix_len] = 1
      current_image[
      x_bot - max(1, 5 // prefix_len):x_bot + 10 // prefix_len,
      y_bot:y_top + 1] = 1
      current_image[
      x_top - max(1, 5 // prefix_len):x_top + 10 // prefix_len,
      y_bot:y_top + 1] = 1
    else:
      # balance for collapsing
      depth = len(path.split('/'))
      keys = tree.keys(path)[1:]
      if len(keys) > 1:
        for key in keys:
          sub_node_depth = len(key.split('/')) - depth
          scale = 4**sub_node_depth/ (4**sub_node_depth-1)
          count *= scale

      current_image[x_bot:x_top + 1, y_bot:y_top + 1] = count
      if has_aux_bit:
        if pos == 1:
          pos_image[x_bot:x_top + 1, y_bot:y_top + 1] = count
        elif pos == 0:
          neg_image[x_bot:x_top + 1, y_bot:y_top + 1] = count
        else:
          raise ValueError(f'Not supported: {pos}')
  return current_image, pos_image, neg_image


def append_to_tree(prefix, tree, tree_prefix_list):
  """
  Append new node to the tree.
  Args:
    prefix: new path, e.g. '10/01/10'
    tree: current tree
    tree_prefix_list: list of tree nodes

  Returns:
    1 if the prefix does not exist in tree otherwise 0.
  """
  if not tree.has_key(prefix):
    tree[prefix] = len(tree_prefix_list)
    tree_prefix_list.append(prefix)
    return 1
  else:
    return 0


def split_regions(tree_prefix_list,
                  vector_counts,
                  split_threshold,
                  image_bit_level,
                  collapse_threshold=None,
                  last_result: AlgResult = None,
                  count_min=None,
                  print_output=False):
  """Modify the tree by splitting and collapsing the nodes.

  This implementation collapses and splits nodes of the tree according to
  the received responses of the users. If there are no new nodes discovered
  the finished flag is returned as True.

  Args:
      tree_prefix_list: matches vector id to the tree prefix.
      vector_counts: vector values aggregated from the users.
      split_threshold: threshold value used to split the nodes.
      image_bit_level: stopping criteria once the final resolution is reached.
      collapse_threshold: threshold value used to collapse the nodes.
      last_result: use previous level results to compute conf intervals,
      count_min: use count-min sketch.
      print_output: print results of splitting.
  Returns:
      new_tree, new_tree_prefix_list, num_newly_expanded_nodes
  """
  collapsed = 0
  created = 0
  num_newly_expanded_nodes = 0
  unchanged = 0
  new_tree_prefix_list = list()
  new_tree = pygtrie.StringTrie()
  for i in range(len(tree_prefix_list)):
    if count_min:
      count = count_min.query(tree_prefix_list[i])
    else:
      count = vector_counts[i] if vector_counts else np.inf
    prefix = tree_prefix_list[i]

    # check whether the tree has reached the bottom
    if len(prefix.split('/')) >= image_bit_level:
      continue
    if last_result:
      cond = evaluate_confidence_interval_condition(last_result, prefix, count,
                                                  split_threshold)
    else:
      cond = count > split_threshold
    if cond:
      for child in DEFAULT_CHILDREN:
        new_prefix = f'{prefix}/{child}'
        num_newly_expanded_nodes += append_to_tree(new_prefix, new_tree, new_tree_prefix_list)
    else:
      if collapse_threshold is not None and \
        count <= collapse_threshold and len(prefix) > 2:
        old_prefix = prefix[:-3]
        collapsed += 1
        created += append_to_tree(old_prefix, new_tree, new_tree_prefix_list)
      else:
        unchanged += append_to_tree(prefix, new_tree, new_tree_prefix_list)
  if print_output:
    print(f'New: {num_newly_expanded_nodes}. Collapsed: {collapsed}. ' + \
        f'Created from collapsed: {created}. Unchanged: {unchanged}.')
  return new_tree, new_tree_prefix_list, num_newly_expanded_nodes


def split_regions_aux(tree_prefix_list,
                  vector_counts,
                  split_threshold,
                  image_bit_level,
                  collapse_threshold=None,
                  last_result: AlgResult = None,
                  count_min=None,
                  print_output=False):
  """Use expansion with aux data.

  We check both counts for positive and negative attributes for each location.

  Args:
      tree_prefix_list: matches vector id to the tree prefix.
      vector_counts: vector values aggregated from the users.
      split_threshold: threshold value used to split the nodes.
      image_bit_level: stopping criteria once the final resolution is reached.
      collapse_threshold: threshold value used to collapse the nodes.
      last_result: use previous level results to compute conf intervals,
      count_min: use count-min sketch
      print_output: print results of splitting
  Returns:
      new_tree, new_tree_prefix_list, num_newly_expanded_nodes
  """
  new_tree_prefix_list = list()
  new_tree = pygtrie.StringTrie()
  collapsed = 0
  created = 0
  num_newly_expanded_nodes = 0
  unchanged = 0

  for i in range(0, len(tree_prefix_list), 2):
    if count_min:
      raise ValueError('CountMin is not implemented for Aux data.')
    neg_count = vector_counts[i] if vector_counts else np.inf
    pos_count = vector_counts[i + 1] if vector_counts else np.inf
    neg_prefix = tree_prefix_list[i]
    pos_prefix = tree_prefix_list[i + 1]

    # check whether the tree has reached the bottom
    if len(pos_prefix.split('/')) >= image_bit_level:
      continue
    if last_result:
      p_cond = evaluate_confidence_interval_condition(last_result, pos_prefix,
                                                    pos_count, split_threshold)
      n_cond = evaluate_confidence_interval_condition(last_result, neg_prefix,
                                                    neg_count, split_threshold)
      cond = p_cond and n_cond
    else:
      cond = (pos_count > split_threshold and neg_count > split_threshold)
    if cond:
      neg_child = get_default_children(has_aux_bit=True, split='neg')
      pos_child = get_default_children(has_aux_bit=True, split='pos')
      for j in range(len(pos_child)):
        new_prefix = f'{neg_prefix}/{neg_child[j]}'
        num_newly_expanded_nodes += append_to_tree(new_prefix, new_tree, new_tree_prefix_list)
        new_prefix = f'{pos_prefix}/{pos_child[j]}'
        append_to_tree(new_prefix, new_tree, new_tree_prefix_list)
    else:
      if collapse_threshold is not None and \
          (pos_count < collapse_threshold or neg_count < collapse_threshold) \
          and len(pos_prefix) > 3 and len(neg_prefix) > 3:
        old_prefix = neg_prefix[:-4]
        collapsed += 1
        created += append_to_tree(old_prefix, new_tree, new_tree_prefix_list)
        old_prefix = pos_prefix[:-4]
        append_to_tree(old_prefix, new_tree, new_tree_prefix_list)
      else:
        unchanged += append_to_tree(neg_prefix, new_tree, new_tree_prefix_list)
        append_to_tree(pos_prefix, new_tree, new_tree_prefix_list)
  if print_output:
    print(f'New: {num_newly_expanded_nodes}. Collapsed: {collapsed}. ' + \
        f'Created from collapsed: {created}. Unchanged: {unchanged}.')
  return new_tree, new_tree_prefix_list, num_newly_expanded_nodes


def build_from_sample(samples, total_size):
  """Restores the image from the list of coordinate tuples."""

  image = np.zeros([total_size, total_size])
  for sample in samples:
    x = sample[0]
    y = sample[1]
    image[x, y] += 1
  return image


def quantize_vector(vector, left_bound, right_bound):
  """Modulo clipping of the provided vector."""

  if left_bound > right_bound:
    raise ValueError('Left bound is higher than the right bound.')
  distance = (right_bound - left_bound)
  scale = (vector - left_bound) // distance
  vector -= distance * scale
  return vector


def make_gaussian(image, total_size, fwhm=3, center=None,
                 convert=False, save=False, load=False):
  """ Make a square gaussian kernel.
  size is the length of a side of the square
  fwhm is full-width-half-maximum, which
  can be thought of as an effective radius.
  """
  import torch

  if load:
    return torch.load(f'split_dataset_{fwhm}_{center[0]}_{center[1]}.pt')
  size = image.shape[0]
  x = np.arange(0, size, 1, float)
  y = x[:, np.newaxis]

  if center is None:
    x0 = y0 = size // 2
  else:
    x0 = center[0]
    y0 = center[1]
  hotspot = np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)
  pos_image = np.floor(hotspot * image)
  pos_image = pos_image.astype(int)
  neg_image = image - pos_image

  if convert:
    pos_dataset = convert_to_dataset(pos_image, total_size, value=1)
    neg_dataset = convert_to_dataset(neg_image, total_size, value=0)
    total_dataset = np.concatenate([pos_dataset, neg_dataset])
    res = dict(mask=hotspot, pos_image=pos_image, neg_image=neg_image,
               pos_dataset=pos_dataset, neg_dataset=neg_dataset,
               total_dataset=total_dataset)
    if save:
      torch.save(res, f'split_dataset_{fwhm}_{center[0]}_{center[1]}.pt')
      print(f'Saved to split_dataset_{fwhm}_{center[0]}_{center[1]}.pt')
    return res
  else:
    return dict(mask=hotspot, pos_image=pos_image, neg_image=neg_image)


def convert_to_dataset(image, total_size, value=None):
  if value is not None:
    dataset = np.zeros(image.sum(),
                       dtype=[('x', np.int16), ('y', np.int16),
                              ('pos', np.int8)])
  else:
    dataset = np.zeros(image.sum(),
                       dtype=[('x', np.int16), ('y', np.int16)])
  z = 0
  for i in tqdm(range(total_size), total=total_size):
    for j in range(total_size):
      for _ in range(int(image[i, j])):
        if value is not None:
          dataset[z] = (i, j, value)
        else:
          dataset[z] = (i, j)
        z += 1

  return dataset


def compute_conf_intervals(sum_vector: np.ndarray, level=95):
  conf_intervals = dict()
  conf_interval_weighted = dict()
  z = norm.ppf(1-(1-level/100)/2)

  for i in range(0, sum_vector.shape[0], 2):
    neg_count = sum_vector[i]
    pos_count = sum_vector[i + 1]
    total_clients_on_map = sum_vector.sum()
    total_region = neg_count + pos_count
    if pos_count > 5 and neg_count > 5:
      p = pos_count / total_region
      conf_interval = z * np.sqrt((1 - p) * p / total_region)
      conf_intervals[i] = conf_interval
      conf_interval_weighted[
        i] = conf_interval * total_region / total_clients_on_map

  return conf_intervals, conf_interval_weighted


def evaluate_confidence_interval_condition(last_result, prefix, count, split_threshold):
  """Evaluate whether the confidence interval is smaller than the the threshold.
  We compute confidence interval by comparing a current value in a sub-region
  with its parent region value from the previous level
  Args:
    last_result: a previous level tree results and vector counts
    prefix: current node prefix.
    count: current node count.
    split_threshold: threshold to cutoff confidence interval.

  Returns:
    whether the node satisfies confidence interval threshold.
  """

  (last_prefix, last_prefix_pos) = last_result.tree.longest_prefix(prefix)
  if last_prefix is None:
    cond = False
  else:
    last_count = last_result.sum_vector[last_prefix_pos]
    p = (last_count - count) / last_count
    if p <= 0 or count < 5 or last_count < 5:
      cond = False
    else:
      conf_int = 1.96 * np.sqrt((p * (1 - p) / last_count)) * last_count
      cond = conf_int < split_threshold

  return cond


def make_step(samples, eps, split_threshold, partial,
              prefix_len, dropout_rate, tree, tree_prefix_list,
              noiser, quantize, total_size, has_aux_bit, count_min):
  """

  Args:
    samples: list of user datapoints.
    eps: current privacy budget for the level.
    split_threshold: threshold for current level.
    partial: uses sub-arrays to prevent OOM.
    prefix_len: current level prefix.
    dropout_rate: rate of the dropout from SecAgg round.
    tree: current tree
    tree_prefix_list: list of tree nodes.
    noiser: class for the noise
    quantize: use of quantization
    total_size: size of the location area (e.g. 1024).
    has_aux_bit: each entry in the dataset has also positivity status (x,y,positivity)
    count_min: use count-min sketch

  Returns: AlgResult and contour of the current split.

  """
  samples_len = len(samples)
  level_animation_list = list()
  if count_min:
    round_vector = np.zeros([partial, count_min.d, count_min.w])
    count_min.M = np.zeros([count_min.d, count_min.w], dtype=np.float64)
    sum_vector = count_min.get_matrix()
  else:
    round_vector = np.zeros([partial, prefix_len])
    sum_vector = np.zeros(prefix_len)
  for j, sample in enumerate(tqdm(samples, leave=False)):
    if dropout_rate and random.random() <= dropout_rate:
      continue
    round_vector[j % partial] = report_coordinate_to_vector(
      sample, tree, tree_prefix_list, count_min)
    if j % partial == 0 or j == samples_len - 1:
      round_vector = noiser.apply_noise(round_vector)
      if quantize is not None:

        round_vector = quantize_vector(round_vector,
                                       -2 ** (
                                           quantize - 1),
                                       2 ** (
                                           quantize - 1))
        sum_vector += quantize_vector(
          round_vector.sum(axis=0), -2 ** (quantize - 1),
          2 ** (quantize - 1))
      else:
        sum_vector += round_vector.sum(axis=0)
      if count_min:
        round_vector = np.zeros([partial, count_min.d, count_min.w])
      else:
        round_vector = np.zeros([partial, prefix_len])

      # save 10 frames for each run for animation
      if j % (samples_len//10) == 0 or j == samples_len - 1:
        test_image, _, _ = rebuild_from_vector(
          np.copy(sum_vector), tree, image_size=total_size, split_threshold=split_threshold if eps else -1,
          has_aux_bit=has_aux_bit, count_min=count_min)
        level_animation_list.append(test_image)
  del round_vector
  rebuilder = np.copy(sum_vector)
  if eps:
    threshold_rebuild = split_threshold
  else:
    threshold_rebuild = 0.0

  test_image, pos_image, neg_image = rebuild_from_vector(
    rebuilder, tree, image_size=total_size, split_threshold=threshold_rebuild,
    has_aux_bit=has_aux_bit, count_min=count_min)

  grid_contour, _, _ = rebuild_from_vector(
    sum_vector,
    tree,
    image_size=total_size,
    contour=True,
    split_threshold=threshold_rebuild, count_min=count_min)

  result = AlgResult(
    image=test_image,
    sum_vector=sum_vector,
    tree=tree,
    tree_prefix_list=tree_prefix_list,
    split_threshold=split_threshold,
    grid_contour=grid_contour,
    pos_image=pos_image,
    level_animation_list=level_animation_list,
    neg_image=neg_image,
    eps=eps)

  return result, grid_contour
