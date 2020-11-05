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
"""Methods for operating the geo classes.

We store the prefix tree using pygtrie objects. Initially we consider user's
coordinate as an (x,y) tuple. We then compute a binary version of this tuple,
e.g. (x=12, y=5) => (1100, 0101) creates a prefix: ‘10/11/00/01’. We keep the
counts using vectors with positions corresponding to the ids of the leafs in the
tree. For each leaf we implement a conversion process into either the coordinate
on some level or a region on the lowest level.
"""

from typing import List

import dataclasses
import numpy as np
import pygtrie

DEFAULT_CHILDREN = ['00', '01', '10', '11']


@dataclasses.dataclass
class AlgResult:
  """Main result object.

  Attributes:
    image: resulting reassembled image
    sum_vector: a vector of reports on the tree leaves.
    tree: a prefix trie used to convert the sum_vector into image.
    tree_prefix_list: a reverse prefix matching vector coordinates to the trie.
    threshold: threshold parameter used to obtain the current tree.
    grid_contour: image showing the tree leafs locations on the map.
    eps: current value of the epsilon in SecAgg round.
  """
  image: np.ndarray
  sum_vector: np.ndarray
  tree: pygtrie.StringTrie
  tree_prefix_list: List[str]
  threshold: float
  grid_contour: np.ndarray
  eps: float


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
  x_coord, y_coord = xy_tuple
  path = ''
  for j in reversed(range(depth)):
    path += f'{(x_coord >> j) & 1}{(y_coord >> j) & 1}/'
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
    x coordinate, y coordinate, total bit level.
  """

  x = 0
  y = 0
  splitted_path = path.split('/')
  for xy in splitted_path:
    x = x << 1
    y = y << 1
    x += int(xy[0])
    y += int(xy[1])
  return (x, y, len(splitted_path))


def report_coordinate_to_vector(xy, tree, tree_prefix_list):
  """Converts a coordinate tuple into a one-hot vector using tree."""

  vector = np.zeros([len(tree_prefix_list)])
  path = coordinates_to_binary_path(xy)
  (_, value) = tree.longest_prefix(path)
  vector[value] += 1
  return vector


def init_tree():
  """Initializes tree to have four leaf nodes.

  Creates pgtrie with leafs from `DEFAULT_CHILDREN` and assigns each node
  a positional identifier using positions from the `DEFAULT_CHILDREN`.

  Args:
    None

  Returns:
    constructed pygtrie, reverse prefix of the trie.
  """

  new_tree = pygtrie.StringTrie()

  for i, z in enumerate(DEFAULT_CHILDREN):
    new_tree[z] = i
  return new_tree, list(DEFAULT_CHILDREN)


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


def rebuild_from_vector(vector, tree, image_size, contour=False, threshold=0):
  """Using coordinate vector and the tree produce a resulting image.

  For each value in the vector it finds the corresponding prefix and plots the
  value of the vector on a square region of the final image.

  Args:
    vector: data vector from the accumulated responses.
    tree: current tree object
    image_size: desired final resolution of the image.
    contour: release only the contours of the grid (for debugging)
    threshold: reduces noise by setting values below threshold to 0.

  Returns:
    image of the size `image_size x image_size`
  """
  image_bit_level = int(np.log2(image_size))
  current_image = np.zeros([image_size, image_size])
  for path in sorted(tree):
    value = vector[tree[path]]
    (x, y, prefix_len) = binary_path_to_coordinates(path)
    (x_bot, x_top, y_bot,
     y_top) = transform_region_to_coordinates(x, y, prefix_len, image_bit_level)
    if value < threshold:
      value = 0
    count = value / 2**(2 * (image_bit_level - prefix_len))

    # Build a grid image without filling the regions.
    if contour:
      current_image[x_bot:x_top + 1, y_bot:y_bot + 1] += 1
      current_image[x_bot:x_top + 1, y_top:y_top + 1] += 1
      current_image[x_bot:x_bot + 1, y_bot:y_top + 1] += 1
      current_image[x_top:x_top + 1, y_bot:y_top + 1] += 1
    else:
      current_image[x_bot:x_top + 1, y_bot:y_top + 1] = count
  return current_image


def split_regions(tree_prefix_list,
                  vector_counts,
                  threshold,
                  image_bit_level,
                  collapse_threshold=None):
  """Modify the tree by splitting and collapsing the nodes.

  This implementation collapses and splits nodes of the tree according to
  the received responses of the users. If there are no new nodes discovered
  the finished flag is returned as True.

  Args:
      tree_prefix_list: matches vector id to the tree prefix.
      vector_counts: vector values aggregated from the users.
      threshold: threshold value used to split the nodes.
      image_bit_level: stopping criteria once the final resolution is reached.
      collapse_threshold: threshold value used to collapse the nodes.
  Returns:
      new_tree, new_tree_prefix_list, finished
  """
  collapsed = 0
  created = 0
  fresh_expand = 0
  unchanged = 0
  new_tree_prefix_list = list()
  new_tree = pygtrie.StringTrie()
  for i, count in enumerate(vector_counts):
    prefix = tree_prefix_list[i]

    # check whether the tree has reached the bottom
    if len(prefix.split('/')) < image_bit_level:
      continue

    if count > threshold:
      for child in DEFAULT_CHILDREN:
        new_prefix = f'{prefix}/{child}'
        if not new_tree.has_key(new_prefix):
          fresh_expand += 1
          new_tree[new_prefix] = len(new_tree_prefix_list)
          new_tree_prefix_list.append(new_prefix)
    else:
      if collapse_threshold is not None and\
          count <= collapse_threshold and\
          len(prefix) > 2:

        old_prefix = prefix[:-3]
        collapsed += 1
        if not new_tree.has_key(old_prefix):
          created += 1
          new_tree[old_prefix] = len(new_tree_prefix_list)
          new_tree_prefix_list.append(old_prefix)
      else:
        unchanged += 1
        new_tree[f'{prefix}'] = len(new_tree_prefix_list)
        new_tree_prefix_list.append(f'{prefix}')
  finished = False
  if collapse_threshold:
    print(f'Collapsed: {collapsed}, created when collapsing: {created},' +\
          f'new expanded: {fresh_expand},' +\
          f'unchanged: {unchanged}, total: {len(new_tree_prefix_list)}')
  if not fresh_expand:  # len(new_tree_prefix_list) <= len(tree_prefix_list):
    finished = True
  return new_tree, new_tree_prefix_list, finished


def build_from_sample(samples, total_size):
  """Restores the image from the list of coordinate tuples."""

  image = np.zeros([total_size, total_size])
  for (x, y) in samples:
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
