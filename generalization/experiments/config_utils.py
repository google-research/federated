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
"""Utility function for defining hyperparameter grids."""

import itertools

from typing import List, Mapping, Sequence, Union


def hyper_grid(
    grid_dict: Mapping[str, Sequence[Union[str, int, float]]]
) -> List[Mapping[str, Union[str, int, float]]]:
  """Converts a param-keyed dict of lists to a list of mapping.

  Args:
    grid_dict: A Mapping from string parameter names to lists of values.

  Returns:
    A list of parameter sweep based on the Cartesian product of
    all options in param_dict.
  """
  return [
      dict(zip(grid_dict, val))
      for val in itertools.product(*grid_dict.values())
  ]
