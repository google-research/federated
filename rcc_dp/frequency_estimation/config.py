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
"""Base configuration."""

from ml_collections.config_dict import config_dict


def get_config():
  """Returns config dictionary for model."""
  config = dict(
      name="defaults",
      # Either use geometric, zipf, or uniform i.e., data variable
      # can take one of "geometric", "zipf", "uniform".
      distribution="zipf",
      lbd_geometric=0.8,
      degree_zipf=1.0,
      # Flags to indicate which methods to compare.
      run_approx_miracle=False,
      run_miracle=False,
      run_modified_miracle=True,
      run_ss=True,
      run_rhr=True,
      encoding_type="fast", # Can take either fast or normal
      # Common parameters.
      num_itr=1,
      coding_cost=14,
      coding_cost_multiplier=1,
      approx_coding_cost_multiplier=3,
      approx_t=6,
      # Specific parameters (leave them as they are for now).
      delta=10**(-6),
      alpha=1.0,
      # Variation.
      vary="eps",  # Can take one of "cc", "k", "n", "eps".
      cc_space=[6, 8, 10, 12, 14],
      k_space=[200, 400, 600, 800, 1000],
      n_space=[2000, 4000, 6000, 8000, 10000],
      eps_space=list(range(1, 9)),
      # Defaults.
      n=5000,
      k=500,
      t=3,
      epsilon_target=6,
  )
  config = config_dict.ConfigDict(config)
  config.lock()  # Prevent addition of new fields.
  return config