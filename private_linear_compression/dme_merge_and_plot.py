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
import re
from typing import Dict, List

from absl import app
from absl import flags
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style='whitegrid', font_scale=1.15)
sns.set_palette('colorblind')

_INPUT_DIR = flags.DEFINE_string('input_dir', '/tmp/ddp_dme_outputs',
                                 'Input directory.')
_TAG = flags.DEFINE_string('tag', '',
                           'Extra subfolder for the output result files.')
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir', None, 'Output directory. Defaults to `input_dir` if none.')
_OUTPUT_NAME = flags.DEFINE_string('output_name', 'plot', 'name of file.')
_RUN_IDS = flags.DEFINE_multi_integer(
    'run_id', None,
    'IDs of runs to be plotted. If not provided, all runs in the specified path'
    'will be used.')

FLAGS = flags.FLAGS


def plot_results(df: pd.DataFrame) -> plt.Figure:
  """Plot the specified DME results."""
  noise_multiplier_key = 'Noise Multiplier, $z$'
  bit_key = 'Bit Width, $b$'
  compression_key = 'Compression Rate, $r$'
  mse_key = 'Mean Squared Error, MSE'
  relative_mse_key = 'MSE Relative to Central DP'
  repeat_key = 'Repeat ID'
  type_key = 'DP Mechanism'
  vector_dim_key = 'Vector Dimensionality, $d$'

  fig, axes = plt.subplots(1, 1)
  axes = np.atleast_1d(axes).reshape([-1])

  # Generate Relative MSE
  subdf = df.loc[(df[bit_key] == 18), :].copy()
  subdf.loc[:, relative_mse_key] = subdf.loc[:, mse_key]
  for cr in subdf[compression_key].unique():
    for nm in subdf[noise_multiplier_key].unique():
      for run_id in subdf[repeat_key].unique():
        central_select = (df[type_key] == 'Central')
        central_select &= (df[compression_key] == 1.0)
        central_select &= (df[noise_multiplier_key] == nm)
        central_select &= (df[repeat_key] == run_id)
        central_mse = df.loc[central_select, mse_key].squeeze()

        ddp_select = (subdf[compression_key] == cr)
        ddp_select &= (subdf[noise_multiplier_key] == nm)
        ddp_select &= (subdf[repeat_key] == run_id)
        subdf.loc[ddp_select, relative_mse_key] /= central_mse

  # Make line hues scale per the log for easier visualization.
  subdf.loc[:, noise_multiplier_key] = np.log10(subdf.loc[:,
                                                          noise_multiplier_key])

  vector_dim = int(subdf.loc[ddp_select, vector_dim_key].squeeze().mean())
  palette = sns.color_palette('flare', as_cmap=True)

  subplot = sns.lineplot(
      x=compression_key,
      y=relative_mse_key,
      ax=axes[0],
      hue=noise_multiplier_key,
      style=noise_multiplier_key,
      data=subdf,
      legend='full',
      palette=palette)

  subplot.semilogy()

  # Undo log scaling for hues to give legend labels the original value.
  handles, labels = subplot.get_legend_handles_labels()
  labels = [str(10**float(label)) for label in labels]
  subplot.legend(
      handles,
      labels,
      ncol=2,
      facecolor='white',
      framealpha=0.2,
      frameon=True,
      fancybox=True,
      loc='upper left',
      title=noise_multiplier_key)
  x = -.14
  subplot.set_title(
      f'DME of Normalized Gaussian Vectors with $n=100$, $d={vector_dim}$',
      fontsize=14,
      ha='left',
      x=x)
  return fig


def _add_rid_if_in_subset(file_: str, subset_rids: Dict[int, bool],
                          files: List[str]) -> None:
  """Appends inplace `file_` to `files` using `subset_rids` to track."""
  rid_match = re.fullmatch('rid=([0-9]*).pkl', file_)
  if rid_match is not None:
    found_rid = int(rid_match.group(1))
    if found_rid in subset_rids:
      subset_rids[found_rid] = True
      files.append(file_)


def main(_):
  input_dir = os.path.join(_INPUT_DIR.value, _TAG.value)

  if _RUN_IDS.value is None:
    subset_rids = None
  else:
    # subset of RIDs to plot.
    subset_rids = {run_id: False for run_id in _RUN_IDS.value}

  files = []
  for file_ in os.listdir(input_dir):
    if file_.find('.pkl') == -1:  # only .pkls hold results.
      continue
    if subset_rids is not None:
      _add_rid_if_in_subset(file_, subset_rids, files)
    else:
      files.append(file_)
  if subset_rids is not None:
    not_found_rids = [rid for rid, found in subset_rids.items() if not found]
    if not_found_rids:
      raise RuntimeError(
          f'Provided run_ids: {not_found_rids} that were not found.')

  print(f'plotting files: {files}')
  paths = [os.path.join(input_dir, file_) for file_ in files]
  df = pd.concat([pd.read_pickle(path) for path in paths], ignore_index=True)
  fig = plot_results(df)

  if _OUTPUT_DIR.value is None:
    output_dir = input_dir
  else:
    output_dir = _OUTPUT_DIR.value

  outpath = os.path.join(output_dir, '{}.{}')
  fig.savefig(outpath.format(_OUTPUT_NAME.value, 'png'))
  fig.savefig(outpath.format(_OUTPUT_NAME.value, 'pdf'))


if __name__ == '__main__':
  app.run(main)
