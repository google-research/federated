# Copyright 2023, Google LLC.
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
"""Generate synthetic Gaussian data.

The canaries `c_j`'s are drawn uniformly at random from a
d-dimensional sphere of radius one.
The DP noise is modeled by `z ~ N(0, sigma^2 I_d)`.

Given a vector `v` as the output of the Gaussian mechanism,
we will track the `k`-dimensional vector `(v . c1, ..., v . c_k)`.
Given a threshold `tau`, our test statistic is
x = ( I(v . c1 >= tau), ..., I(v . c_k >= tau) )

We can vary n, k, d, sigma^2.
"""

from collections.abc import Sequence
import datetime
import math
import time

from absl import app
from absl import flags
from absl import logging
import numpy as np
import pandas as pd

from lidp_auditing.synthetic import generate_data
from lidp_auditing.synthetic import process_data

# Threshold type
THRESHOLD_TUNE = 'tune'
THRESHOLD_SAVED = 'saved'
THRESHOLD_REUSE = 'reuse'

# General flags
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    None,
    'The name of the output directory.',
    required=True,
)
_SEED = flags.DEFINE_integer('seed', None, 'seed', required=True)
_OUT_NAME = flags.DEFINE_string(
    'out_name', None, 'prefix of output name', required=True
)
_NUM_SAMPLES = flags.DEFINE_integer(
    'num_samples', None, 'Number of samples', required=True
)
_NUM_CANARIES = flags.DEFINE_integer(
    'num_canaries', None, 'Number of canaries', required=True
)
_DIM = flags.DEFINE_integer('dimension', None, 'Dimension', required=True)
_BETA = flags.DEFINE_float(
    'beta',
    None,
    'Failure probability',
    required=True,
    lower_bound=0,
    upper_bound=1,
)
_THRESHOLD_TYPE = flags.DEFINE_enum(
    'threshold_type',
    None,
    [THRESHOLD_TUNE, THRESHOLD_SAVED, THRESHOLD_REUSE],
    'How to find the right threshold',
    required=True,
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  n = _NUM_SAMPLES.value
  k = _NUM_CANARIES.value
  dim = _DIM.value
  seed = _SEED.value
  beta = _BETA.value
  delta = 1e-5  # default value, fixed
  # These (squared) noise multipliers have been precomputed to correspond to
  # eps = [0.5, 1, 1.5, 2, 3, 4, 8, 16] at delta=1e-5
  sigmasq_list = [58.8, 16.36, 7.788, 4.62, 2.23, 1.34, 0.4066, 0.13135]
  num_candidate_thresholds = 500

  # Check sizes
  if k * dim > 1e9 or n * k > 1e8:
    logging.warning(
        'Maximum size exceeded with n = %d, k = %d, dim = %d', n, k, dim
    )
    return  # Quit

  filename_core = (
      f'{_OUTPUT_DIR.value}/{_OUT_NAME.value}_n{n}_k{k}_dim{dim}_beta{beta}'
  )
  threshold_filename = f'{filename_core}_threshold.csv'  # no seed
  output_filename = f'{filename_core}_seed{seed}_empeps.csv'

  if _THRESHOLD_TYPE.value == THRESHOLD_SAVED:
    threshold_df = pd.read_csv(threshold_filename, index_col=0)
    threshold_df.columns = [float(a) for a in threshold_df.columns]  # convert
    logging.warning('Loaded thresholds from %s', threshold_df)
    # row: estimator, col: sigmasq
  else:
    threshold_df = None

  outputs_for_sigmasq = get_results_for_sigmasq(
      n,
      k,
      dim,
      seed,
      beta,
      delta,
      sigmasq_list,
      threshold_df,
      num_candidate_thresholds,
  )

  if _THRESHOLD_TYPE.value == THRESHOLD_TUNE:
    # Tune the optimal threshold from validation data
    logging.warning('***Threshold filename: %s', threshold_filename)
    # Find the threshold that returns the best epsilon
    threshold_df = pd.DataFrame({
        key: results.idxmax(axis=1)
        for key, results in outputs_for_sigmasq.items()
    })
    threshold_df.to_csv(threshold_filename)
    logging.warning('Saved file %s', threshold_filename)
  else:
    # We will write the output file. Common scaffolding code
    logging.warning('***Output filename: %s', output_filename)

    if _THRESHOLD_TYPE.value == THRESHOLD_REUSE:
      # Use the max over all threshold (no tuning)
      epsilon_df = pd.DataFrame({
          key: results.max(axis=1)
          for key, results in outputs_for_sigmasq.items()
      })
      # columns: sigmasq, row: estimator
      # For testing only, do not use for experiments
      epsilon_df.to_csv(output_filename)
      logging.warning('Saved file %s', output_filename)

    elif _THRESHOLD_TYPE.value == THRESHOLD_SAVED:
      # Load the saved values of the threshold and use them
      # columns: sigmasq, row: estimator
      epsilon_df = pd.DataFrame({
          key: dataframe_select(results, threshold_df[key])
          for key, results in outputs_for_sigmasq.items()
      })
      epsilon_df.to_csv(output_filename)
      logging.warning('Saved file %s', output_filename)


def get_results_for_sigmasq(
    n,
    k,
    dim,
    seed,
    beta,
    delta,
    sigmasq_list,
    threshold_df,
    num_candidate_thresholds,
):
  """Tune the optimal threshold on one sample of data."""
  temp = generate_data.generate_data_numba(100, 10, 10, 10, 0)  # To compile
  logging.warning('Numba compilation done: %d', len(temp))

  # generate
  start_time = time.time()
  samples = generate_data.generate_data_numba(n, k, dim, seed)
  generation_time = datetime.timedelta(
      seconds=round(time.time() - start_time, 2)
  )
  logging.warning('Generation time (#1): %s', generation_time)

  # Get CIs
  dot_prod_tpr, dot_prod_fpr, dot_prod_z_tpr, dot_prod_z_fpr = samples
  outputs_for_sigmasq = {}
  for i, sigmasq_z in enumerate(sigmasq_list):
    # Statistic before thresholding
    stat_tpr = dot_prod_tpr + math.sqrt(sigmasq_z) * dot_prod_z_tpr  # (n, k)
    stat_fpr = dot_prod_fpr + math.sqrt(sigmasq_z) * dot_prod_z_fpr  # (n, m)

    if threshold_df is None:
      max_val = max(
          np.quantile(stat_tpr.reshape(-1), 0.9),
          np.quantile(stat_fpr.reshape(-1), 0.9),
      )
      min_val = min(
          np.quantile(stat_tpr.reshape(-1), 0.1),
          np.quantile(stat_fpr.reshape(-1), 0.1),
      )
      thresholds = np.linspace(min_val, max_val, num_candidate_thresholds)
    else:
      thresholds = np.unique(threshold_df[sigmasq_z])

    t1 = time.time()
    if n * k * num_candidate_thresholds > 1e9:
      # Too large for vectorization, run in a loop
      logging.warning('Running in a loop!!')
      out = process_data.get_eps_for_all_thresholds_loop(
          stat_tpr, stat_fpr, thresholds, beta, delta
      )
    else:
      # No problem, we vectorize
      logging.warning('Running vectorized!!')
      out = process_data.get_eps_for_all_thresholds_vectorized(
          stat_tpr, stat_fpr, thresholds, beta, delta
      )
    outputs_for_sigmasq[sigmasq_z] = out
    t2 = time.time()
    logging.warning(
        'sigmasq %d / %d = %.2f seconds', i, len(sigmasq_list), t2 - t1
    )

  return outputs_for_sigmasq


def dataframe_select(df: pd.DataFrame, cols_to_select: pd.Series) -> pd.Series:
  """Select one entry per row of a dataframe.

  Args:
    df: pd.DataFrame to select values from.
    cols_to_select: pd.Series with the same index as `df`.

  Returns:
    pd.Series with the selected entires per row of df.
  """
  d = {}
  for row in cols_to_select.index:
    d[row] = df.at[row, cols_to_select[row]]
  return pd.Series(d)


if __name__ == '__main__':
  app.run(main)
