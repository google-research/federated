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
"""Process the generated synthetic data into epsilons."""

import numpy as np
import pandas as pd

from lidp_auditing.confidence_estimators import asymptotic as conf
from lidp_auditing.confidence_estimators import asymptotic_vectorized as conf_vec


def get_eps_for_all_thresholds_vectorized(
    stat_tpr: np.ndarray,
    stat_fpr: np.ndarray,
    thresholds: np.ndarray,
    beta: float,
    delta: float,
) -> pd.DataFrame:
  """Compute the empirical eps for all thresholds using vectorized code."""
  # stat_tpr, stat_fpr: inner products
  # Statistic after thresholding
  x_tpr = (stat_tpr[:, :, None] >= thresholds[None, None, :]).astype(np.int32)
  # (n, k, T)
  x_fpr = (stat_fpr[:, :, None] >= thresholds[None, None, :]).astype(np.int32)
  # (n, m, T)
  # Get confidence intervals
  # Row: estimator, Columns: threshold
  outputs_tpr_l, _ = conf_vec.get_asymptotic_confidence_intervals(
      x_tpr, beta / 2, thresholds
  )
  _, outputs_fpr_r = conf_vec.get_asymptotic_confidence_intervals(
      x_fpr, beta / 2, thresholds
  )
  # index: estimator, columns: threshold
  return np.log((outputs_tpr_l - delta) / outputs_fpr_r)


def get_eps_for_all_thresholds_loop(
    stat_tpr: np.ndarray,
    stat_fpr: np.ndarray,
    thresholds: np.ndarray,
    beta: float,
    delta: float,
) -> pd.DataFrame:
  """Compute the empirical eps for all thresholds in a loop."""
  results = {}
  for tau in thresholds:
    x_tpr = (stat_tpr >= tau).astype(np.int32)  # (n, k)
    x_fpr = (stat_fpr >= tau).astype(np.int32)  # (n, m)
    # Get confidence intervals
    out = conf.get_asymptotic_confidence_intervals(
        x_tpr, beta / 2, return_statistics=False
    )
    outputs_tpr_l = out['left']
    out = conf.get_asymptotic_confidence_intervals(
        x_fpr, beta / 2, return_statistics=False
    )
    outputs_fpr_r = out['right']
    results[tau] = np.log((outputs_tpr_l - delta) / outputs_fpr_r)
  # index: index, columns: threshold
  return pd.DataFrame(results)  # index: estimator, columns: threshold
