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
"""Experiment definitions (i.e., evaluation of miracle, sqkr, privunit methods 
when the coding cost is varied)."""

import json
import math
import time
import matplotlib
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from scipy import stats
from rcc_dp.mean_estimation import get_parameters
from rcc_dp.mean_estimation import miracle
from rcc_dp.mean_estimation import modify_pi
from rcc_dp.mean_estimation import privunit
from rcc_dp.mean_estimation import sqkr


def evaluate(work_path, config, file_open=open):
  """Evaluates miracle, sqkr, privunit methods."""
  with file_open(work_path + "/config.json", "w") as f:
    json.dump(config.to_dict(), f)

  start_time = time.time()

  delta = config.delta
  budget = config.budget
  alpha = config.alpha
  # Get default values.
  d = config.d
  n = config.n
  epsilon_target = config.epsilon_target

  vary_space = config.cc

  print("coding space = " + str(vary_space))

  modified_miracle_mse = np.zeros((config.num_itr, len(vary_space)))
  sqkr_mse = np.zeros((config.num_itr, 1))
  privunit_mse = np.zeros((config.num_itr, 1))

  sqkr_coding_cost = epsilon_target

  for itr in range(num_itr):
    print("itr = %d" % itr)
    print("epsilon target = " + str(epsilon_target))
    print("n = " + str(n))
    print("d = %d" % d)
    if config.run_modified_miracle:
      eta = epsilon_target / 2.0
      print("eta = " + str(eta))
      print("alpha = " + str(alpha))

    if config.data == "unbiased_data":
        x = np.random.normal(0, 1, (d, n))
        x /= np.linalg.norm(x, axis=0)
      elif config.data == "biased_data":
        x = np.zeros((d, n))
        x[:, 0::2] = np.random.normal(10, 1, (d, (n + 1) // 2))
        x[:, 1::2] = np.random.normal(1, 1, (d, n // 2))
        x /= np.linalg.norm(x, axis=0)
      elif config.data == "same_data":
        x = np.random.normal(0, 1, (d, 1))
        x /= np.linalg.norm(x, axis=0)
        x = np.repeat(x, n, axis=1)
      else:
        raise ValueError(
            "data should be either be biased_data, unbiased_data, same_data.")

    if config.run_privunit:
      x_privunit, _ = privunit.apply_privunit(x, epsilon_target, budget)
      x_privunit = np.mean(np.array(x_privunit), axis=1, keepdims=True)
      privunit_mse[itr, 0] = np.linalg.norm(
          np.mean(x, axis=1, keepdims=True) - x_privunit)**2

    if config.run_sqkr:
      # Generate a random tight frame satisfying UP -- for sqkr.
      frame = 2**int(math.ceil(math.log(d, 2)) + 1)
      u = stats.ortho_group.rvs(dim=frame).T[:, 0:d]

      k_equiv = min(epsilon_target, sqkr_coding_cost)
      [_, _, q_perturb] = sqkr.kashin_encode(u, x, k_equiv, epsilon_target)
      x_kashin = sqkr.kashin_decode(u, k_equiv, epsilon_target, q_perturb)
      sqkr_mse[itr, 0] = np.linalg.norm(
          np.mean(x, axis=1, keepdims=True) - x_kashin)**2

    for step, vary_parameter in enumerate(vary_space):
      coding_cost = vary_parameter
      print("coding cost = %d" % coding_cost)    

      if config.run_modified_miracle:
        x_modified_miracle = np.zeros((d, n))
        c1, c2, m, gamma = (
            get_parameters.get_parameters_unbiased_modified_miracle(
                alpha * epsilon_target, d, 2**coding_cost, budget))
        for i in range(n):
          _, _, pi = miracle.encoder(i + itr * n, x[:, i], 2**coding_cost, c1,
                                     c2, gamma)
          pi_all = modify_pi.modify_pi(pi, eta, epsilon_target,
                                       c1 / (np.exp(epsilon_target / 2)))
          k = np.random.choice(2**coding_cost, 1, p=pi_all[-1])[0]
          z_k = miracle.decoder(i + itr * n, k, d, 2**coding_cost)
          x_modified_miracle[:, i] = z_k / m
        x_modified_miracle = np.mean(x_modified_miracle, axis=1, keepdims=True)
        modified_miracle_mse[itr, step] = np.linalg.norm(
            np.mean(x, axis=1, keepdims=True) - x_modified_miracle)**2

    print(time.time() - start_time)

  print("--------------")
  if config.run_approx_miracle:
    print("approx miracle mse:")
    print(np.mean(approx_miracle_mse, axis=0))
  if config.run_privunit:
    print("privunit mse:")
    print(np.mean(privunit_mse, axis=0))
  if config.run_sqkr:
    print("sqkr mse:")
    print(np.mean(sqkr_mse, axis=0))

  plt.figure(figsize=((8, 5)), dpi=80)
  plt.axes((.15, .2, .83, .75))
  if config.run_modified_miracle:
    plt.errorbar(
        vary_space,
        np.mean(modified_miracle_mse, axis=0),
        yerr=np.std(modified_miracle_mse, axis=0)/np.sqrt(num_itr),
        linewidth = 3.0,
        label="MMRC")
  if config.run_privunit:
    line1 = plt.errorbar(vary_space, 
      [np.mean(privunit_mse, axis=0)[0]]*len(vary_space), 
      yerr = [np.std(privunit_mse, axis=0)[0]/np.sqrt(num_itr)]*len(vary_space), 
      ls='--',
      linewidth = 3.0,
      label="PrivUnit$_{2}$")
  if config.run_sqkr:
    line2 = plt.errorbar(vary_space, 
      [np.mean(sqkr_mse, axis=0)[0]]*len(vary_space), 
      yerr = [np.std(sqkr_mse, axis=0)[0]/np.sqrt(num_itr)]*len(vary_space), 
      ls='--',
      linewidth = 3.0,
      label="SQKR")
  plt.xticks(fontsize=28)
  plt.yticks(fontsize=28)
  plt.ylabel("$\ell_{2}$ error", fontsize=28)
  plt.xlabel(r"$\#$ bits", fontsize=28)
  plt.yticks([0.04,0.05,0.06,0.07,0.08])
  plt.legend(fontsize=24, loc='center right')

  with file_open(work_path + "/rcc_dp_mse_vs_coding_cost.png", "wb") as f:
    plt.savefig(f, format="png")

  with file_open(work_path + "/time.txt", "w") as f:
    np.savetxt(f, np.array(time.time() - start_time).reshape(-1, 1))

  if config.run_modified_miracle:
    with file_open(work_path + "/modified_miracle_mse.csv", "w") as f:
      np.savetxt(f, modified_miracle_mse, delimiter=",")

  if config.run_privunit:
    with file_open(work_path + "/privunit_mse.csv", "w") as f:
      np.savetxt(f, privunit_mse, delimiter=",")

  if config.run_sqkr:
    with file_open(work_path + "/sqkr_mse.csv", "w") as f:
      np.savetxt(f, sqkr_mse, delimiter=",")