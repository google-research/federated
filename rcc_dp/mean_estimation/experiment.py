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
when either the data dimension d, the number of users n, or the privacy
parameter epsilon is varied)."""

import json
import math
import time
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from scipy import stats
from rcc_dp.mean_estimation import get_parameters
from rcc_dp.mean_estimation import miracle
from rcc_dp.mean_estimation import modify_pi
from rcc_dp.mean_estimation import privunit
from rcc_dp.mean_estimation import sqkr
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True


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

  if config.vary == "d":
    vary_space = config.d_space
    print("d space = " + str(vary_space))
  elif config.vary == "n":
    vary_space = config.n_space
    print("n space = " + str(vary_space))
  elif config.vary == "eps":
    vary_space = config.eps_space
    print("eps space = " + str(vary_space))
  else:
    raise ValueError("vary should be either be d, n or eps.")

  approx_miracle_mse = np.zeros((config.num_itr, len(vary_space)))
  miracle_mse = np.zeros((config.num_itr, len(vary_space)))
  modified_miracle_mse = np.zeros((config.num_itr, len(vary_space)))
  sqkr_mse = np.zeros((config.num_itr, len(vary_space)))
  privunit_mse = np.zeros((config.num_itr, len(vary_space)))

  for step, vary_parameter in enumerate(vary_space):
    if config.vary == "d":
      d = vary_parameter
      coding_cost = config.coding_cost
    elif config.vary == "n":
      n = vary_parameter
      coding_cost = config.coding_cost
    elif config.vary == "eps":
      epsilon_target = vary_parameter
      coding_cost = config.coding_cost_multiplier * epsilon_target / np.log(2)
      coding_cost += config.t
      coding_cost = max(int(np.ceil(coding_cost)), 8)
    print("epsilon target = " + str(epsilon_target))
    print("n = " + str(n))
    print("d = %d" % d)
    print("coding cost = %d" % coding_cost)

    if config.run_approx_miracle:
      print("budget = " + str(budget))

    if config.run_modified_miracle:
      eta = epsilon_target / 2.0
      print("eta = " + str(eta))
      print("alpha = " + str(alpha))

    for itr in range(config.num_itr):
      print("itr = %d" % itr)
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

      if config.run_miracle:
        x_miracle = np.zeros((d, n))
        c1, c2, m, gamma = get_parameters.get_parameters_unbiased_miracle(
            epsilon_target / 2, d, 2**coding_cost, budget)
        for i in range(n):
          k, _, _ = miracle.encoder(i + itr * n, x[:, i], 2**coding_cost, c1,
                                    c2, gamma)
          z_k = miracle.decoder(i + itr * n, k, d, 2**coding_cost)
          x_miracle[:, i] = z_k / m
        x_miracle = np.mean(x_miracle, axis=1, keepdims=True)
        miracle_mse[itr, step] = np.linalg.norm(
            np.mean(x, axis=1, keepdims=True) - x_miracle)**2

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

      if config.run_approx_miracle:
        x_approx_miracle = np.zeros((d, n))
        approx_coding_cost = int(
            np.ceil(config.approx_coding_cost_multiplier * epsilon_target /
                    np.log(2) + config.approx_t))
        c1, c2, m, gamma, _ = (
            get_parameters.get_parameters_unbiased_approx_miracle(
                epsilon_target, d, 2**approx_coding_cost, 2**coding_cost,
                budget, delta))
        for i in range(n):
          k, _, _ = miracle.encoder(i + itr * n, x[:, i], 2**coding_cost, c1,
                                    c2, gamma)
          z_k = miracle.decoder(i + itr * n, k, d, 2**coding_cost)
          x_approx_miracle[:, i] = z_k / m
        x_approx_miracle = np.mean(x_approx_miracle, axis=1, keepdims=True)
        approx_miracle_mse[itr, step] = np.linalg.norm(
            np.mean(x, axis=1, keepdims=True) - x_approx_miracle)**2

      if config.run_privunit:
        x_privunit, _ = privunit.apply_privunit(x, epsilon_target, budget)
        x_privunit = np.mean(np.array(x_privunit), axis=1, keepdims=True)
        privunit_mse[itr, step] = np.linalg.norm(
            np.mean(x, axis=1, keepdims=True) - x_privunit)**2

      if config.run_sqkr:
        # Generate a random tight frame satisfying UP -- for sqkr.
        frame = 2**int(math.ceil(math.log(d, 2)) + 1)
        u = stats.ortho_group.rvs(dim=frame).T[:, 0:d]

        k_equiv = min(epsilon_target, coding_cost)
        [_, _, q_perturb] = sqkr.kashin_encode(u, x, k_equiv, epsilon_target)
        x_kashin = sqkr.kashin_decode(u, k_equiv, epsilon_target, q_perturb)
        sqkr_mse[itr, step] = np.linalg.norm(
            np.mean(x, axis=1, keepdims=True) - x_kashin)**2

    if config.run_approx_miracle:
      print("approx miracle mse: " + str(approx_miracle_mse[:, step]))
    if config.run_miracle:
      print("miracle mse: " + str(miracle_mse[:, step]))
    if config.run_modified_miracle:
      print("modified miracle mse: " + str(modified_miracle_mse[:, step]))
    if config.run_privunit:
      print("privunit mse: " + str(privunit_mse[:, step]))
    if config.run_sqkr:
      print("sqkr mse: " + str(sqkr_mse[:, step]))
    print(time.time() - start_time)

  print("--------------")
  if config.run_approx_miracle:
    print("approx miracle mse:")
    print(np.mean(approx_miracle_mse, axis=0))
  if config.run_miracle:
    print("miracle mse:")
    print(np.mean(miracle_mse, axis=0))
  if config.run_modified_miracle:
    print("modified miracle mse:")
    print(np.mean(modified_miracle_mse, axis=0))
  if config.run_privunit:
    print("privunit mse:")
    print(np.mean(privunit_mse, axis=0))
  if config.run_sqkr:
    print("sqkr mse:")
    print(np.mean(sqkr_mse, axis=0))

  plt.figure(figsize=((8, 5)), dpi=80)
  plt.axes((.15, .2, .83, .75))
  if config.run_approx_miracle:
    plt.errorbar(
        vary_space,
        np.mean(approx_miracle_mse, axis=0),
        yerr=np.std(approx_miracle_mse, axis=0) / np.sqrt(config.num_itr),
        linewidth = 3.0,
        label="MRC")
  if config.run_miracle:
    plt.errorbar(
        vary_space,
        np.mean(miracle_mse, axis=0),
        yerr=np.std(miracle_mse, axis=0) / np.sqrt(config.num_itr),
        linewidth = 3.0,
        label="MRC")
  if config.run_modified_miracle:
    plt.errorbar(
        vary_space,
        np.mean(modified_miracle_mse, axis=0),
        yerr=np.std(modified_miracle_mse, axis=0) / np.sqrt(config.num_itr),
        linewidth = 3.0,
        label="MMRC")
  if config.run_privunit:
    plt.errorbar(
        vary_space,
        np.mean(privunit_mse, axis=0),
        yerr=np.std(privunit_mse, axis=0) / np.sqrt(config.num_itr),
        ls='--',
        linewidth = 3.0,
        label="PrivUnit$_{2}$")
  if config.run_sqkr:
    plt.errorbar(
        vary_space,
        np.mean(sqkr_mse, axis=0),
        yerr=np.std(sqkr_mse, axis=0) / np.sqrt(config.num_itr),
        ls='--',
        linewidth = 3.0,
        label="SQKR")
  plt.legend(fontsize=24)
  plt.xticks(fontsize=28)
  plt.yticks(fontsize=28)
  plt.ylabel("$\ell_{2}$ error", fontsize=28)
  if config.vary == "d":
    plt.xlabel(r"$d$", fontsize=28)
    plt.xticks([200,400,600,800,1000])
    plt.yticks([0.00,0.04,0.08,0.12,0.16])
    plt.legend(fontsize=24, loc='upper left')
  elif config.vary == "n":
    plt.xlabel(r"$n$", fontsize=28)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(3,3))
    plt.gca().xaxis.get_offset_text().set_fontsize(16)
    plt.yticks([0.00,0.05,0.10,0.15,0.20])
    plt.legend(fontsize=24, loc='upper right')
  elif config.vary == "eps":
    plt.xlabel(r"$\varepsilon$", fontsize=28)
    plt.yticks([0.3,0.6,0.9,1.2,1.5])
    plt.legend(fontsize=24, loc='upper right')

  with file_open(work_path + "/rcc_dp_comparison.png", "wb") as f:
    plt.savefig(f, format="png")

  with file_open(work_path + "/time.txt", "w") as f:
    np.savetxt(f, np.array(time.time() - start_time).reshape(-1, 1))

  if config.run_approx_miracle:
    with file_open(work_path + "/approx_miracle_mse.csv", "w") as f:
      np.savetxt(f, approx_miracle_mse, delimiter=",")

  if config.run_miracle:
    with file_open(work_path + "/miracle_mse.csv", "w") as f:
      np.savetxt(f, miracle_mse, delimiter=",")

  if config.run_modified_miracle:
    with file_open(work_path + "/modified_miracle_mse.csv", "w") as f:
      np.savetxt(f, modified_miracle_mse, delimiter=",")

  if config.run_privunit:
    with file_open(work_path + "/privunit_mse.csv", "w") as f:
      np.savetxt(f, privunit_mse, delimiter=",")

  if config.run_sqkr:
    with file_open(work_path + "/sqkr_mse.csv", "w") as f:
      np.savetxt(f, sqkr_mse, delimiter=",")
