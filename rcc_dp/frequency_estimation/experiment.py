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
"""Experiment definitions (i.e., evaluation of miracle, rhr, subset selection 
methods when either the data dimension k, the number of users n, or the 
privacy parameter epsilon is varied)."""

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
from rcc_dp.frequency_estimation import miracle
from rcc_dp.frequency_estimation import modify_pi
from rcc_dp.frequency_estimation import rhr
from rcc_dp.frequency_estimation import ss
from rcc_dp.frequency_estimation import unbias


def generate_geometric_distribution(k,lbd):
    elements = range(0,k)
    prob = [(1-lbd)*math.pow(lbd,x)/(1-math.pow(lbd,k)) for x in elements]
    return prob


def generate_uniform_distribution(k):
    raw_distribution = [1] * k
    sum_raw = sum(raw_distribution)
    prob = [float(y)/float(sum_raw) for y in raw_distribution]
    return prob


def generate_zipf_distribution(k,degree):
    raw_distribution = [1/(float(i)**(degree)) for i in range(1,k+1)]
    sum_raw = sum(raw_distribution)
    prob = [float(y)/float(sum_raw) for y in raw_distribution]
    return prob


def evaluate(work_path, config, file_open=open):
  """Evaluates miracle, rhr, ss methods."""
  with file_open(work_path + "/config.json", "w") as f:
    json.dump(config.to_dict(), f)

  start_time = time.time()

  delta = config.delta
  alpha = config.alpha
  # Get default values.
  k = config.k
  n = config.n
  epsilon_target = config.epsilon_target

  if config.vary == "k":
    vary_space = config.k_space
    print("k space = " + str(vary_space))
  elif config.vary == "n":
    vary_space = config.n_space
    print("n space = " + str(vary_space))
  elif config.vary == "eps":
    vary_space = config.eps_space
    print("eps space = " + str(vary_space))
  else:
    raise ValueError("vary should be either be k, n or eps.")

  approx_miracle_error = np.zeros((config.num_itr, len(vary_space)))
  miracle_error = np.zeros((config.num_itr, len(vary_space)))
  modified_miracle_error = np.zeros((config.num_itr, len(vary_space)))
  rhr_error = np.zeros((config.num_itr, len(vary_space)))
  ss_error = np.zeros((config.num_itr, len(vary_space)))

  for step, vary_parameter in enumerate(vary_space):
    if config.vary == "k":
      k = vary_parameter
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
    print("k = %d" % k)
    print("coding cost = %d" % coding_cost)

    if config.run_modified_miracle:
      eta = epsilon_target / 2.0
      print("eta = " + str(eta))
      print("alpha = " + str(alpha))

    for itr in range(config.num_itr):
      print("itr = %d" % itr)
      if config.distribution == "geometric":
        lbd = config.lbd_geometric
        prob = generate_geometric_distribution(k,lbd)
      elif config.distribution == "zipf":
        degree = config.degree_zipf
        prob = generate_zipf_distribution(k,degree)
      elif config.distribution == "uniform":
        prob = generate_uniform_distribution(k)
      else:
        raise ValueError(
            "distribution should be either be geometric, zipf, uniform.")
      x = np.random.choice(k, n, p=prob)

      if config.run_miracle:
        x_miracle = np.zeros((k,n))
        for i in range(n):
          if config.encoding_type == "fast":
            x_miracle[:,i] = miracle.encode_decode_miracle_fast(i+itr*n, x[i], 
              k, epsilon_target/2, 2**coding_cost)
          else:
            _, _, index = miracle.encoder(i+itr*n, x[i], k, epsilon_target/2, 
              2**coding_cost)
            x_miracle[:,i] = miracle.decoder(i+itr*n, index, k, epsilon_target/2, 
              2**coding_cost)
        prob_miracle = unbias.unbias_miracle(k, epsilon_target/2, 2**coding_cost, 
          x_miracle.T, n, normalization = 1)
        miracle_error[itr, step] = np.linalg.norm([p_i - phat_i for p_i, phat_i 
          in zip(prob, prob_miracle)], ord=1)

      if config.run_modified_miracle:
        x_modified_miracle = np.zeros((k, n))
        for i in range(n):
          if config.encoding_type == "fast":
            x_modified_miracle[:,i] = miracle.encode_decode_modified_miracle_fast(
              i+itr*n, x[i], k, alpha*epsilon_target, 2**coding_cost)
          else:
            _, pi, _ = miracle.encoder(i+itr*n, x[i], k, alpha*epsilon_target, 
              2**coding_cost)
            expected_beta = np.ceil(k/(np.exp(epsilon_target)+1))/k
            pi_all = modify_pi.modify_pi(pi, eta, epsilon_target, 
              (np.exp(epsilon_target/2))/(1+expected_beta*(np.exp(epsilon_target)-1)))
            index = np.random.choice(2**coding_cost, 1, p=pi_all[-1])[0]
            x_modified_miracle[:,i] = miracle.decoder(i+itr*n, index, k, 
              alpha*epsilon_target, 2**coding_cost)
        prob_modified_miracle = unbias.unbias_modified_miracle(k, alpha*epsilon_target, 
          2**coding_cost, x_modified_miracle.T, n, normalization = 1)
        modified_miracle_error[itr, step] = np.linalg.norm([p_i - phat_i for p_i, phat_i 
          in zip(prob, prob_modified_miracle)], ord=1)

      if config.run_approx_miracle:
        x_approx_miracle = np.zeros((k,n))
        approx_coding_cost = int(np.ceil(
          config.approx_coding_cost_multiplier*epsilon_target/np.log(2) + config.approx_t))
        epsilon_approx = miracle.get_approx_epsilon(epsilon_target, k, 
          2**approx_coding_cost, delta)
        for i in range(n):
          if config.encoding_type == "fast":
            x_approx_miracle[:,i] = miracle.encode_decode_miracle_fast(i+itr*n, x[i], 
              k, epsilon_approx, 2**coding_cost)
          else:
            _, _, index = miracle.encoder(i+itr*n, x[i], k, epsilon_approx, 2**coding_cost)
            x_approx_miracle[:,i] = miracle.decoder(i+itr*n, index, k, epsilon_approx, 
              2**coding_cost)
        prob_approx_miracle = unbias.unbias_miracle(k, epsilon_approx, 2**coding_cost, 
          x_approx_miracle.T, n, normalization = 1)
        approx_miracle_error[itr, step] = np.linalg.norm([p_i - phat_i for p_i, phat_i 
          in zip(prob, prob_approx_miracle)], ord=1)

      if config.run_ss:
        x_ss = ss.encode_string_fast(k, epsilon_target, x)
        prob_ss = ss.decode_string(k, epsilon_target, x_ss, n, normalization = 1)
        ss_error[itr, step] = np.linalg.norm([p_i - phat_i for p_i, phat_i 
          in zip(prob, prob_ss)], ord=1)

      if config.run_rhr:
        x_rhr = rhr.encode_string(k, epsilon_target, coding_cost, x)
        prob_rhr = rhr.decode_string_fast(k, epsilon_target, coding_cost, x_rhr, 
        normalization = 1) # estimate the original underlying distribution
        rhr_error[itr, step] = np.linalg.norm([p_i - phat_i for p_i, phat_i 
          in zip(prob, prob_rhr)], ord=1)

    if config.run_approx_miracle:
      print("approx miracle error: " +
            str(approx_miracle_error[:, step]))
    if config.run_miracle:
      print("miracle error: " + str(miracle_error[:, step]))
    if config.run_modified_miracle:
      print("modified miracle error: " +
            str(modified_miracle_error[:, step]))
    if config.run_ss:
      print("ss error: " + str(ss_error[:, step]))
    if config.run_rhr:
      print("rhr error: " + str(rhr_error[:, step]))
    print(time.time() - start_time)

  print("--------------")
  if config.run_approx_miracle:
    print("approx miracle error:")
    print(np.mean(approx_miracle_error, axis=0))
  if config.run_miracle:
    print("miracle error:")
    print(np.mean(miracle_error, axis=0))
  if config.run_modified_miracle:
    print("modified miracle error:")
    print(np.mean(modified_miracle_error, axis=0))
  if config.run_ss:
    print("ss error:")
    print(np.mean(ss_error, axis=0))
  if config.run_rhr:
    print("rhr error:")
    print(np.mean(rhr_error, axis=0))

  plt.figure(figsize=((8, 5)), dpi=80)
  plt.axes((.15, .2, .83, .75))
  if config.run_approx_miracle:
    plt.errorbar(
        vary_space,
        np.mean(approx_miracle_error, axis=0),
        yerr=np.std(approx_miracle_error, axis=0)/np.sqrt(num_itr),
        linewidth = 3.0,
        label="MRC")
  if config.run_miracle:
    plt.errorbar(
        vary_space,
        np.mean(miracle_error, axis=0),
        yerr=np.std(miracle_error, axis=0)/np.sqrt(num_itr),
        linewidth = 3.0,
        label="MRC")
  if config.run_modified_miracle:
    plt.errorbar(
        vary_space,
        np.mean(modified_miracle_error, axis=0),
        yerr=np.std(modified_miracle_error, axis=0)/np.sqrt(num_itr),
        linewidth = 3.0,
        label="MMRC")
  if config.run_ss:
    plt.errorbar(
        vary_space,
        np.mean(ss_error, axis=0),
        yerr=np.std(ss_error, axis=0)/np.sqrt(num_itr),
        ls='--',
        linewidth = 3.0,
        label="Subset Selection")
  if config.run_rhr:
    plt.errorbar(
        vary_space,
        np.mean(rhr_error, axis=0),
        yerr=np.std(rhr_error, axis=0)/np.sqrt(num_itr),
        ls='--',
        linewidth = 3.0,
        label="RHR")
  plt.xticks(fontsize=28)
  plt.yticks(fontsize=28)
  plt.ylabel("$\ell_{2}$ error", fontsize=28)
  if config.vary == "k":
    plt.xlabel(r"$d$", fontsize=28)
    plt.xticks([200,400,600,800,1000])
    plt.yticks([0.2,0.4,0.6,0.8])
    plt.legend(fontsize=24, loc='upper left')
  elif config.vary == "n":
    plt.xlabel(r"$n$", fontsize=28)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(3,3))
    plt.gca().xaxis.get_offset_text().set_fontsize(16)
    plt.yticks([0.3,0.4,0.5,0.6,0.7])
    plt.legend(fontsize=24, loc='upper right')
  elif config.vary == "eps":
    plt.xlabel(r"$\varepsilon$", fontsize=28)
    plt.yticks([0.3,0.6,0.9,1.2,1.5])
    plt.legend(fontsize=24, loc='upper right')

  with file_open(work_path + "/rcc_dp_ss_comparison.png", "wb") as f:
    plt.savefig(f, format="png")

  if config.run_approx_miracle:
    with file_open(work_path + "/approx_miracle_error.csv", "w") as f:
      np.savetxt(f, approx_miracle_error, delimiter=",")

  if config.run_miracle:
    with file_open(work_path + "/miracle_error.csv", "w") as f:
      np.savetxt(f, miracle_error, delimiter=",")

  if config.run_modified_miracle:
    with file_open(work_path + "/modified_miracle_error.csv", "w") as f:
      np.savetxt(f, modified_miracle_error, delimiter=",")

  if config.run_ss:
    with file_open(work_path + "/ss_error.csv", "w") as f:
      np.savetxt(f, ss_error, delimiter=",")

  if config.run_rhr:
    with file_open(work_path + "/rhr_error.csv", "w") as f:
      np.savetxt(f, rhr_error, delimiter=",")