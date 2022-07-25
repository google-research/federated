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
"""Run script for distributed mean estimation."""

import collections
import os
from typing import Any, List, DefaultDict, Tuple

from absl import app
from absl import flags
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_privacy as tfp

from distributed_dp import accounting_utils
from distributed_dp import ddpquery_utils
from distributed_dp import dme_utils
from private_linear_compression import count_sketching_utils

_RUN_ID = flags.DEFINE_integer(
    'run_id',
    None, 'ID of the run, useful for identifying '
    'the run when parallelizing this script.',
    required=True)
_OUTPUT_DIR = flags.DEFINE_string('output_dir', '/tmp/ddp_dme_outputs',
                                  'Output directory.')
_TAG = flags.DEFINE_string('tag', '',
                           'Extra subfolder for the output result files.')
_MECHANISM = flags.DEFINE_enum('mechanism', 'ddgauss', ['ddgauss', 'dskellam'],
                               'DDP mechanism to use.')
_CLIP_NORM = flags.DEFINE_float('clip_norm', 1.0,
                                'Norm of the randomly generated vectors.')
_SQRTN_NORM_GROWTH = flags.DEFINE_boolean(
    'sqrtn_norm_growth', False, 'Whether to assume the bound '
    'norm(sum_i x_i) <= sqrt(n) * c.')
_COMPRESSION_RATES = flags.DEFINE_multi_float(
    'compression_rates',
    default=np.arange(1.0, 5.25, 0.25),
    help='List of `Float` values reprensenting the compression rates, r.')
_BITS = flags.DEFINE_multi_integer(
    'bits', [18],
    'List of `Int` values reprensenting the bit widths, b.',
    lower_bound=1,
    upper_bound=32)
_NOISE_MULTIPLIERS = flags.DEFINE_multi_float(
    'noise_multipliers', [0.1, 0.5, 1.0, 10.0],
    '`Float` values reprensenting the DP noise multipliers.',
    lower_bound=0.)
_VECTOR_DIMENSIONALITY = flags.DEFINE_integer(
    'vector_dimensionality',
    300,
    'Number of coordinates in each client\'s vector.',
    lower_bound=0)
_NUM_CLIENTS = flags.DEFINE_integer(
    'num_clients',
    100,
    'Number of clients or cohort size of the DME experiment.',
    lower_bound=1)
_COMPRESS_CDP = flags.DEFINE_boolean(
    'compress_central', False,
    'True to use linear compression on central DP too.')

FLAGS = flags.FLAGS


def _add_to_results(ddp_results: DefaultDict[str, Any],
                    *key_val_pairs: Tuple[str, Any]) -> None:
  """Appends inplace each `value` to associated `key` in ddp_results."""
  for key, val in key_val_pairs:
    ddp_results[key].append(val)


def _mean_squared_error(a: np.ndarray, b: np.ndarray) -> float:
  """Calculates the mean squared error (MSE) between vectors `a` and `b`."""
  return float(np.square(a - b).mean())


def experiment(client_data: List[np.ndarray],
               ddp_result_holder: DefaultDict[str, List[Any]]):
  """Run a distributed mean estimation experiment.

  Args:
    client_data: A list of `n` np.array vectors, each with shape (d,).
    ddp_result_holder: A defaultdict returning a list.

  Returns:
    Experiment results as lists of MSE.
  """
  # Generic local params.
  vector_dimensionality = _VECTOR_DIMENSIONALITY.value
  run_id = _RUN_ID.value
  num_clients = _NUM_CLIENTS.value

  # Fixed DDP params.
  k_stddevs = 4
  beta = np.exp(-0.5)
  mechanism = _MECHANISM.value
  delta = 1e-5

  # Fixed compression params.
  length = 15
  index_seeds = (run_id, run_id)
  sign_seeds = (run_id, run_id)

  # Key local variables
  noise_multiplier_key = 'Noise Multiplier, $z$'
  bit_key = 'Bit Width, $b$'
  compression_key = 'Compression Rate, $r$'
  mse_key = 'Mean Squared Error, MSE'
  repeat_key = 'Repeat ID'
  type_key = 'DP Mechanism'
  num_client_key = 'Cohort Size, $n$'
  vector_dim_key = 'Vector Dimensionality, $d$'

  # `client_data` has shape (n, d).
  true_avg_vector = np.mean(client_data, axis=0)
  l2_clip_norm = _CLIP_NORM.value * 1.1 * np.sqrt(length)

  for noise_multiplier in _NOISE_MULTIPLIERS.value:
    for compression_rate in _COMPRESSION_RATES.value:
      # Setup compression
      width = int(vector_dimensionality // (length * compression_rate))
      if width < 1:
        continue

      if compression_rate == 1.0:
        sketch_size = vector_dimensionality
      else:
        sketch_size = length * width

      def compress_fn(vector, width=width, compression_rate=compression_rate):
        if compression_rate > 1:
          sketch = count_sketching_utils.encode(
              tf.convert_to_tensor(vector),
              length=tf.convert_to_tensor(length),
              width=tf.convert_to_tensor(width),
              index_seeds=tf.convert_to_tensor(index_seeds),
              sign_seeds=tf.convert_to_tensor(sign_seeds))
          return np.reshape(sketch.numpy(), [length * width])
        else:
          return vector

      def decompress_fn(aggregate_sketch,
                        width=width,
                        compression_rate=compression_rate):
        if compression_rate > 1:
          aggregate_sketch = np.reshape(aggregate_sketch, [length, width])
          aggregate_sketch = tf.convert_to_tensor(aggregate_sketch)
          aggregate_vector = count_sketching_utils.decode(
              aggregate_sketch,
              gradient_length=tf.convert_to_tensor(vector_dimensionality),
              index_seeds=tf.convert_to_tensor(index_seeds),
              sign_seeds=tf.convert_to_tensor(sign_seeds),
              method=count_sketching_utils.DecodeMethod.MEAN)
          return np.reshape(aggregate_vector.numpy(), [vector_dimensionality])
        else:
          return aggregate_sketch

      # Setup data.
      compressed_data = [
          compress_fn(client_datum) for client_datum in client_data
      ]
      padded_dim = np.math.pow(2, np.ceil(np.log2(sketch_size)))
      client_template = tf.zeros_like(compressed_data[0])

      if compression_rate == 1.0 or _COMPRESS_CDP.value:
        gauss_query = tfp.GaussianSumQuery(
            l2_norm_clip=l2_clip_norm, stddev=noise_multiplier)
        gauss_avg_sketch = dme_utils.compute_dp_average(
            compressed_data, gauss_query, is_compressed=False, bits=None)

        gauss_avg_vector = decompress_fn(gauss_avg_sketch)

        _add_to_results(
            ddp_result_holder,
            (compression_key, compression_rate),
            (noise_multiplier_key, noise_multiplier),
            (bit_key, 32),
            (mse_key, _mean_squared_error(gauss_avg_vector, true_avg_vector)),
            (repeat_key, run_id),
            (type_key, 'Central'),
            (num_client_key, num_clients),
            (vector_dim_key, vector_dimensionality),
        )

      # Perform a separate Distributed Discrete Gaussian call for each `bit`
      # to compare against the corresponding continuous Gaussian baseline.
      for bit in _BITS.value:
        # Setup DDP
        eps = accounting_utils.get_eps_gaussian(
            q=1,
            noise_multiplier=noise_multiplier,
            steps=1,
            target_delta=delta,
            orders=accounting_utils.RDP_ORDERS)

        gamma, local_stddev = accounting_utils.ddgauss_params(
            q=1,
            epsilon=eps,
            l2_clip_norm=l2_clip_norm,
            bits=bit,
            num_clients=num_clients,
            dim=padded_dim,
            delta=delta,
            beta=beta,
            steps=1,
            k=k_stddevs,
            sqrtn_norm_growth=_SQRTN_NORM_GROWTH.value)
        scale = 1.0 / gamma

        ddp_query = ddpquery_utils.build_ddp_query(
            mechanism,
            local_stddev,
            l2_norm_bound=l2_clip_norm,
            beta=beta,
            padded_dim=padded_dim,
            scale=scale,
            client_template=client_template)

        # Perform DDP averaging
        distributed_avg_sketch = dme_utils.compute_dp_average(
            compressed_data, ddp_query, is_compressed=True, bits=bit)

        distributed_avg_vector = decompress_fn(distributed_avg_sketch)

        _add_to_results(
            ddp_result_holder,
            (compression_key, compression_rate),
            (noise_multiplier_key, noise_multiplier),
            (bit_key, bit),
            (mse_key,
             _mean_squared_error(distributed_avg_vector, true_avg_vector)),
            (repeat_key, run_id),
            (type_key, 'Distributed'),
            (num_client_key, num_clients),
            (vector_dim_key, vector_dimensionality),
        )


def main(_):
  """Run distributed mean estimation experiments."""
  np.random.seed(FLAGS.run_id)

  ddp_results = collections.defaultdict(list)

  client_data = dme_utils.generate_client_data(
      _VECTOR_DIMENSIONALITY.value,
      _NUM_CLIENTS.value,
      l2_norm=_CLIP_NORM.value)
  experiment(client_data, ddp_results)

  # Save to file.
  dirname = os.path.join(_OUTPUT_DIR.value, _TAG.value)
  if not os.path.exists(dirname):
    os.makedirs(dirname)
  fname = f'rid={FLAGS.run_id}.pkl'
  out_path = os.path.join(dirname, fname)
  ddp_results = pd.DataFrame(ddp_results)
  ddp_results.to_pickle(out_path)
  print(f'Run {FLAGS.run_id} done.')


if __name__ == '__main__':
  app.run(main)
