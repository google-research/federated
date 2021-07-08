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
"""Secret sharer framework for measuring memorization in stackoverflow."""

from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import scipy as sp
import tensorflow as tf
import tensorflow_federated as tff


def build_secret_inserting_transform_fn(
    client_ids: List[str],
    secrets: Dict[str, Tuple[int, float]],
    seed: int = 0) -> Callable[[str], Callable[[Any], Any]]:
  """Builds secret inserting make_transform_fn for the StackOverflow dataset.

  Args:
    client_ids: A list of all client IDs.
    secrets: A dict mapping secrets of type `str` to an (int, float) 2-tuple.
      The int is the number of clients that will have that secret inserted, and
      the float is the probability that any given example of a selected client
      will be replaced with that client's secret (p_e in Thakkar et. al (2020)).
    seed: Random seed for client and example selection.

  Returns:
    A function that can be passed to the initializer of TransformingClientData
    to insert secrets.
  """

  if (not secrets or not isinstance(secrets, dict) or
      not all([isinstance(secret, str) for secret in secrets])):
    raise ValueError('`secrets` must be a non-zero length dict with str keys.')

  if any(value[0] <= 0 for value in secrets.values()):
    raise ValueError('Client count for each secret must be positive.')

  if any(not 0 < value[1] <= 1 for value in secrets.values()):
    raise ValueError('p_e values must be valid probabilities in (0, 1].')

  secret_client_count = sum(cc for (cc, _) in secrets.values())
  if secret_client_count >= len(client_ids):
    raise ValueError(
        'Client counts cannot sum to more than total number of clients.')

  secret, count, prob = zip(*[(s, c, p) for s, (c, p) in secrets.items()])

  np.random.seed(seed)
  secret_clients = np.random.choice(client_ids, secret_client_count, False)

  secret_idx = []
  for i, c in enumerate(count):
    secret_idx.extend([i] * c)

  default_secret_idx = len(secrets)
  secret = list(secret) + ['']
  prob = list(prob) + [0.]

  def make_transform_fn(client_id: str):
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            tf.convert_to_tensor(secret_clients),
            tf.convert_to_tensor(secret_idx)),
        default_value=default_secret_idx)
    client_id_tensor = tf.convert_to_tensor(client_id)
    client_secret_idx = table.lookup(client_id_tensor)
    client_secret = tf.convert_to_tensor(secret)[client_secret_idx]
    client_secret_prob = tf.convert_to_tensor(prob)[client_secret_idx]
    client_hash = tf.strings.to_hash_bucket_fast(client_id_tensor,
                                                 tf.int64.max) + seed

    @tf.function
    def transform_fn(example):
      example_hash = tf.strings.to_hash_bucket_fast(example['creation_date'],
                                                    tf.int64.max)
      uniform = tf.random.stateless_uniform((), [client_hash, example_hash])
      result = example.copy()
      if uniform < client_secret_prob:
        result['tokens'] = client_secret
      return result

    return transform_fn

  return make_transform_fn


def stackoverflow_with_secrets(
    stackoverflow_client_data: tff.simulation.datasets.ClientData,
    secrets: Dict[str, Tuple[int, float]],
    seed: int = 0):
  """Adds secrets to stackoverflow data.

  Assigns secret phrases to some clients. If a client is assigned a secret, each
  of that client's examples will be selected independently at random for
  replacement, in which case the `tokens` field of the example will be replaced
  by the secret.

  The method is similar to that used by Thakkar et. al (2020)
  https://arxiv.org/abs/2006.07490 except a deterministic number of clients
  are selected for each secret.

  Args:
    stackoverflow_client_data: The original stackoverflow data.
    secrets: A dict mapping secrets of type `str` to an (int, float) 2-tuple.
      The int is the number of clients that will have that secret inserted, and
      the float is the probability that any given example of a selected client
      will be replaced with that client's secret (p_e in Thakkar et. al (2020)).
    seed: Random seed for client and example selection.

  Returns:
    A `tff.simulation.datasets.TransformingClientData` that expands each client
      into `client_expansion_factor` pseudo-clients.
  """
  make_transform_fn = build_secret_inserting_transform_fn(
      stackoverflow_client_data.client_ids, secrets, seed)

  return tff.simulation.datasets.TransformingClientData(
      stackoverflow_client_data, make_transform_fn)


def generate_secrets(word_counts: Dict[str, int], secret_len: int,
                     num_secrets: int) -> List[str]:
  """Generates a list of secrets.

  Each secret consists of `secret_len` tokens chosen independently from
  the marginal distribution over tokens.

  Args:
    word_counts: A dict mapping string tokens to integer counts.
    secret_len: The number of tokens in each secret.
    num_secrets: The number of secrets to generate.

  Returns:
    A list of string secrets.
  """
  weights = np.array([float(c) for c in word_counts.values()])
  weights /= sum(weights)
  token_ids = np.random.choice(
      len(word_counts), size=(num_secrets, secret_len), p=weights)
  vocab = list(word_counts.keys())
  return [
      ' '.join([vocab[t] for t in token_ids[i, :]]) for i in range(num_secrets)
  ]


def compute_exposure(secrets: List[str], reference_secrets: List[str],
                     get_perplexity: Callable[[str], float]) -> List[float]:
  """Computes exposure of list of secrets using extrapolation method.

  See Carlini et al., 2019 https://arxiv.org/pdf/1802.08232.pdf for details.

  Args:
    secrets: List of secrets to compute exposure of.
    reference_secrets: List of reference secrets to estimate perplexity
      distribution.
    get_perplexity: A callable that returns the model's perplexity on a secret.

  Returns:
    A list of exposures of the same length as `secrets`.
  """
  perplexities_reference = [get_perplexity(s) for s in reference_secrets]
  snormal_param = sp.stats.skewnorm.fit(perplexities_reference)

  return [
      -np.log2(sp.stats.skewnorm.cdf(get_perplexity(s), *snormal_param))
      for s in secrets
  ]
