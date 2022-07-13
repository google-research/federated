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
"""Library for synthesizing federated dataset by Dirichlet distribution over classes."""

import collections
from typing import Any, List, Mapping, MutableMapping, Optional

from absl import logging

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from generalization.utils import client_data_utils


class _DirichletOverLabelsSynthesizer():
  """The backend class of function synthesize_by_dirichlet_over_labels()."""

  def __init__(self,
               dataset: tf.data.Dataset,
               num_clients: int,
               concentration_factor: float,
               seed: Optional[int] = None):

    if not isinstance(dataset.element_spec, Mapping):
      raise TypeError("Input dataset should have element_spec of type Mapping.")
    if "label" not in dataset.element_spec:
      raise TypeError("Input dataset should contain label keyed by `label`.")

    # Construct random generator to be used for all random procedures.
    self._rng = np.random.default_rng(seed)
    self._client_ids = list(map(str, range(num_clients)))
    self._concentration_factor = concentration_factor
    self._element_spec = dataset.element_spec

    # Unpack the entire dataset into a list to facilitate construction.

    logging.info("Starting unpacking dataset.")
    self._dataset_list = list()
    for logging_cnt, elem in enumerate(dataset.as_numpy_iterator()):
      if logging_cnt % 10000 == 0:
        logging.info("Unpacking dataset, %d elements processed", logging_cnt)
      self._dataset_list.append(elem)
    logging.info("Finished unpacking dataset.")

    # Preprocessing.
    self._elem_pools_by_label = self._build_elem_pools_by_label()
    self._client_multinomials = self._sample_multinomial_of_all_clients()

  def _build_elem_pools_by_label(self) -> MutableMapping[Any, List[Any]]:
    """Build a pool of elements for each label.

    Returns:
      A mapping with key to be the possible labels and value to be the
      corresponding indices of this label in the original dataset.
    """
    elem_pools_by_label = collections.OrderedDict()

    for logging_cnt, element in enumerate(self._dataset_list):
      if logging_cnt % 100000 == 0:
        logging.info("Building element pools by label, %d of %d processed.",
                     logging_cnt, len(self._dataset_list))
      label = element["label"]
      if label not in elem_pools_by_label:
        elem_pools_by_label[label] = list()

      elem_pools_by_label[label].append(element)
    map(self._rng.shuffle, elem_pools_by_label)
    return elem_pools_by_label

  def _compute_prior(self) -> Mapping[Any, float]:
    """Compute the prior distribution to be the relative popularity."""
    prior = collections.OrderedDict()
    for label, pool in self._elem_pools_by_label.items():
      prior[label] = len(pool) / len(self._dataset_list)
    return prior

  def _sample_multinomial_of_all_clients(
      self) -> Mapping[str, MutableMapping[Any, float]]:
    """Sample the multinomial distribution for all the clients."""
    prior = self._compute_prior()

    clients_multinomial = collections.OrderedDict()

    for client_id in self._client_ids:
      multinomial = self._rng.dirichlet(self._concentration_factor *
                                        np.array(list(prior.values())))
      clients_multinomial[client_id] = {
          label: prob
          for label, prob in zip(self._elem_pools_by_label, multinomial)
      }

    return clients_multinomial

  def _renormalize_multinomial(self, multinomial: MutableMapping[Any, float],
                               label_to_reset: Any):
    """Reset and renormalize a given multinomial in place."""
    multinomial[label_to_reset] = 0
    normalizer = sum(multinomial.values())

    for label in multinomial:
      multinomial[label] /= normalizer

  def _renormalize_multinomial_of_all_clients(self, label):
    """Reset and renormalize the multinomials of all clients in place."""
    for client_id in self._client_ids:
      self._renormalize_multinomial(self._client_multinomials[client_id], label)

  def _sample_a_label_by_multinomial(self, multinomial: MutableMapping[Any,
                                                                       float]):
    """Sample a label according to some multinomial."""
    label_idx = self._rng.choice(
        range(len(multinomial)), p=list(multinomial.values()))
    return list(multinomial.keys())[label_idx]

  def build_client_data(
      self, rotate_draw: bool) -> tff.simulation.datasets.ClientData:
    """Build a clientdata instance in memory."""
    samples_per_client = len(self._dataset_list) // len(self._client_ids)
    client_pools = {client_id: list() for client_id in self._client_ids}
    logging_cnt = 0

    def _draw_once(client_id: str, logging_cnt: int):
      if logging_cnt % ((len(self._dataset_list) + 9) // 10) == 0:
        logging.info(
            "Creating synthesized dataset, %d out of %d processed.",
            logging_cnt,
            len(self._dataset_list),
        )

      multinomial = self._client_multinomials[client_id]

      sampled_label = self._sample_a_label_by_multinomial(multinomial)
      sampled_item = self._elem_pools_by_label[sampled_label].pop()
      client_pools[client_id].append(sampled_item)

      # If a label is exhausted, renormalize client_multinomial.
      if not self._elem_pools_by_label[sampled_label]:
        self._renormalize_multinomial_of_all_clients(sampled_label)

      return logging_cnt + 1

    if rotate_draw:
      for _ in range(samples_per_client):
        for client_id in self._rng.permutation(self._client_ids):
          logging_cnt = _draw_once(client_id, logging_cnt)
    else:
      for client_id in self._rng.permutation(self._client_ids):
        for _ in range(samples_per_client):
          logging_cnt = _draw_once(client_id, logging_cnt)

    tensor_slices_dict = {
        client_id: client_data_utils.convert_list_of_elems_to_tensor_slices(
            client_pools[client_id]) for client_id in self._client_ids
    }

    return tff.simulation.datasets.TestClientData(tensor_slices_dict)


def synthesize_by_dirichlet_over_labels(
    dataset: tf.data.Dataset,
    num_clients: int,
    concentration_factor: float = 1,
    use_rotate_draw: bool = True,
    seed: Optional[int] = 1,
) -> tff.simulation.datasets.TestClientData:
  """Construct a federated dataset from a centralized dataset `tf.data.Dataset`.

  Sampling based on Dirichlet distribution over categories, following the paper
  Measuring the Effects of Non-Identical Data Distribution for
  Federated Visual Classification (https://arxiv.org/abs/1909.06335).

  This implementation is adapted from
  https://github.com/google-research/federated/blob/master/utils/datasets/cifar10_dataset.py.

  Assumptions:
    1) `dataset` should has `element_spec` of type `Mapping[str,
    tf.TensorSpec]`, with a hashable label keyed by 'label'.

  Limitations:
    1) The current implementations will unpack the entire dataset into
    the memory. This could result in large memory use if the dataset is large.

  Args:
    dataset: The original tf.data.Dataset to be partitioned.
    num_clients: The number of clients the examples are going to be partitioned
      on.
    concentration_factor: Parameter of Dirichlet distribution. Each client will
      sample from Dirichlet(concentration_factor * label_relative_popularity) to
      get a multinomial distribution over labels. It controls the data
      heterogeneity of clients. If approaches 0, then each client only have data
      from a single category label. If approaches infinity, then the client
      distribution will approach overall popularity.
    use_rotate_draw: Whether to rotate the drawing clients. If True, each client
      will draw only one sample at once, and then rotate to the next random
      client. This is intended to prevent the last clients from deviating from
      its desired distribution. If False, a client will draw all the samples at
      once before moving to the next client.
    seed: An optional integer representing the random seed for all random
      procedures. If None, no random seed is used.

  Returns:
    A ClientData instance.
  """

  synthesizer = _DirichletOverLabelsSynthesizer(dataset, num_clients,
                                                concentration_factor, seed)
  return synthesizer.build_client_data(rotate_draw=use_rotate_draw)  # pytype: disable=bad-return-type  # dynamic-method-lookup
