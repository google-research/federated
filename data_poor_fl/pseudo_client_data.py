# Copyright 2022, Google LLC.
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
"""Utilities for partitioning ClientData datasets among pseudo-clients.

Note that we cannot use `tff.simulation.datasets.TransformingClientData`, as
this works at a per-element level. By contrast, we want to do this on a per
client level, using `tf.data.Dataset.skip` and `tf.data.Dataset.take`.
"""

from typing import List, Optional

import tensorflow as tf
import tensorflow_federated as tff


@tf.function
def _get_dataset_size(dataset: tf.data.Dataset) -> int:
  return dataset.reduce(0, lambda x, _: x + 1)


@tf.function
def _create_pseudo_client(dataset: tf.data.Dataset,
                          examples_per_pseudo_client: int,
                          pseudo_client_index: int):
  dataset = dataset.skip(examples_per_pseudo_client * pseudo_client_index)
  return dataset.take(examples_per_pseudo_client)


def create_pseudo_client_data(
    base_client_data: tff.simulation.datasets.ClientData,
    examples_per_pseudo_client: int,
    pseudo_client_ids: Optional[List[str]] = None,
    separator: str = '-'):
  """Partitions an existing `ClientData` into pseudo-clients.

  A client with dataset of size `num_examples` will be partitioned into
  `ceil(num_client_examples/examples_per_pseudo_client)`. In particular, all
  but one of these partitions will have exactly `examples_per_pseudo_client`
  examples, with the last pseudo-client potentially having fewer.

  For example, if `base_client_data` has client ids 'A' and 'B', each of which
  has 2 examples, then `create_pseudo_client_data(base_client_data, 1)` will
  produce a `tff.simulation.datasets.ClientData` with 4 clients. These clients
  will have ids `A-0`, `A-1`, `B-0`, and `B-1`, each of which will have a single
  example.

  Args:
    base_client_data: A `tff.simulation.datasets.ClientData`.
    examples_per_pseudo_client: An integer specifying the max number of examples
      per pseudo-clients. All pseudo-clients will have this many examples,
      except possibly the last pseudo-client associated to a given real client.
    pseudo_client_ids: A list of pseudo-client ids to use. Each element must be
      of the form `'x-y'` where `x` is an element of
      `base_client_data.client_ids` and `y` is some nonnegative integer. If set
      to `None`, this will be computed based on the dataset sizes in
      `base_client_ids`.
    separator: A string used to separate the base client id and pseudo-client
      index.

  Returns:
    A `tff.simulation.datasets.ClientData`.
  """
  if pseudo_client_ids is None:
    pseudo_client_ids = []
    for client_id in base_client_data.client_ids:
      base_dataset = base_client_data.create_tf_dataset_for_client(client_id)
      base_dataset_length = _get_dataset_size(base_dataset)
      num_pseudo_clients = int(
          tf.math.ceil(base_dataset_length / examples_per_pseudo_client))
      expanded_client_ids = [
          client_id + separator + str(i) for i in range(num_pseudo_clients)
      ]
      pseudo_client_ids += expanded_client_ids

  def serializable_dataset_fn(pseudo_client_id):
    split_client_id = tf.strings.split(pseudo_client_id, sep=separator)
    base_client_id = tf.strings.reduce_join(
        split_client_id[:-1], separator=separator)
    pseudo_client_index = tf.strings.to_number(
        split_client_id[-1], out_type=tf.int64)
    base_client_ds = base_client_data.serializable_dataset_fn(base_client_id)
    return _create_pseudo_client(base_client_ds, examples_per_pseudo_client,
                                 pseudo_client_index)

  return tff.simulation.datasets.ClientData.from_clients_and_tf_fn(
      pseudo_client_ids, serializable_dataset_fn)
