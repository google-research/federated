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
"""Canary insertion framework for privacy auditing."""

import tensorflow as tf
import tensorflow_federated as tff


def add_canaries(
    client_data: tff.simulation.datasets.ClientData,
    num_canaries: int,
) -> tff.simulation.datasets.ClientData:
  """Constructs a dataset with real and canary clients.

  Real clients have client IDs of the form 'real:{orig_client_id}' for each
  'orig_client_id' in `client_data`, and their data is taken from the original
  dataset. Canary clients have client IDs of the form 'canary:{i}' for index
  'i' and have an empty dataset. They are intended to be used in a training loop
  which provides each canary user with an update that is a pseudorandom vector
  seeded by their index.

  Args:
    client_data: Original `tff.simulation.datasets.ClientData` to which we add
      canaries.
    num_canaries: The number of canaries to add. May be zero, in which case the
      client_data is simply prefixed with 'real:'.

  Returns:
    Dataset that contains both real and canary clients.

  Raises:
    ValueError: If make_transform_fn is not None and num_canaries is larger
    than the number of clients.
  """
  if num_canaries < 0:
    raise ValueError(
        f'Number of canaries must be non-negative. Found {num_canaries}.'
    )

  def assert_is_fully_defined(spec):
    if not spec.shape.is_fully_defined():
      raise ValueError(
          f'Tensor shape must be fully defined. Found {spec.shape}. If '
          '`client_data` is batched, you may need to add canaries before '
          'batching.'
      )

  tf.nest.map_structure(
      assert_is_fully_defined, client_data.element_type_structure
  )
  canary_client_ids = ['canary:' + str(i) for i in range(num_canaries)]

  @tf.function
  def canary_data() -> tf.data.Dataset:
    """Construct data for canary clients.

    Create a dataset with one element. When using a gradient attack,
    `take(0)` to get an empty dataset with the correct
    `element_type_structure`. When using an example attack, take a pseudo
    random example from class other than the target class.

    Returns:
      A serializable dataset function for the canary.
    """
    element = tf.nest.map_structure(
        lambda spec: tf.zeros(spec.shape, spec.dtype),
        client_data.element_type_structure,
    )
    return tf.data.Dataset.from_tensors(element).take(0)

  @tf.function
  def serializable_dataset_fn(client_id):
    split_id = tf.strings.split(client_id, sep=':', maxsplit=1)
    return tf.cond(
        tf.math.equal(split_id[0], 'canary'),
        canary_data,
        lambda: client_data.serializable_dataset_fn(split_id[1]),
    )

  real_client_ids = ['real:' + orig_id for orig_id in client_data.client_ids]
  return tff.simulation.datasets.ClientData.from_clients_and_tf_fn(
      real_client_ids + canary_client_ids, serializable_dataset_fn
  )
