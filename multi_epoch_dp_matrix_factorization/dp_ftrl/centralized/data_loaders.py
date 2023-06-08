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
"""Defines data loaders for centralized tasks."""
from collections.abc import Sequence
import random
from typing import Optional

import numpy as np
import tensorflow_datasets as tfds

DataStreamType = Sequence[tuple[np.ndarray, np.ndarray]]


def shuffle_data_stream(data: DataStreamType) -> DataStreamType:
  return random.sample(data, len(data))


def _split_data(
    data: dict[str, np.ndarray], fraction: float
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
  # Our callers guarantee that 'y' is always present.
  to_split = int(len(data['y']) * fraction)
  fraction_set = dict(**{key: val[:to_split, ...] for key, val in data.items()})
  remainder_set = dict(
      **{key: val[to_split:, ...] for key, val in data.items()}
  )
  return fraction_set, remainder_set


def _batch_and_tuple_data(
    data: dict[str, np.ndarray], batch_size: int
) -> DataStreamType:
  """Splits arrays of data to batches, and packages as pairwise tuples."""
  if len(data['y']) % batch_size != 0:
    # TODO(b/244756081): Clearly we can relax this, but it requires relaxing the
    # assumptions we make in the DPGradientProcessor which is TFF Aggregator
    # backed. This would not be too difficult (renormalizing on the outside of
    # TFF if necessary), and is something to follow up on if it becomes a
    # problem.
    raise ValueError(
        'Assumed that batch size would evenly divide the number '
        f'of training examples. Got batch size {batch_size}, '
        f'number of examples {len(data["y"])}'
    )
  split_x = np.array_split(data['x'], len(data['y']) / batch_size, axis=0)
  split_y = np.array_split(data['y'], len(data['y']) / batch_size, axis=0)
  return tuple(zip(split_x, split_y))


def get_tfds_data(
    data_name: str,
    batch_size: int,
    test_val_split: Optional[float] = None,
) -> tuple[DataStreamType, DataStreamType, DataStreamType, int]:
  """Loads TFDS data into a format consumable by the training loop.

  Args:
    data_name: String to be passed to tfds.load, specifing the dataset to be
      loaded. May be loaded entirely into memory in the body of this function,
      so this function generally should support only small-scale datasets.
    batch_size: Batch size to use for batches of the loaded data.
    test_val_split: Optional float. If unspecified, the test set will simply be
      used as the validation set. Otherwise, this float will specify the
      fraction of the original test set which will be returned as the 'test set'
      from this function, with the remainder packaged in the val set.

  Returns:
    A four-tuple. The first three elements are streams of numpy arrays,
    representing train, val, and test data. The final element is the number of
    classes present in the labels (so in particular, this function assumes it
    will load and process supervised data)
  """
  if data_name.startswith('cifar10'):
    data = tfds.as_numpy(tfds.load(name='cifar10', batch_size=-1))
    nclass = 10
    train = dict(
        x=data['train']['image'].transpose(0, 3, 1, 2) / 127.5 - 1,
        y=data['train']['label'],
    )
    test = dict(
        x=data['test']['image'].transpose(0, 3, 1, 2) / 127.5 - 1,
        y=data['test']['label'],
    )
  elif data_name in ['mnist', 'fashion_mnist', 'emnist_class', 'emnist_merge']:
    if data_name.startswith('emnist'):
      data_name = 'emnist/by' + data_name[7:]
    data, info = tfds.load(name=data_name, batch_size=-1, with_info=True)
    data = tfds.as_numpy(data)
    train = dict(
        x=data['train']['image'].transpose(0, 3, 1, 2) / 127.5 - 1,
        y=data['train']['label'],
    )
    test = dict(
        x=data['test']['image'].transpose(0, 3, 1, 2) / 127.5 - 1,
        y=data['test']['label'],
    )
    nclass = info.features['label'].num_classes
  else:
    raise ValueError(f'Invalid `data_name`={data_name} for tfds.')
  # Splitting validation from test, rather than train, is inherited from the
  # existing centralized FTAL implementation. There, the split percentage was
  # 50%. Presumably the difference here will not be noticeable.
  if test_val_split is not None:
    test, val = _split_data(test, test_val_split)
  else:
    # We add this here to enable replication of the logic from the *first*
    # DP-FTRL paper; the code has evolved somewhat since then, but using test as
    # val was indeed the pattern used there.
    val = test
  batched_train = _batch_and_tuple_data(train, batch_size)
  batched_val = _batch_and_tuple_data(val, batch_size)
  batched_test = _batch_and_tuple_data(test, batch_size)
  return batched_train, batched_val, batched_test, nclass
