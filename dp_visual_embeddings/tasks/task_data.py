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
"""Classes for loading and preprocessing data for tasks."""

import collections
from collections.abc import Callable
import itertools
from typing import Any, Optional, Union

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

CentralOrClientData = Union[tf.data.Dataset, tff.simulation.datasets.ClientData]
PreprocessFnType = Union[Callable[[tf.data.Dataset], tf.data.Dataset],
                         tff.Computation]

_CLIENT_ID_JOINER = ' '


def _random_sample_client_ids(clientdata: tff.simulation.datasets.ClientData,
                              num_clients: int,
                              dynamic_clients: int,
                              replace: bool = False,
                              random_seed: Optional[int] = None) -> list[str]:
  """Random sampling clients from a TFF client dataset.

  Args:
    clientdata: A `tff.simulation.datasets.ClientData` to be sampled.
    num_clients: A positive integer representing number of clients to be
      sampled. Each client represents a "super" client merged from
      `dynamic_clients` clients in clientdata.
    dynamic_clients: Dynamically merge clients to super clients for user-level
        DP. It is used to generate samples.
    replace: Whether to sample with replacement. If set to `False`, then
      `num_clients` cannot exceed the number of training clients in the
      associated train data.
    random_seed: An optional integer used to set a random seed for sampling. If
      no random seed is passed or the random seed is set to `None`, this will
      attempt to set the random seed according to the current system time (see
      `numpy.random.RandomState` for details).

  Returns:
    A list of `str` client ids.
  """
  random_state = np.random.RandomState(seed=random_seed)
  total_clients = num_clients * dynamic_clients
  client_ids = random_state.choice(
      clientdata.client_ids, size=total_clients, replace=replace)
  if dynamic_clients == 1:
    return client_ids
  else:
    # See itertools library recipe (`grouper` method) for context on below
    # (https://docs.python.org/3/library/itertools.html#itertools-recipes).
    args = [iter(client_ids)] * dynamic_clients
    return [
        _CLIENT_ID_JOINER.join(ids)
        for ids in itertools.zip_longest(*args, fillvalue='')
    ]


@tf.function
def _get_super_client(client_data: tff.simulation.datasets.ClientData,
                      super_client_id: str,
                      dynamic_clients: int) -> tf.data.Dataset:
  client_ids = tf.unstack(
      tf.strings.split(super_client_id, sep=_CLIENT_ID_JOINER),
      num=dynamic_clients)
  datasets = [client_data.serializable_dataset_fn(cid) for cid in client_ids]
  return tf.data.Dataset.sample_from_datasets(datasets)


def _random_sample_clients(
    clientdata: tff.simulation.datasets.ClientData,
    serializable_dataset_fn: Callable[[str], tf.data.Dataset],
    num_clients: int,
    dynamic_clients: int,
    replace: bool = False,
    random_seed: Optional[int] = None) -> list[tf.data.Dataset]:
  """Random sampling clients from a TFF client dataset.

  Args:
    clientdata: A `tff.simulation.datasets.ClientData` to be sampled; providing
      `clientdata.client_ids`.
    serializable_dataset_fn: A function that returns a client `tf.data.dataset`
      by taking an input argument for client ID.
    num_clients: A positive integer representing number of clients to be
      sampled.
    dynamic_clients: Dynamically merge clients to super clients for user-level
        DP. It is used to generate samples.
    replace: Whether to sample with replacement. If set to `False`, then
      `num_clients` cannot exceed the number of training clients in the
      associated train data.
    random_seed: An optional integer used to set a random seed for sampling. If
      no random seed is passed or the random seed is set to `None`, this will
      attempt to set the random seed according to the current system time (see
      `numpy.random.RandomState` for details).

  Returns:
    A list of `tf.data.Dataset` instances representing the client datasets.
  """
  client_ids = _random_sample_client_ids(
      clientdata,
      num_clients,
      dynamic_clients=dynamic_clients,
      replace=replace,
      random_seed=random_seed)
  return [serializable_dataset_fn(cid) for cid in client_ids]


def _get_element_spec(data: CentralOrClientData,
                      preprocess_fn: Optional[PreprocessFnType] = None):
  """Determines the element type of a dataset after preprocessing."""
  if isinstance(data, tff.simulation.datasets.ClientData):
    if preprocess_fn is not None:
      preprocessed_data = data.preprocess(preprocess_fn)
    else:
      preprocessed_data = data
    element_spec = preprocessed_data.element_type_structure
  else:
    if preprocess_fn is not None:
      preprocessed_data = preprocess_fn(data)
    else:
      preprocessed_data = data
    element_spec = preprocessed_data.element_spec
  return element_spec


def _get_record_type(client_data: CentralOrClientData) -> tuple[str, str]:
  """Returns a tuple of strings for data type and number of clients."""
  if isinstance(client_data, tff.simulation.datasets.ClientData):
    data_type = 'Federated'
    num_clients = str(len(client_data.client_ids))
  else:
    data_type = 'Centralized'
    num_clients = 'N/A'
  return data_type, num_clients


def _get_centralized_from_clientdata(
    client_data: tff.simulation.datasets.ClientData,
    preprocess_fn: Optional[PreprocessFnType]) -> tf.data.Dataset:
  centralized_data = client_data.create_tf_dataset_from_all_clients()
  if preprocess_fn is not None:
    centralized_data = preprocess_fn(centralized_data)
  return centralized_data


def _get_centralized_from_dataset(
    dataset: tf.data.Dataset,
    preprocess_fn: Optional[PreprocessFnType]) -> tf.data.Dataset:
  if preprocess_fn is not None:
    return preprocess_fn(dataset)
  else:
    return dataset


def _get_serializable_dataset_fn(
    client_data: tff.simulation.datasets.ClientData,
    preprocess_fn: Optional[PreprocessFnType],
    dynamic_clients: int) -> Callable[[str], tf.data.Dataset]:
  """Returns `serializable_dataset_fn` for getting preprocessed client data."""

  if dynamic_clients > 1:

    def serializable_client_fn(super_client_id):
      return _get_super_client(client_data, super_client_id, dynamic_clients)
  elif dynamic_clients == 1:
    serializable_client_fn = client_data.serializable_dataset_fn

  if preprocess_fn is None:
    serializable_dataset_fn = serializable_client_fn
  else:

    def serializable_dataset_fn(client_id):
      return preprocess_fn(serializable_client_fn(client_id))

  return serializable_dataset_fn


class EmbeddingTaskDatasets(object):
  """A convenience class for a task's data and preprocessing logic.

  Attributes:
    train_data: A `tff.simulation.datasets.ClientData` for training.
    test_data: The test data for the baseline task. Can be a
      `tff.simulation.datasets.ClientData` or a `tf.data.Dataset`.
    validation_data:  A `tff.simulation.datasets.ClientData` for validation.
    train_dataset_computation: A `tff.tf_computation` returns a preprocessed
      dataset for client ID in training data. Set to `None` for centralized
      training dataset.
    validation_dataset_computation: A `tff.tf_computation` returns a
      preprocessed dataset for client ID in validation data. Set to `None` for
      centralized validation dataset.
    train_preprocess_fn: A callable mapping accepting and return
      `tf.data.Dataset` instances, used for preprocessing train datasets. Set to
      `None` if no train preprocessing occurs for the task.
    validation_preprocess_fn: A callable mapping accepting and return
      `tf.data.Dataset` instances, used for preprocessing validation datasets.
      Set to `None` if no eval preprocessing occurs for the task.
    test_preprocess_fn: A callable mapping accepting and return
      `tf.data.Dataset` instances, used for preprocessing test datasets. Set to
      `None` if no eval preprocessing occurs for the task.
    element_type_structure: A nested structure of `tf.TensorSpec` objects
      defining the type of the elements contained in datasets associated to this
      task.
  """

  def __init__(self,
               *,
               train_data: CentralOrClientData,
               test_data: CentralOrClientData,
               validation_data: CentralOrClientData,
               centralized_train_data: Optional[tf.data.Dataset] = None,
               train_preprocess_fn: Optional[PreprocessFnType] = None,
               validation_preprocess_fn: Optional[PreprocessFnType] = None,
               test_preprocess_fn: Optional[PreprocessFnType] = None,
               dynamic_clients: int = 1):
    """Creates a `BaselineTaskDatasets`.

    This defines training, validation, and test datasets. Each datasets may be
    accessed as either a centralized dataset or a federated dataset. A
    centralized dataset will be a `tf.data.Dataset` that yields all examples in
    one dataset. The federated dataset will contain the same set of examples,
    but as a `tff.simulation.datasets.ClientData` from which one can request the
    examples for individual clients.

    The centralized datasets will either come from the
    `create_tf_dataset_from_all_clients` method of `ClientData` or, in the case
    of the train dataset when `centralized_train_data` is set to a value other
    than `None`, it will come directly from that `tf.data.Dataset`.

    Args:
      train_data: A `tff.simulation.datasets.ClientData` for training.
      test_data: An optional `tff.simulation.datasets.ClientData` for computing
        test metrics.
      validation_data: A `tff.simulation.datasets.ClientData` validation.
      centralized_train_data: Optional centralized version of the training data.
      train_preprocess_fn: An optional callable accepting and returning a
        `tf.data.Dataset`, used to perform dataset preprocessing for training.
        If set to `None`, we use the identity map for all train preprocessing.
      validation_preprocess_fn: An optional callable accepting and returning a
        `tf.data.Dataset`, used to perform validation preprocessing. If `None`,
        validation preprocessing will be done via the identity map.
      test_preprocess_fn: An optional callable accepting and returning a
        `tf.data.Dataset`, used to perform validation, testing preprocessing. If
        `None`, testing preprocessing will be done via the identity map.
      dynamic_clients: Dynamically merge clients to super clients for user-level
        DP. It is used to generate samples.

    Raises:
      ValueError: If `train_data` and `test_data` have different element types
        after preprocessing with `train_preprocess_fn` and `eval_preprocess_fn`,
        or if `validation_data` has a different element type than the test data.
    """
    self._train_data = train_data
    self._test_data = test_data
    self._validation_data = validation_data
    self._centralized_train_data = centralized_train_data
    self._dynamic_clients = dynamic_clients

    if (train_preprocess_fn is not None and not callable(train_preprocess_fn)):
      raise ValueError('The train_preprocess_fn must be None or callable.')
    self._train_preprocess_fn = train_preprocess_fn

    if (validation_preprocess_fn
        is not None) and (not callable(validation_preprocess_fn)):
      raise ValueError('The validation_preprocess_fn must be None or callable.')
    self._validation_preprocess_fn = validation_preprocess_fn

    if (test_preprocess_fn is not None) and (not callable(test_preprocess_fn)):
      raise ValueError('The test_preprocess_fn must be None or callable.')
    self._test_preprocess_fn = test_preprocess_fn

    post_preprocess_train_type = _get_element_spec(train_data,
                                                   train_preprocess_fn)
    post_preprocess_validation_type = _get_element_spec(
        validation_data, validation_preprocess_fn)
    post_preprocess_test_type = _get_element_spec(test_data, test_preprocess_fn)
    if post_preprocess_train_type != post_preprocess_test_type:
      raise ValueError(
          'The train and test element structures after preprocessing must be '
          'equal. Found train type {} and test type {}'.format(
              post_preprocess_train_type, post_preprocess_test_type))
    if post_preprocess_test_type != post_preprocess_validation_type:
      raise ValueError(
          'The validation and test element structures after preprocessing must '
          'be equal. Found validation type {} and test type {}'.format(
              post_preprocess_validation_type, post_preprocess_test_type))
    self._element_type_structure = post_preprocess_train_type

    if isinstance(self._train_data, tff.simulation.datasets.ClientData):
      self._train_serializable_dataset_fn = _get_serializable_dataset_fn(
          train_data, train_preprocess_fn, dynamic_clients=dynamic_clients)

      @tff.tf_computation(tf.string)
      def train_dataset_computation(client_id):
        return self._train_serializable_dataset_fn(client_id)

      self._train_dataset_computation = train_dataset_computation
    else:
      self._train_serializable_dataset_fn = None
      self._train_dataset_computation = None
    if isinstance(self._validation_data, tff.simulation.datasets.ClientData):
      self._validation_serializable_dataset_fn = _get_serializable_dataset_fn(
          validation_data,
          validation_preprocess_fn,
          dynamic_clients=dynamic_clients)

      @tff.tf_computation(tf.string)
      def validation_dataset_computation(client_id):
        return self._validation_serializable_dataset_fn(client_id)

      self._validation_dataset_computation = validation_dataset_computation
    else:
      self._validation_serializable_dataset_fn = None
      self._validation_dataset_computation = None

    self._data_info = None

  @property
  def train_data(self) -> CentralOrClientData:
    return self._train_data

  @property
  def test_data(self) -> CentralOrClientData:
    return self._test_data

  @property
  def validation_data(self) -> CentralOrClientData:
    return self._validation_data

  @property
  def train_dataset_computation(self) -> tff.tf_computation:
    return self._train_dataset_computation

  @property
  def validation_dataset_computation(self) -> tff.tf_computation:
    return self._validation_dataset_computation

  @property
  def train_preprocess_fn(self) -> Optional[PreprocessFnType]:
    return self._train_preprocess_fn

  @property
  def validation_preprocess_fn(self) -> Optional[PreprocessFnType]:
    return self._validation_preprocess_fn

  @property
  def test_preprocess_fn(self) -> Optional[PreprocessFnType]:
    return self._validation_preprocess_fn

  @property
  def element_type_structure(self):
    return self._element_type_structure

  def _record_dataset_information(self):
    """Records a summary of the train, test, and validation data."""
    data_info = collections.OrderedDict()
    data_info['header'] = ['Split', 'Dataset Type', 'Number of Clients']

    train_type, num_train_clients = _get_record_type(self._train_data)
    data_info['train'] = ['Train', train_type, num_train_clients]

    test_type, num_test_clients = _get_record_type(self._test_data)
    data_info['test'] = ['Test', test_type, num_test_clients]

    validation_type, num_validation_clients = _get_record_type(
        self._validation_data)
    data_info['validation'] = [
        'Validation', validation_type, num_validation_clients
    ]
    return data_info

  def _sample_train_clients(
      self,
      num_clients: int,
      replace: bool = False,
      random_seed: Optional[int] = None) -> list[tf.data.Dataset]:
    """Samples training clients uniformly at random.

    Args:
      num_clients: A positive integer representing number of clients to be
        sampled.
      replace: Whether to sample with replacement. If set to `False`, then
        `num_clients` cannot exceed the number of training clients in the
        associated train data.
      random_seed: An optional integer used to set a random seed for sampling.
        If no random seed is passed or the random seed is set to `None`, this
        will attempt to set the random seed according to the current system time
        (see `numpy.random.RandomState` for details).

    Returns:
      A list of `tf.data.Dataset` instances representing the client datasets.
    """
    if isinstance(self._train_data, tff.simulation.datasets.ClientData):
      return _random_sample_clients(
          self._train_data,
          self._train_serializable_dataset_fn,
          num_clients=num_clients,
          dynamic_clients=self._dynamic_clients,
          replace=replace,
          random_seed=random_seed)
    else:
      raise TypeError('Cannot sample clients from train data.')

  def sample_train_client_ids(self,
                              num_clients: int,
                              replace: bool = False,
                              random_seed: Optional[int] = None) -> list[str]:
    """Samples training client ids uniformly at random.

    For use with a TFF iterative process that was composed with
    `tff.Computation` that constructs a preprocessed dataset given a client id
    (using `tff.simulation.compose_dataset_computation_with_iterative_process`
    `tff.Computation`). This pattern can accelerate TFF runtime in multi-machine
    environment.

    Args:
      num_clients: A positive integer representing number of clients to be
        sampled.
      replace: Whether to sample with replacement. If set to `False`, then
        `num_clients` cannot exceed the number of training clients in the
        associated train data.
      random_seed: An optional integer used to set a random seed for sampling.
        If no random seed is passed or the random seed is set to `None`, this
        will attempt to set the random seed according to the current system time
        (see `numpy.random.RandomState` for details).

    Returns:
      A list of `str` client ids.
    """
    if isinstance(self._train_data, tff.simulation.datasets.ClientData):
      return _random_sample_client_ids(
          self._train_data,
          num_clients=num_clients,
          dynamic_clients=self._dynamic_clients,
          replace=replace,
          random_seed=random_seed)
    else:
      raise TypeError('Cannot sample clients from train data.')

  def _sample_validation_clients(
      self,
      num_clients: int,
      replace: bool = False,
      random_seed: Optional[int] = None) -> list[tf.data.Dataset]:
    """Samples training clients uniformly at random.

    Args:
      num_clients: A positive integer representing number of clients to be
        sampled.
      replace: Whether to sample with replacement. If set to `False`, then
        `num_clients` cannot exceed the number of training clients in the
        associated train data.
      random_seed: An optional integer used to set a random seed for sampling.
        If no random seed is passed or the random seed is set to `None`, this
        will attempt to set the random seed according to the current system time
        (see `numpy.random.RandomState` for details).

    Returns:
      A list of `tf.data.Dataset` instances representing the client datasets.
    """
    if isinstance(self._validation_data, tff.simulation.datasets.ClientData):
      return _random_sample_clients(
          self._validation_data,
          self._validation_serializable_dataset_fn,
          dynamic_clients=self._dynamic_clients,
          num_clients=num_clients,
          replace=replace,
          random_seed=random_seed)
    else:
      raise TypeError('Cannot sample clients from validation data.')

  def sample_validation_client_ids(
      self,
      num_clients: int,
      replace: bool = False,
      random_seed: Optional[int] = None) -> list[str]:
    """Samples training clients uniformly at random.

    For use with a TFF iterative process that was composed with
    `tff.Computation` that constructs a preprocessed dataset given a client id
    (using `tff.simulation.compose_dataset_computation_with_iterative_process`
    `tff.Computation`). This pattern can accelerate TFF runtime in multi-machine
    environment.

    Args:
      num_clients: A positive integer representing number of clients to be
        sampled.
      replace: Whether to sample with replacement. If set to `False`, then
        `num_clients` cannot exceed the number of training clients in the
        associated train data.
      random_seed: An optional integer used to set a random seed for sampling.
        If no random seed is passed or the random seed is set to `None`, this
        will attempt to set the random seed according to the current system time
        (see `numpy.random.RandomState` for details).

    Returns:
      A list of `str` client ids.
    """
    if isinstance(self._validation_data, tff.simulation.datasets.ClientData):
      return _random_sample_client_ids(
          self._validation_data,
          dynamic_clients=self._dynamic_clients,
          num_clients=num_clients,
          replace=replace,
          random_seed=random_seed)
    else:
      raise TypeError('Cannot sample clients from validation data.')

  def get_centralized_train_data(self) -> tf.data.Dataset:
    """Returns a `tf.data.Dataset` with all clients` training data for the task.

    This method will first amalgamate the client datasets into a single dataset,
    then apply preprocessing.
    """
    if self._centralized_train_data is not None:
      return _get_centralized_from_dataset(self._centralized_train_data,
                                           self._train_preprocess_fn)
    elif isinstance(self._train_data, tff.simulation.datasets.ClientData):
      return _get_centralized_from_clientdata(self._train_data,
                                              self._train_preprocess_fn)
    else:
      return _get_centralized_from_dataset(self._train_data,
                                           self._train_preprocess_fn)

  def get_centralized_validation_data(self) -> tf.data.Dataset:
    """Returns a `tf.data.Dataset` with all clients` training data for the task.

    This method will first amalgamate the client datasets into a single dataset,
    then apply preprocessing.
    """
    if isinstance(self._validation_data, tff.simulation.datasets.ClientData):
      return _get_centralized_from_clientdata(self._validation_data,
                                              self._validation_preprocess_fn)
    else:
      return _get_centralized_from_dataset(self._validation_data,
                                           self._validation_preprocess_fn)

  def get_centralized_test_data(self) -> tf.data.Dataset:
    """Returns a `tf.data.Dataset` of test data for the task.

    If the baseline task has centralized data, then this method will return
    the centralized data after applying preprocessing. If the test data is
    federated, then this method will first amalgamate the client datasets into
    a single dataset, then apply preprocessing.
    """
    if isinstance(self._test_data, tff.simulation.datasets.ClientData):
      return _get_centralized_from_clientdata(self._test_data,
                                              self._test_preprocess_fn)
    else:
      return _get_centralized_from_dataset(self._test_data,
                                           self._test_preprocess_fn)

  def summary(self, print_fn: Callable[[str], Any] = print):
    """Prints a summary of the train, test, and validation data.

    The summary will be printed as a table containing information on the type
    of train, test, and validation data (ie. federated or centralized) and the
    number of clients each data structure has (if it is federated). For example,
    if the train data has 10 clients, validation data has 5 clients, and the
    test data is centralized, then this will print the following table:

    ```
    Split      |Dataset Type |Number of Clients |
    =============================================
    Train      |Federated    |10                |
    Test       |Centralized  |N/A               |
    Validation |Federated    |5                 |
    _____________________________________________
    ```

    In addition, this will print two lines after the table indicating whether
    train and eval preprocessing functions were passed in. In the example above,
    if we passed in a train preprocessing function but no eval preprocessing
    function, it would also print the lines:
    ```
    Train Preprocess Function: True
    Eval Preprocess Function: False
    ```

    To capture the summary, you can use a custom print function. For example,
    setting `print_fn = summary_list.append` will cause each of the lines above
    to be appended to `summary_list`.

    Args:
      print_fn: An optional callable accepting string inputs. Used to print each
        row of the summary. Defaults to `print` if not specified.
    """
    if self._data_info is None:
      self._data_info = self._record_dataset_information()
    data_info = self._data_info
    num_cols = len(data_info['header'])
    max_lengths = [0 for _ in range(num_cols)]
    for col_values in data_info.values():
      for j, col_value in enumerate(col_values):
        max_lengths[j] = max([len(str(col_value)), max_lengths[j]])

    col_lengths = [a + 1 for a in max_lengths]

    row_strings = []
    for col_values in data_info.values():
      row_string = ''
      for (col_val, col_len) in zip(col_values, col_lengths):
        row_string += '{col_val:<{col_len}}|'.format(
            col_val=col_val, col_len=col_len)
      row_strings.append(row_string)

    total_width = sum(col_lengths) + num_cols
    row_strings.insert(1, '=' * total_width)
    row_strings.append('_' * total_width)

    for x in row_strings:
      print_fn(x)

    train_preprocess_fn_exists = (self._train_preprocess_fn is not None)
    print_fn('Train Preprocess Function: {}'.format(train_preprocess_fn_exists))

    validation_preprocess_fn_exists = (
        self._validation_preprocess_fn is not None)
    print_fn('Validation Preprocess Function: {}'.format(
        validation_preprocess_fn_exists))

    test_preprocess_fn_exists = (self._test_preprocess_fn is not None)
    print_fn('Test Preprocess Function: {}'.format(test_preprocess_fn_exists))
