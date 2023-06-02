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
"""Constants and type annotations used in the experiments."""

from typing import Callable, OrderedDict
import tensorflow as tf
import tensorflow_federated as tff

DATASET_NAMES = ['emnist', 'stackoverflow', 'landmark', 'ted_multi']
TRAIN_CLIENTS_KEY = 'train_clients'
VALID_CLIENTS_KEY = 'valid_clients'
TEST_CLIENTS_KEY = 'test_clients'
SPLIT_CLIENTS_SEED = 13
FINETUNING_FN_NAME = 'finetuning'
# In the evaluation phase, each client's dataset is shuffled and split into two
# equal-sized unbatched datasets: a personalization set and an evaluation set.
# The personalization set is used for finetuning the model in
# `finetuning_trainer` or choosing the best model in `hypcluster_trainer`.
PERSONALIZATION_DATA_KEY = 'train_data'
TEST_DATA_KEY = 'test_data'

ModelFnType = Callable[[], tff.learning.models.VariableModel]
FederatedDatasetsType = OrderedDict[str, tff.simulation.datasets.ClientData]
ProcessFnType = Callable[[tf.data.Dataset], tf.data.Dataset]
SplitDataFnType = Callable[[tf.data.Dataset], OrderedDict[str, tf.data.Dataset]]
