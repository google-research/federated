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
"""Constants, type annotations, and tuned parameters for the experiments."""

import tensorflow as tf

# CANARY TYPES
NO_CANARY = 'no_canary'
STATIC_DATA_CANARY = 'static_data'
RANDOM_GRADIENT_CANARY = 'random_gradient'
CANARY_TYPES = [
    STATIC_DATA_CANARY,
    RANDOM_GRADIENT_CANARY,
    NO_CANARY,
]

# MODEL TYPES
LINEAR_MODEL = 'linear'
MLP_MODEL = 'mlp'
MODEL_TYPES = [
    LINEAR_MODEL,
    MLP_MODEL,
]

# DATASETS
FASHION_MNIST_DATASET = 'fashion_mnist'
PURCHASE_DATASET = 'purchase'
DATASET_NAMES = [FASHION_MNIST_DATASET, PURCHASE_DATASET]

# Types
DatasetTupleType = tuple[
    tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset
]


# Tuned constants
def get_clip_norm(
    dataset: str, model_type: str, dp_eps: float, run_nonprivate: bool
) -> float:
  """Get clip norm based on tuned hyperparameters."""
  if run_nonprivate or dp_eps >= 1e5:  # Treat it as no DP.
    return 1e10  # Extremely large clip norm, should never be reached.
  if dataset == FASHION_MNIST_DATASET and model_type == MLP_MODEL:
    # FashionMNIST + MLP
    clip_norm_dict = {
        1.0: 2.0,
        2.0: 4.0,
        4.0: 4.0,
        8.0: 8.0,
        16.0: 8.0,
        32.0: 8.0,
    }
    if dp_eps not in clip_norm_dict:
      raise ValueError('DP_EPSILON = %s not known.' % dp_eps)
    return clip_norm_dict[dp_eps]
  elif dataset == FASHION_MNIST_DATASET and model_type == LINEAR_MODEL:
    # FashionMNIST + Linear
    clip_norm_dict = {
        1.0: 4.0,
        2.0: 4.0,
        4.0: 4.0,
        8.0: 8.0,
        16.0: 8.0,
        32.0: 8.0,
    }
    if dp_eps not in clip_norm_dict:
      raise ValueError('DP_EPSILON = %s not known.' % dp_eps)
    return clip_norm_dict[dp_eps]
  elif dataset == PURCHASE_DATASET and model_type == MLP_MODEL:
    # Purchase + MLP
    clip_norm_dict = {
        1.0: 0.25,
        2.0: 0.5,
        4.0: 1.0,
        8.0: 1.0,
        16.0: 1.0,
        32.0: 2.0,
    }
    if dp_eps not in clip_norm_dict:
      raise ValueError('DP_EPSILON = %s not known.' % dp_eps)
    return clip_norm_dict[dp_eps]
  else:
    raise ValueError('Unknown dataset-model: %s, %s' % (dataset, model_type))
