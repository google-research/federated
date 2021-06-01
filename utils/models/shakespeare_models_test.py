# Copyright 2019, Google LLC.
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

import tensorflow as tf

from utils import keras_metrics
from utils.models import shakespeare_models


class ModelsTest(tf.test.TestCase):

  def test_run_simple_model(self):
    vocab_size = 6
    mask_model = shakespeare_models.create_recurrent_model(
        vocab_size, sequence_length=5)
    mask_model.compile(
        optimizer='sgd',
        loss='sparse_categorical_crossentropy',
        metrics=[keras_metrics.MaskedCategoricalAccuracy()])

    no_mask_model = shakespeare_models.create_recurrent_model(
        vocab_size, sequence_length=5, mask_zero=False)
    no_mask_model.compile(
        optimizer='sgd',
        loss='sparse_categorical_crossentropy',
        metrics=[keras_metrics.MaskedCategoricalAccuracy()])

    constant_test_weights = tf.nest.map_structure(tf.ones_like,
                                                  mask_model.weights)
    mask_model.set_weights(constant_test_weights)
    no_mask_model.set_weights(constant_test_weights)

    # `tf.data.Dataset.from_tensor_slices` aggresively coalesces the input into
    # a single tensor, but we want a tuple of two tensors per example, so we
    # apply a transformation to split.
    def split_to_tuple(t):
      return (t[0, :], t[1, :])

    data = tf.data.Dataset.from_tensor_slices([
        ([0, 1, 2, 3, 4], [1, 2, 3, 4, 0]),
        ([2, 3, 4, 0, 1], [3, 4, 0, 1, 2]),
    ]).map(split_to_tuple).batch(2)
    mask_metrics = mask_model.evaluate(data)
    no_mask_metrics = no_mask_model.evaluate(data)

    self.assertNotAllClose(mask_metrics, no_mask_metrics, atol=1e-3)

  def test_model_initialization_uses_random_seed(self):
    model_1_with_seed_0 = shakespeare_models.create_recurrent_model(
        vocab_size=6, sequence_length=5, seed=0)
    model_2_with_seed_0 = shakespeare_models.create_recurrent_model(
        vocab_size=6, sequence_length=5, seed=0)
    model_1_with_seed_1 = shakespeare_models.create_recurrent_model(
        vocab_size=6, sequence_length=5, seed=1)
    model_2_with_seed_1 = shakespeare_models.create_recurrent_model(
        vocab_size=6, sequence_length=5, seed=1)
    self.assertAllClose(model_1_with_seed_0.weights,
                        model_2_with_seed_0.weights)
    self.assertAllClose(model_1_with_seed_1.weights,
                        model_2_with_seed_1.weights)
    self.assertNotAllClose(model_1_with_seed_0.weights,
                           model_1_with_seed_1.weights)


if __name__ == '__main__':
  tf.test.main()
