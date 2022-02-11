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

import collections
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from data_poor_fl import personalization_utils


def _create_dataset():
  """Constructs an unbatched dataset with three datapoints."""
  return tf.data.Dataset.from_tensor_slices({
      'x': np.array([[-1.0, -1.0], [1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
      'y': np.array([[1.0], [1.0], [1.0]], dtype=np.float32)
  })


def _model_fn():
  """Constructs a simple linear model initialized with zeros."""
  input_dim = 2
  output_dim = 1
  inputs = tf.keras.Input(shape=(input_dim,))
  output = tf.keras.layers.Dense(output_dim, kernel_initializer='zeros')(inputs)
  keras_model = tf.keras.Model(inputs=inputs, outputs=output)
  input_spec = collections.OrderedDict(
      x=tf.TensorSpec([None, input_dim], dtype=tf.float32),
      y=tf.TensorSpec([None, output_dim], dtype=tf.float32))
  return tff.learning.from_keras_model(
      keras_model=keras_model,
      input_spec=input_spec,
      loss=tf.keras.losses.MeanSquaredError(),
      metrics=[tf.keras.metrics.MeanAbsoluteError()])


class PersonalizationUtilsTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      ('even_examples', tf.data.Dataset.range(8), [0, 1, 2, 3], [4, 5, 6, 7]),
      ('odd_examples', tf.data.Dataset.range(7), [0, 1, 2], [3, 4, 5, 6]))
  def test_split_half(self, input_data, expected_first, expected_second):
    first_data, second_data = personalization_utils.split_half(input_data)
    self.assertListEqual(list(first_data.as_numpy_iterator()), expected_first)
    self.assertListEqual(list(second_data.as_numpy_iterator()), expected_second)

  def test_evaluate_fn(self):
    eval_metrics = personalization_utils.evaluate_fn(
        model=_model_fn(), dataset=_create_dataset())
    self.assertDictEqual(
        eval_metrics,
        collections.OrderedDict(
            # Evaluation uses batch size 1, so `num_batches` == `num_examples`.
            mean_absolute_error=1.0,
            loss=1.0,
            num_examples=3,
            num_batches=3))

  def test_build_and_run_personalize_fn(self):
    personalize_fn = personalization_utils.build_personalize_fn(
        optimizer_fn=lambda: tf.keras.optimizers.SGD(0.5),
        batch_size=2,
        max_num_epochs=1)
    p13n_metrics = personalize_fn(
        model=_model_fn(),
        train_data=_create_dataset(),
        test_data=_create_dataset())
    self.assertDictEqual(
        p13n_metrics,
        # The model weights become [0, 0, 1] after training one epoch, so both
        # the `loss` and `MeanAbsoluteError` are 0. The `personalize_fn`
        # will split the `test_data` (of size 3) into a validation set and a
        # test set, so valid `num_examples` is 1 and test `num_examples` is 2.
        collections.OrderedDict(
            epoch_1_valid=collections.OrderedDict(
                mean_absolute_error=0.0,
                loss=0.0,
                num_examples=1,
                num_batches=1),
            epoch_1_test=collections.OrderedDict(
                mean_absolute_error=0.0,
                loss=0.0,
                num_examples=2,
                num_batches=1),
            num_train_examples=3))

  def test_postprocess_metrics(self):
    # Constructs a fake dict representing the personalization eval metrics
    # collected from 2 clients (so each leaf is a list of size 2).
    metrics_dict = collections.OrderedDict(
        baseline_metrics=collections.OrderedDict(
            valid=collections.OrderedDict(
                num_examples=[1, 3], accuracy=[0.1, 0.1]),
            test=collections.OrderedDict(
                num_examples=[2, 3], accuracy=[0.2, 0.1])),
        finetuning=collections.OrderedDict(
            epoch_1_valid=collections.OrderedDict(
                num_examples=[1, 3], accuracy=[0.0, 0.3]),
            epoch_1_test=collections.OrderedDict(
                num_examples=[2, 3], accuracy=[0.1, 0.2]),
            num_train_examples=[3, 6]))
    expected_processed_metrics = collections.OrderedDict(
        baseline_metrics=collections.OrderedDict(
            valid_accuracies_mean=0.1,  # mean([0.1, 0.1])
            valid_num_examples_mean=2,  # mean([1, 3])
            test_accuracies_mean=0.15,  # mean([0.2, 0.1])
            test_num_examples_mean=2.5),  # mean([2, 3])
        finetuning=collections.OrderedDict(
            # The baseline average valid accuracy is 0.1. At Epoch 1, the mean
            # valid accuracy is mean([0.0, 0.3]) = 0.15, so the best epoch is 1.
            best_epoch=1,
            valid_accuracies_at_best_epoch_mean=0.15,  # mean([0.0, 0.3])
            test_accuracies_at_best_epoch_mean=0.15,  # mean([0.1, 0.2])
            # The first client's test accuracy at Epoch 1 (0.1) is lower than
            # the baseline accuracy (0.2). The second client's test accuracy at
            # Epoch 1 (0.2) is higher than the baseline accuracy (0.1). Hence,
            # the fraction of clients hurt at Epoch 1 is 1/2 = 0.5.
            fraction_clients_hurt_at_best_epoch=0.5))
    processed_metrics = personalization_utils.postprocess_finetuning_metrics(
        metrics_dict, 'accuracy', 'finetuning')
    tf.nest.map_structure(self.assertAllClose, processed_metrics,
                          expected_processed_metrics)


if __name__ == '__main__':
  tf.test.main()
