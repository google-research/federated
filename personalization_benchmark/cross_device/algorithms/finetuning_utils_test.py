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

from personalization_benchmark.cross_device.algorithms import finetuning_utils


def _create_dataset():
  """Constructs an unbatched dataset with three datapoints."""
  return tf.data.Dataset.from_tensor_slices({
      'x': np.array([[-1.0, -1.0], [1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
      'y': np.array([[1.0], [1.0], [1.0]], dtype=np.float32)
  })


def _model_fn(num_layers: int = 1, initializer='zeros'):
  """Constructs a simple multi-layer model initialized with zeros."""
  input_dim = 2
  output_dim = 1
  inputs = tf.keras.Input(shape=(input_dim,))
  output = tf.keras.layers.Dense(
      output_dim, kernel_initializer=initializer)(
          inputs)
  for _ in range(num_layers - 1):
    output = tf.keras.layers.Dense(
        output_dim, kernel_initializer=initializer)(
            output)
  keras_model = tf.keras.Model(inputs=inputs, outputs=output)
  input_spec = collections.OrderedDict(
      x=tf.TensorSpec([None, input_dim], dtype=tf.float32),
      y=tf.TensorSpec([None, output_dim], dtype=tf.float32))
  return tff.learning.from_keras_model(
      keras_model=keras_model,
      input_spec=input_spec,
      loss=tf.keras.losses.MeanSquaredError(),
      metrics=[tf.keras.metrics.MeanAbsoluteError()])


class FinetuningUtilsTest(parameterized.TestCase, tf.test.TestCase):

  def test_evaluate_fn(self):
    eval_metrics = finetuning_utils.evaluate_fn(
        model=_model_fn(), dataset=_create_dataset())
    self.assertDictEqual(
        eval_metrics,
        collections.OrderedDict(
            # Evaluation uses batch size 1, so `num_batches` == `num_examples`.
            mean_absolute_error=1.0,
            loss=1.0,
            num_examples=3,
            num_batches=3))

  @parameterized.named_parameters(('finetune_last_layer', True),
                                  ('finetune_all_layers', False))
  def test_build_and_run_finetune_eval_fn(self, finetune_last_layer):
    finetune_eval_fn = finetuning_utils.build_finetune_eval_fn(
        optimizer_fn=lambda: tf.keras.optimizers.SGD(0.5),
        batch_size=2,
        num_finetuning_epochs=1,
        # The model has only 1 layer, so it does not matter whether we finetune
        # the last layer or all layers.
        finetune_last_layer=finetune_last_layer)
    finetuning_metrics = finetune_eval_fn(
        model=_model_fn(num_layers=1),
        train_data=_create_dataset(),
        test_data=_create_dataset())
    self.assertDictEqual(
        finetuning_metrics,
        # The model weights become [0, 0, 1] after training one epoch, so both
        # the `loss` and `MeanAbsoluteError` are 0. The batch size is 2 and the
        # number of examples is 3, so there are 2 batches.
        collections.OrderedDict(
            epoch_1=collections.OrderedDict(
                mean_absolute_error=0.0,
                loss=0.0,
                num_examples=3,
                num_batches=2),
            num_train_examples=3))

  def test_run_finetune_eval_fn_only_finetune_last_layer(self):
    finetune_eval_fn = finetuning_utils.build_finetune_eval_fn(
        optimizer_fn=lambda: tf.keras.optimizers.SGD(1.0),
        batch_size=2,
        num_finetuning_epochs=1,
        finetune_last_layer=True)
    model = _model_fn(num_layers=5, initializer='ones')
    initial_model_weights_tensors = tf.nest.map_structure(
        lambda var: var.numpy(), model.trainable_variables)
    finetune_eval_fn(
        model=model, train_data=_create_dataset(), test_data=_create_dataset())
    final_model_weights_tensors = tf.nest.map_structure(
        lambda var: var.numpy(), model.trainable_variables)
    # Assert that the final model weights have the same values as the initial
    # model weights for the first n-2 tensors, and different values for the last
    # 2 tensors (note the last layer has two tensors: kernal and bias).
    num_weight_tensors = len(initial_model_weights_tensors)
    for i in range(num_weight_tensors - 2):
      self.assertAllEqual(initial_model_weights_tensors[i],
                          final_model_weights_tensors[i])
    for i in [num_weight_tensors - 2, num_weight_tensors - 1]:
      self.assertNotAllEqual(initial_model_weights_tensors[i],
                             final_model_weights_tensors[i])

  def test_postprocess_finetuning_metrics(self):
    accuracy_name = 'accuracy'
    finetuning_fn_name = 'finetuning'
    baseline_metrics_name = finetuning_utils._BASELINE_METRICS
    num_examples_name = finetuning_utils._NUM_TEST_EXAMPLES
    num_finetune_examples_name = finetuning_utils._NUM_FINETUNE_EXAMPLES
    # Constructs a fake dictionary representing the finetuning eval metrics
    # collected from 2 validation clients (so each leaf is a list of size 2).
    valid_metrics_dict = collections.OrderedDict()
    valid_metrics_dict[baseline_metrics_name] = collections.OrderedDict([
        (accuracy_name, [0.1, 0.1]), (num_examples_name, [1, 3])
    ])
    valid_metrics_dict[finetuning_fn_name] = collections.OrderedDict()
    valid_metrics_dict[finetuning_fn_name]['epoch_1'] = collections.OrderedDict(
        [(accuracy_name, [0.0, 0.3]), (num_examples_name, [1, 3])])
    valid_metrics_dict[finetuning_fn_name][num_finetune_examples_name] = [1, 2]
    # Constructs a fake dictionary representing the finetuning eval metrics
    # collected from 2 test clients (so each leaf is a list of size 2).
    test_metrics_dict = collections.OrderedDict()
    test_metrics_dict[baseline_metrics_name] = collections.OrderedDict([
        (accuracy_name, [0.2, 0.2]), (num_examples_name, [2, 3])
    ])
    test_metrics_dict[finetuning_fn_name] = collections.OrderedDict()
    test_metrics_dict[finetuning_fn_name]['epoch_1'] = collections.OrderedDict([
        (accuracy_name, [0.1, 0.2]), (num_examples_name, [2, 3])
    ])
    test_metrics_dict[finetuning_fn_name][num_finetune_examples_name] = [2, 2]
    expected_processed_metrics = collections.OrderedDict()
    expected_processed_metrics[baseline_metrics_name] = collections.OrderedDict(
        valid_accuracy_mean=0.1,  # mean([0.1, 0.1])
        test_accuracy_mean=0.2,  # mean([0.2, 0.2])
        test_num_eval_examples_mean=2.5,  # mean([2, 3])
        test_num_finetune_examples_mean=2)  # mean([2, 2])
    expected_processed_metrics[finetuning_fn_name] = collections.OrderedDict(
        # The baseline average valid accuracy is 0.1. At Epoch 1, the mean
        # valid accuracy is mean([0.0, 0.3]) = 0.15, so the best epoch is 1.
        best_finetuning_epoch=1,
        valid_accuracy_at_best_epoch_mean=0.15,  # mean([0.0, 0.3])
        test_accuracy_at_best_epoch_mean=0.15,  # mean([0.1, 0.2])
        # The first test client's accuracy at Epoch 1 (0.1) is lower than
        # the baseline accuracy (0.2). The second test client's accuracy at
        # Epoch 1 (0.2) is equal to the baseline accuracy (0.2). Hence,
        # the fraction of clients hurt at Epoch 1 is 1/2 = 0.5.
        fraction_clients_hurt_at_best_epoch=0.5)
    expected_processed_metrics[finetuning_utils._RAW_METRICS_BEFORE_PROCESS] = (
        collections.OrderedDict(
            valid=valid_metrics_dict, test=test_metrics_dict))
    processed_metrics = finetuning_utils.postprocess_finetuning_metrics(
        valid_metrics_dict, test_metrics_dict, accuracy_name,
        finetuning_fn_name)
    tf.nest.map_structure(self.assertAllClose, processed_metrics,
                          expected_processed_metrics)


if __name__ == '__main__':
  tf.test.main()
