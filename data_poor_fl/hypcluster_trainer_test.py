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
import tempfile

from absl.testing import flagsaver
from absl.testing import parameterized
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff

from data_poor_fl import hypcluster_trainer


def create_dataset():
  # Create data satisfying y = x + 1
  x = [[1.0], [2.0], [3.0]]
  y = [[2.0], [3.0], [4.0]]
  return tf.data.Dataset.from_tensor_slices((x, y)).batch(1)


def get_input_spec():
  return create_dataset().element_spec


def model_fn(initializer='zeros'):
  keras_model = tf.keras.Sequential([
      tf.keras.layers.Dense(
          1,
          kernel_initializer=initializer,
          bias_initializer=initializer,
          input_shape=(1,))
  ])
  return tff.learning.from_keras_model(
      keras_model=keras_model,
      input_spec=get_input_spec(),
      loss=tf.keras.losses.MeanSquaredError())


class GlobalTrainerTest(tf.test.TestCase, parameterized.TestCase):

  def test_pseudo_client_id_generation(self):
    data = dict(client_id=['A', 'B'], num_examples=[3, 5])
    df = pd.DataFrame(data=data)
    actual_pseudo_client_ids = hypcluster_trainer._get_pseudo_client_ids(
        examples_per_pseudo_clients=2, base_client_examples_df=df)
    expected_pseudo_client_ids = ['A-0', 'A-1', 'B-0', 'B-1', 'B-2']
    self.assertEqual(actual_pseudo_client_ids, expected_pseudo_client_ids)

  def test_split_pseudo_client_ids(self):
    raw_client_ids = [str(i) for i in range(3400)]
    pseudo_client_ids = [str(i) + '-0' for i in range(3400)
                        ] + [str(i) + '-1' for i in range(3400)]
    pseudo_train_client_ids, pseudo_eval_client_ids = (
        hypcluster_trainer._split_pseudo_client_ids(raw_client_ids,
                                                    pseudo_client_ids))
    self.assertLen(pseudo_eval_client_ids,
                   hypcluster_trainer._NUM_RAW_EVAL_CLIENTS * 2)
    self.assertLen(pseudo_client_ids,
                   len(pseudo_train_client_ids) + len(pseudo_eval_client_ids))

  @parameterized.named_parameters(
      ('clusters1', 1),
      ('clusters2', 2),
      ('clusters3', 3),
      ('clusters5', 5),
  )
  def test_hypcluster_eval_returns_same_num_examples(self, num_clusters):
    client_dataset = collections.OrderedDict(
        selection_data=create_dataset(), test_data=create_dataset())
    batch_type = tff.to_type(model_fn().input_spec)
    client_data_type = tff.to_type(
        collections.OrderedDict(
            selection_data=tff.SequenceType(batch_type),
            test_data=tff.SequenceType(batch_type)))
    eval_comp = hypcluster_trainer._build_hypcluster_eval(
        model_fn=model_fn,
        num_clusters=num_clusters,
        client_data_type=client_data_type)

    def weight_tensors_from_model(model: tff.learning.Model):
      return tf.nest.map_structure(lambda var: var.numpy(),
                                   tff.learning.ModelWeights.from_model(model))

    model_weights_list = [
        weight_tensors_from_model(model_fn()) for _ in range(num_clusters)
    ]
    eval_metrics = eval_comp(model_weights_list,
                             [client_dataset, client_dataset])
    # There are two clients. Each client has three examples.
    self.assertEqual(eval_metrics['best']['num_examples'], 6)
    for i in range(num_clusters):
      self.assertEqual(eval_metrics[f'model_{i}']['num_examples'], 6)

  @flagsaver.flagsaver(
      root_output_dir=tempfile.mkdtemp(),
      experiment_name='test_experiment',
      clients_per_train_round=1,
      num_clusters=2,
      total_rounds=2,
      client_optimizer='sgd',
      client_learning_rate=0.01,
      server_optimizer='adam',
      server_learning_rate=1.0,
      clients_per_evaluation=1,
      use_synthetic_data=True)
  def test_executes(self):
    hypcluster_trainer.main([])


if __name__ == '__main__':
  tf.test.main()
