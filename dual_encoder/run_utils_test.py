# Copyright 2021, Google LLC.
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

from absl.testing import absltest
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from dual_encoder import run_utils


Batch = collections.namedtuple('Batch', ['x', 'y'])


def create_input_spec():
  return Batch(
      x=tf.TensorSpec(shape=[None, 784], dtype=tf.float32),
      y=tf.TensorSpec(dtype=tf.int32, shape=[None, 1]))


def tff_model_fn():
  keras_model = tff.simulation.models.mnist.create_keras_model(
      compile_model=False)
  input_spec = create_input_spec()
  return tff.learning.keras_utils.from_keras_model(
      keras_model=keras_model,
      input_spec=input_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy())


def create_client_data():
  emnist_batch = collections.OrderedDict([('label', [5]),
                                          ('pixels', np.random.rand(28, 28))])

  output_types = collections.OrderedDict([('label', tf.int32),
                                          ('pixels', tf.float32)])

  output_shapes = collections.OrderedDict([
      ('label', tf.TensorShape([1])),
      ('pixels', tf.TensorShape([28, 28])),
  ])

  dataset = tf.data.Dataset.from_generator(lambda: (yield emnist_batch),
                                           output_types, output_shapes)

  def client_data():
    return tff.simulation.models.mnist.keras_dataset_from_emnist(
        dataset).repeat(2).batch(2)

  return client_data


class RunUtilsTest(absltest.TestCase):

  def test_train_and_eval_multiple_rounds(self):
    trainer = tff.learning.build_federated_averaging_process(
        tff_model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1))
    evaluator = tff.learning.build_federated_evaluation(tff_model_fn)

    client_data = create_client_data()
    train_data = [client_data()]

    states = run_utils.train_and_eval(
        trainer=trainer,
        evaluator=evaluator,
        num_rounds=5,
        train_datasets=train_data,
        test_datasets=train_data,
        num_clients_per_round=1)

    self.assertEqual(states.model.trainable[0].shape, (5, 5, 1, 32))


if __name__ == '__main__':
  absltest.main()
