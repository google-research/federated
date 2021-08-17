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

import os
import tempfile

from absl import flags
from absl.testing import absltest
import tensorflow as tf

from dual_encoder.movielens import model_launcher_centralized as launcher

FLAGS = flags.FLAGS


def _int64_feature(value_list):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value_list))


EXAMPLE_1 = tf.train.Example(
    features=tf.train.Features(
        feature={
            'context': _int64_feature([11, 4, 7, 3, 0]),
            'label': _int64_feature([12])
        })).SerializeToString()


def setUpModule():
  tmp_dir = tempfile.mkdtemp()
  test_tfrecord_file = os.path.join(tmp_dir, 'test.tfrecord')

  with tf.io.TFRecordWriter(
      test_tfrecord_file, options=tf.io.TFRecordOptions()) as writer:
    writer.write(EXAMPLE_1)

  # Parameters.
  FLAGS.training_data_filepattern = test_tfrecord_file
  FLAGS.testing_data_filepattern = test_tfrecord_file
  FLAGS.item_vocab_size = 16
  FLAGS.item_embedding_dim = 8
  FLAGS.context_hidden_dims = [8, 8]
  FLAGS.context_hidden_activations = ['relu', None]
  FLAGS.steps_per_epoch = 2
  FLAGS.num_epochs = 1
  FLAGS.num_eval_steps = 1


class ModelLauncherTest(absltest.TestCase):

  def test_model_train_and_eval(self):
    """Verifies that the model can be executed for training and evaluation."""
    train_input_fn = launcher.get_input_fn(FLAGS.training_data_filepattern)
    eval_input_fn = launcher.get_input_fn(FLAGS.testing_data_filepattern)

    model = launcher.create_keras_model()

    history = launcher.train_and_eval(
        model=model,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        steps_per_epoch=FLAGS.steps_per_epoch,
        epochs=FLAGS.num_epochs,
        eval_steps=FLAGS.num_eval_steps,
        callbacks=None)

    # Ensure the model has a valid loss after one epoch (not NaN).
    self.assertIn('loss', history.history)
    losses = history.history['loss']
    self.assertLen(losses, FLAGS.num_epochs)
    self.assertFalse(tf.math.is_nan(losses[0]))


if __name__ == '__main__':
  absltest.main()
