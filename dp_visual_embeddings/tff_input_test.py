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
"""Tests for TFF input processing."""

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_federated as tff

from dp_visual_embeddings import tff_input


class PreprocessTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('size84', 84), ('size224', 224))
  def test_scale_image_pixels(self, image_size):
    batch_size = 16
    batch_image = tf.random.stateless_uniform(
        [batch_size, image_size, image_size, 3],
        seed=[0, 42],
        minval=0,
        maxval=256,
        dtype=tf.int32)
    scaled_image = tff_input._scale_image_pixels(batch_image)
    min_val = tf.reduce_min(scaled_image)
    self.assertNear(min_val, -1, 0.1)
    max_val = tf.reduce_max(scaled_image)
    self.assertNear(max_val, 1, 0.1)

  def test_preprocess(self):
    image_shapes = [(84, 21, 3), (44, 44, 1)]
    images = [
        tf.random.uniform(
            shape=image_shape, minval=0, maxval=256, dtype=tf.int32)
        for image_shape in image_shapes
    ]
    identities = [
        tf.constant([2], dtype=tf.int64),
        tf.constant([5], dtype=tf.int64)
    ]
    image_key = 'image/decoded'
    identity_key = 'class'

    def iter_input_examples():
      for image, identity in zip(images, identities):
        example = {
            image_key: image,
            identity_key: identity,
        }
        yield example

    input_signature = {
        image_key:
            tf.TensorSpec(shape=(None, None, None), dtype=tf.dtypes.uint8),
        identity_key:
            tf.TensorSpec(shape=(1), dtype=tf.dtypes.int64),
    }
    input_dataset = tf.data.Dataset.from_generator(
        iter_input_examples, output_signature=input_signature)

    target_shape = (64, 64, 3)
    preprocess_spec = tff.simulation.baselines.ClientSpec(
        num_epochs=1, batch_size=2, shuffle_buffer_size=1)
    preprocess_fn = tff_input.get_integer_label_preprocess_fn(
        preprocess_spec,
        image_shape=target_shape,
        resize_images=True,
        image_key=image_key,
        identity_key=identity_key)
    preprocessed_dataset = preprocess_fn(input_dataset)

    output_elements = list(preprocessed_dataset)
    self.assertLen(output_elements, 1)
    output_batch = output_elements[0]
    self.assertCountEqual(output_batch.keys(), ['x', 'y'])
    self.assertCountEqual(output_batch['x'].keys(), ['images'])
    self.assertSequenceEqual(output_batch['x']['images'].numpy().shape,
                             (2,) + target_shape)
    self.assertCountEqual(output_batch['y'].keys(),
                          ['identity_names', 'identity_indices'])
    self.assertCountEqual(output_batch['y']['identity_indices'].numpy(), [2, 5])

  def test_resize_eager(self):
    image_shapes = [(84, 21, 1), (44, 44, 1)]
    images = [
        tf.random.uniform(
            shape=image_shape, minval=0, maxval=256, dtype=tf.int32)
        for image_shape in image_shapes
    ]
    identities = [
        tf.constant([2], dtype=tf.int64),
        tf.constant([5], dtype=tf.int64)
    ]
    image_key = 'image/decoded'
    identity_key = 'class'

    def iter_input_examples():
      for image, identity in zip(images, identities):
        example = {
            image_key: image,
            identity_key: identity,
        }
        yield example

    input_signature = {
        image_key:
            tf.TensorSpec(shape=(None, None, None), dtype=tf.dtypes.uint8),
        identity_key:
            tf.TensorSpec(shape=(1), dtype=tf.dtypes.int64),
    }
    input_dataset = tf.data.Dataset.from_generator(
        iter_input_examples, output_signature=input_signature)
    target_shape = (64, 64, 3)
    resize_fn = tff_input._get_gray2color_resize_image_fn(
        image_key=image_key, image_shape=target_shape)
    for element in input_dataset:
      resized = resize_fn(element)
      self.assertAllEqual(resized[identity_key], resized[identity_key])
      self.assertAllEqual(tf.shape(resized[image_key]), target_shape)


if __name__ == '__main__':
  tf.test.main()
