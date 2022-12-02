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
"""Tests for model_size."""

import tensorflow as tf

from dp_visual_embeddings.models import keras_mobilenet_v2 as mobilenet
from dp_visual_embeddings.models import keras_resnet as resnet

_IMAGE_SHAPE = (224, 224, 3)
_NUM_TRAIN_IDENTITIES = 9047
_EMBED_SIZE = 128


class ModelSizeTest(tf.test.TestCase):

  def test_mobilenet_full(self):
    model = mobilenet.create_mobilenet_v2_for_backbone_training(
        input_shape=_IMAGE_SHAPE, num_identities=_NUM_TRAIN_IDENTITIES)
    model.model.summary()

  def test_mobilenet_inference(self):
    model = mobilenet.create_mobilenet_v2_for_embedding_prediction(
        input_shape=_IMAGE_SHAPE, embedding_dim_size=_EMBED_SIZE)
    model.summary()

  def test_resnet_full(self):
    model = resnet.resnet50_with_head(num_classes=_NUM_TRAIN_IDENTITIES)
    model.model.summary()

  def test_resnet_partial(self):
    model = resnet.resnet50_image2embedding(trainable_conv=False)
    model.summary()

if __name__ == '__main__':
  tf.test.main()
