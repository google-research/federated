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

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

from compressed_communication import builder
from compressed_communication.aggregators import entropy
from compressed_communication.aggregators import group
from compressed_communication.aggregators import histogram_weights
from compressed_communication.aggregators import quantize
from compressed_communication.aggregators import quantize_encode
from compressed_communication.aggregators import quantize_encode_client_lambda
from compressed_communication.aggregators.comparison_methods import drive
from compressed_communication.aggregators.comparison_methods import one_bit_sgd
from compressed_communication.aggregators.comparison_methods import qsgd
from compressed_communication.aggregators.comparison_methods import terngrad
from compressed_communication.aggregators.comparison_methods import three_lc
from compressed_communication.aggregators.comparison_methods import top_k


class QuantizationEncodeAggregatorTest(tff.test.TestCase):

  def test_types_expected(self):
    test_model_weights = tff.TensorType(shape=[10], dtype=tf.float32)
    factory = builder.build_quantization_encode_aggregator(
        step_size=0.5,
        concatenate=False,
        zeroing=False,
        clipping=False,
        weighted=True)
    aggregation_process = factory.create(
        test_model_weights, weight_type=tff.TensorType(tf.float32))
    reference_process = tff.aggregators.MeanFactory(
        quantize_encode.QuantizeEncodeFactory(initial_step_size=0.5)).create(
            test_model_weights, weight_type=tff.TensorType(tf.float32))
    # `initialize` function should be identical.
    self.assert_types_identical(aggregation_process.initialize.type_signature,
                                reference_process.initialize.type_signature)
    # `next` function should be identical.
    self.assert_types_identical(aggregation_process.next.type_signature,
                                reference_process.next.type_signature)


class HistogramAggregatorTest(tff.test.TestCase):

  def test_types_expected(self):
    test_model_weights = tff.TensorType(shape=[10], dtype=tf.float32)
    factory = builder.build_histogram_aggregator(
        rotation='identity',
        concatenate=False,
        zeroing=False,
        clipping=False,
        weighted=True)
    aggregation_process = factory.create(
        test_model_weights, weight_type=tff.TensorType(tf.float32))
    reference_process = tff.aggregators.MeanFactory(
        histogram_weights.HistogramWeightsFactory(
            mn=-1.0, mx=1.0, nbins=2001)).create(
                test_model_weights, weight_type=tff.TensorType(tf.float32))
    # `initialize` function should be identical.
    self.assert_types_identical(aggregation_process.initialize.type_signature,
                                reference_process.initialize.type_signature)
    # `next` function should be identical.
    self.assert_types_identical(aggregation_process.next.type_signature,
                                reference_process.next.type_signature)


class EntropyCrossEntropyAggregatorTest(tff.test.TestCase):

  def test_types_expected(self):
    test_model_weights = tff.StructType([
        ('layer_0', tff.TensorType(shape=[10], dtype=tf.float32)),
        ('layer_1', tff.TensorType(shape=[10], dtype=tf.float32)),
        ('layer_2', tff.TensorType(shape=[10], dtype=tf.float32)),
        ('layer_3', tff.TensorType(shape=[10], dtype=tf.float32)),
        ('layer_4', tff.TensorType(shape=[10], dtype=tf.float32)),
        ('layer_5', tff.TensorType(shape=[10], dtype=tf.float32)),
        ('layer_6', tff.TensorType(shape=[10], dtype=tf.float32)),
        ('layer_7', tff.TensorType(shape=[10], dtype=tf.float32))
    ])
    factory = builder.build_entropy_cross_entropy_aggregator(
        step_size=0.5,
        concatenate=True,
        zeroing=False,
        clipping=False,
        weighted=True,
        group_layers=True,
        task='emnist_character')
    aggregation_process = factory.create(
        test_model_weights, weight_type=tff.TensorType(tf.float32))
    reference_inner_factory = tff.aggregators.concat_factory(
        quantize.QuantizeFactory(
            0.5, entropy.EntropyFactory(compute_cross_entropy=True)))
    reference_process = tff.aggregators.MeanFactory(
        group.GroupFactory(
            grouped_indices=collections.OrderedDict(
                kernel=[0, 2, 4, 6], bias=[1, 3, 5, 7]),
            inner_agg_factories=collections.OrderedDict(
                kernel=reference_inner_factory,
                bias=reference_inner_factory))).create(
                    test_model_weights, weight_type=tff.TensorType(tf.float32))
    # `initialize` function should be identical.
    self.assert_types_identical(aggregation_process.initialize.type_signature,
                                reference_process.initialize.type_signature)
    # `next` function should be identical.
    self.assert_types_identical(aggregation_process.next.type_signature,
                                reference_process.next.type_signature)


class VoteQStepSizeAggregatorTest(tff.test.TestCase):

  def test_types_expected(self):
    test_model_weights = tff.TensorType(shape=[10], dtype=tf.float32)
    factory = builder.build_vote_step_size_aggregator(
        step_size=0.5,
        concatenate=False,
        zeroing=False,
        clipping=False,
        weighted=True)
    aggregation_process = factory.create(
        test_model_weights, weight_type=tff.TensorType(tf.float32))
    step_size_options = [0.5 * scale for scale in 1.0**np.linspace(-3, 3, 7)]
    reference_process = tff.aggregators.MeanFactory(
        quantize_encode_client_lambda.QuantizeEncodeClientLambdaFactory(
            lagrange_multiplier=0.5,
            step_size=0.5,
            step_size_options=step_size_options)).create(
                test_model_weights, weight_type=tff.TensorType(tf.float32))
    # `initialize` function should be identical.
    self.assert_types_identical(aggregation_process.initialize.type_signature,
                                reference_process.initialize.type_signature)
    # `next` function should be identical.
    self.assert_types_identical(aggregation_process.next.type_signature,
                                reference_process.next.type_signature)


class RotationAblationAggregatorTest(tff.test.TestCase):

  def test_types_expected(self):
    test_model_weights = tff.TensorType(shape=[10], dtype=tf.float32)
    factory = builder.build_rotation_ablation_aggregator(
        step_size=0.5,
        rounding_type='uniform',
        rotation='dft',
        concatenate=False,
        zeroing=False,
        clipping=False,
        weighted=True)
    aggregation_process = factory.create(
        test_model_weights, weight_type=tff.TensorType(tf.float32))
    reference_process = tff.aggregators.MeanFactory(
        tff.aggregators.DiscreteFourierTransformFactory(
            quantize.QuantizeFactory(0.5,
                                     entropy.EntropyFactory(include_zeros=True),
                                     'uniform'))).create(
                                         test_model_weights,
                                         weight_type=tff.TensorType(tf.float32))
    # `initialize` function should be identical.
    self.assert_types_identical(aggregation_process.initialize.type_signature,
                                reference_process.initialize.type_signature)
    # `next` function should be identical.
    self.assert_types_identical(aggregation_process.next.type_signature,
                                reference_process.next.type_signature)


class NoCompressionAggregatorTest(tff.test.TestCase):

  def test_types_expected(self):
    test_model_weights = tff.TensorType(shape=[10], dtype=tf.float32)
    factory = builder.build_no_compression_aggregator(
        rotation='identity',
        concatenate=False,
        zeroing=False,
        clipping=False,
        weighted=True)
    aggregation_process = factory.create(
        test_model_weights, weight_type=tff.TensorType(tf.float32))
    reference_process = tff.aggregators.MeanFactory(
        tff.aggregators.SumFactory()).create(
            test_model_weights, weight_type=tff.TensorType(tf.float32))
    # `initialize` function should be identical.
    self.assert_types_identical(aggregation_process.initialize.type_signature,
                                reference_process.initialize.type_signature)
    # `next` function should be identical.
    self.assert_types_identical(aggregation_process.next.type_signature,
                                reference_process.next.type_signature)


class DRIVEAggregatorTest(tff.test.TestCase):

  def test_types_expected(self):
    test_model_weights = tff.TensorType(shape=[10], dtype=tf.float32)
    factory = builder.build_drive_aggregator(
        rotation='hadamard',
        concatenate=False,
        zeroing=False,
        clipping=False,
        weighted=True)
    aggregation_process = factory.create(
        test_model_weights, weight_type=tff.TensorType(tf.float32))
    reference_process = tff.aggregators.MeanFactory(
        tff.aggregators.HadamardTransformFactory(drive.DRIVEFactory())).create(
            test_model_weights, weight_type=tff.TensorType(tf.float32))
    # `initialize` function should be identical.
    self.assert_types_identical(aggregation_process.initialize.type_signature,
                                reference_process.initialize.type_signature)
    # `next` function should be identical.
    self.assert_types_identical(aggregation_process.next.type_signature,
                                reference_process.next.type_signature)


class OneBitSGDAggregatorTest(tff.test.TestCase):

  def test_types_expected(self):
    test_model_weights = tff.TensorType(shape=[10], dtype=tf.float32)
    factory = builder.build_one_bit_sgd_aggregator(
        rotation='identity',
        concatenate=False,
        zeroing=False,
        clipping=False,
        weighted=True)
    aggregation_process = factory.create(
        test_model_weights, weight_type=tff.TensorType(tf.float32))
    reference_process = tff.aggregators.MeanFactory(
        one_bit_sgd.OneBitSGDFactory()).create(
            test_model_weights, weight_type=tff.TensorType(tf.float32))
    # `initialize` function should be identical.
    self.assert_types_identical(aggregation_process.initialize.type_signature,
                                reference_process.initialize.type_signature)
    # `next` function should be identical.
    self.assert_types_identical(aggregation_process.next.type_signature,
                                reference_process.next.type_signature)


class QSGDAggregatorTest(tff.test.TestCase):

  def test_types_expected(self):
    test_model_weights = tff.TensorType(shape=[10], dtype=tf.float32)
    factory = builder.build_qsgd_aggregator(
        num_steps=32.,
        rotation='identity',
        concatenate=False,
        zeroing=False,
        clipping=False,
        weighted=True)
    aggregation_process = factory.create(
        test_model_weights, weight_type=tff.TensorType(tf.float32))
    reference_process = tff.aggregators.MeanFactory(
        qsgd.QSGDFactory(num_steps=32.)).create(
            test_model_weights, weight_type=tff.TensorType(tf.float32))
    # `initialize` function should be identical.
    self.assert_types_identical(aggregation_process.initialize.type_signature,
                                reference_process.initialize.type_signature)
    # `next` function should be identical.
    self.assert_types_identical(aggregation_process.next.type_signature,
                                reference_process.next.type_signature)


class TernGradAggregatorTest(tff.test.TestCase):

  def test_types_expected(self):
    test_model_weights = tff.TensorType(shape=[10], dtype=tf.float32)
    factory = builder.build_terngrad_aggregator(
        rotation='identity',
        concatenate=False,
        zeroing=False,
        clipping=False,
        weighted=True)
    aggregation_process = factory.create(
        test_model_weights, weight_type=tff.TensorType(tf.float32))
    reference_process = tff.aggregators.MeanFactory(
        terngrad.TernGradFactory()).create(
            test_model_weights, weight_type=tff.TensorType(tf.float32))
    # `initialize` function should be identical.
    self.assert_types_identical(aggregation_process.initialize.type_signature,
                                reference_process.initialize.type_signature)
    # `next` function should be identical.
    self.assert_types_identical(aggregation_process.next.type_signature,
                                reference_process.next.type_signature)


class ThreeLCAggregatorTest(tff.test.TestCase):

  def test_types_expected(self):
    test_model_weights = tff.TensorType(shape=[10], dtype=tf.float32)
    factory = builder.build_three_lc_aggregator(
        sparsity_factor=1.,
        rotation='identity',
        concatenate=False,
        zeroing=False,
        clipping=False,
        weighted=True)
    aggregation_process = factory.create(
        test_model_weights, weight_type=tff.TensorType(tf.float32))
    reference_process = tff.aggregators.MeanFactory(
        three_lc.ThreeLCFactory(sparsity_factor=1.)).create(
            test_model_weights, weight_type=tff.TensorType(tf.float32))
    # `initialize` function should be identical.
    self.assert_types_identical(aggregation_process.initialize.type_signature,
                                reference_process.initialize.type_signature)
    # `next` function should be identical.
    self.assert_types_identical(aggregation_process.next.type_signature,
                                reference_process.next.type_signature)


class TopKAggregatorTest(tff.test.TestCase):

  def test_types_expected(self):
    test_model_weights = tff.TensorType(shape=[10], dtype=tf.float32)
    factory = builder.build_top_k_aggregator(
        fraction_to_select=0.05,
        rotation='identity',
        concatenate=False,
        zeroing=False,
        clipping=False,
        weighted=True)
    aggregation_process = factory.create(
        test_model_weights, weight_type=tff.TensorType(tf.float32))
    reference_process = tff.aggregators.MeanFactory(
        top_k.TopKFactory(fraction_to_select=0.05)).create(
            test_model_weights, weight_type=tff.TensorType(tf.float32))
    # `initialize` function should be identical.
    self.assert_types_identical(aggregation_process.initialize.type_signature,
                                reference_process.initialize.type_signature)
    # `next` function should be identical.
    self.assert_types_identical(aggregation_process.next.type_signature,
                                reference_process.next.type_signature)


if __name__ == '__main__':
  tff.test.main()
