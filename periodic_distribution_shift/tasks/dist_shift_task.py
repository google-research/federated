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
"""Utilities for distribution shift tasks."""

from typing import Callable

import attr
import tensorflow as tf
import tensorflow_federated as tff

from periodic_distribution_shift.tasks import dist_shift_task_data


@attr.s(frozen=True, init=True)
class DistShiftTask(object):
  """Specification for a baseline learning simulation.

  Attributes:
    datasets: A `tff.simulation.baselines.BaselineTaskDatasets` object
      specifying dataset-related aspects of the task, including training data
      and preprocessing functions.
    model_fn: A no-arg callable returning a `tff.learning.Model` used for the
      task. Note that `model_fn().input_spec` must match
      `datasets.element_type_structure`.
  """
  datasets: dist_shift_task_data.DistShiftDatasets = attr.ib(
      validator=attr.validators.instance_of(
          dist_shift_task_data.DistShiftDatasets))
  model_fn: Callable[[], tff.learning.Model] = attr.ib(
      validator=attr.validators.is_callable())

  def __attrs_post_init__(self):
    # Wrap model construction in a graph to avoid polluting the global context
    # with variables created for this model.
    with tf.Graph().as_default():
      tff_model = self.model_fn()
    if not isinstance(tff_model, tff.learning.Model):
      raise TypeError('Expected model_fn to output a tff.learning.Model, '
                      'found {} instead'.format(type(tff_model)))

    dataset_element_spec = self.datasets.element_type_structure
    model_input_spec = tff_model.input_spec

    if dataset_element_spec != model_input_spec:
      raise ValueError(
          'Dataset element spec and model input spec do not match.'
          'Found dataset element spec {}, but model input spec {}'.format(
              dataset_element_spec, model_input_spec))
