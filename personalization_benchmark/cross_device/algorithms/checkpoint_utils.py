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
"""Utilities used to extract models from checkpoints saved by `finetuning_trainer.py`.

To save the model weights to file, one can first assign the model weights to a
Keras model and then save the Keras model to file:
```
keras_model = model_fn()._keras_model
model_weights.assign_weights_to(keras_model)
keras_model.save(path_to_saved_model)
```
The saved path can be used as the flag `path_to_initial_model_weights` in the
`finetuning_trainer` and `hypcluster_trainer`.
"""

import asyncio
from typing import Callable, Optional
import tensorflow as tf
import tensorflow_federated as tff


def extract_model_weights_from_checkpoint(
    model_fn: Callable[[], tff.learning.Model],
    path_to_checkpoint_dir: str,
    round_num: Optional[int] = None) -> tff.learning.ModelWeights:
  """Extracts model weights from a checkpoint saved by `finetuning_trainer.py`.

  The checkpoints saved by `finetuning_trainer.py` (with name `program_state_*`)
  contain the server state structure of FedAvg training process. It saves the
  the FedAvg model weights *before* fine-tuning on each client locally.

  Args:
    model_fn: A no-arg function that returns a `tff.learning.Model`. Must be the
      same `model_fn` used to create the checkpoint.
    path_to_checkpoint_dir: A `str` path to the directory that contains the
      checkpoints. For example, if `/path/to/dir/program_state_1500/` is the
      desired checkpoint, then you should pass in `/path/to/dir/`.
    round_num: An optional integer specifying which round number to load the
      saved FedAvg checkpoint. Note that 'program_state_{round_num}' must exist
      under `path_to_checkpoint_dir`. If None, will load the latest checkpoint.

  Returns:
    A `tff.learning.ModelWeights`.
  """
  learning_process_for_metedata = tff.learning.algorithms.build_weighted_fed_avg(
      model_fn=model_fn,
      # The actual learning rate in `client_optimizer_fn`/`server_optimizer_fn`
      # does not matter here. But the optimizer (i.e., SGD and Adam) has to be
      # the same as the one used in creating the checkpoint.
      client_optimizer_fn=lambda: tf.keras.optimizers.SGD(1.0),
      server_optimizer_fn=lambda: tf.keras.optimizers.Adam(1.0),
      model_aggregator=tff.learning.robust_aggregator())
  loop = asyncio.get_event_loop()
  state_manager = tff.program.FileProgramStateManager(path_to_checkpoint_dir)
  init_state = learning_process_for_metedata.initialize()
  if round_num is None:
    state, _ = loop.run_until_complete(
        state_manager.load_latest(structure=init_state))
  else:
    state = loop.run_until_complete(
        state_manager.load(version=round_num, structure=init_state))
  return learning_process_for_metedata.get_model_weights(state)
