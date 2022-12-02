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
"""A TFF ProgramStateManager that periodically runs an additional export."""
from collections.abc import Callable
import os
from typing import Any

from absl import logging
import tensorflow_federated as tff


class ExportingProgramStateManager(tff.program.FileProgramStateManager):
  """Both saves checkpoints, and runs an export function on the server state.

  When passed to the TFF training loop, this both saves checkpoints for TFF to
  resume training and exports the server state in other formats. This can be
  used, for example, to periodically export a full `SavedModel`, with the
  inference graph included, for further evaluation and use.
  """

  def __init__(self, checkpoint_root_dir: str, export_fn: Callable[[Any, str],
                                                                   None],
               rounds_per_export: int, export_root_dir: str, *args, **kwargs):
    """Initializes the ExportingProgramStateManager.

    Args:
      checkpoint_root_dir: Directory to save the `ServerState` checkpoints, for
        the TFF training process to use for resuming.
      export_fn: A callable to convert and save the server state to a directory.
        This is meant to handle converting the TFF `ServerState` to the
        exportable format, such as a Keras model, and saving it to the directory
        specified in the second argument.
      rounds_per_export: Sets the interval between exports. Note this assumes
        that `save` is called with a `version` argument that is divisible by
        this period.
      export_root_dir: Location where the exported model data is stored, in
        versioned subdirectories.
      *args: Positional args passed to FileProgramStateManager.
      **kwargs: Keyword args passed to FileProgramStateManager.
    """
    super().__init__(checkpoint_root_dir, *args, **kwargs)
    self._export_fn = export_fn
    self._rounds_per_export = rounds_per_export
    self._export_root_dir = export_root_dir

  async def save(self, program_state: Any, version: int):
    """See base class."""
    await super().save(program_state, version)

    if version % self._rounds_per_export == 0:
      export_dir = os.path.join(self._export_root_dir,
                                'inference_%06d' % version)
      logging.info('Exporting model to "%s"', export_dir)
      self._export_fn(program_state, export_dir)
    else:
      logging.info('Skipping export when saving version %d', version)
