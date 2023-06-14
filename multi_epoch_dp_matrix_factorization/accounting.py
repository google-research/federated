# Copyright 2023, Google LLC.
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
"""Sample code for calculating the DP Guarantees of our mechanisms."""
import functools
import dp_accounting


def compute_amplified_privacy(
    amplified_nm: float,
    num_train: int,
    batch_size: int,
    num_epochs: int,
    delta: float = 1e-5,
) -> float:
  """Calculates the Poisson-Amplified DP-Guarantee of DP-SGD."""
  # You see we have quite a different method of accounting here for DP-SGDM.
  # Note that we didnt actually use poisson sampling
  sampled_event = dp_accounting.PoissonSampledDpEvent(
      sampling_probability=batch_size / num_train,
      event=dp_accounting.GaussianDpEvent(amplified_nm),
  )
  steps = int(num_train / batch_size * num_epochs)
  composed_event = dp_accounting.SelfComposedDpEvent(sampled_event, steps)
  accountant = dp_accounting.pld.PLDAccountant()
  accountant.compose(composed_event)
  return accountant.get_epsilon(target_delta=delta)


@functools.lru_cache(maxsize=1024)
def get_single_release_eps(noise_multiplier: float, delta: float) -> float:
  """Calculates the privacy-cost of MF-DP-FTRL approaches."""
  # All our matrices H are scaled to have a maximum column norm of one,
  # so the total contribution of one user to the output is like an
  # outer product ||h g^T||_2 <= ||g^T|| <= clip_norm for any column h of H.
  # We add Gaussian noise N(0, (noise_multiplier*clip_norm)**2)
  # to each of the released measurements, which have sensitivity,
  # and so the appropriate dp event is:
  e = dp_accounting.GaussianDpEvent(noise_multiplier)
  # This is very slow for small noise_multipliers (e.g. 0.01)
  # accountant = pld_privacy_accountant.PLDAccountant()
  # add or remove one is default initialization
  accountant = dp_accounting.pld.PLDAccountant()
  accountant.compose(e)
  return accountant.get_epsilon(target_delta=delta)
