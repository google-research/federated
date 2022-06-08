#!/bin/bash
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
# Assumed to be run from project root.
if [[ ! -f "dp_matrix_factorization/requirements.txt" ]]; then
  echo "run.sh assumed to be run from federated_research project root, including e.g. a file dp_matrix_factorization/requirements.txt. No requirements found."
  exit 1
fi

# Initialize Python env.
initial_dir="$(pwd)"
# Create a temp dir in which to place the virtualenv
tmp_dir="$(mktemp -d -t dp-matfac-env-XXXX)"
# Enter tempdir; we will pop before leaving.
pushd "${tmp_dir}"
  virtualenv -p python3 pip_env
  source pip_env/bin/activate
  pip install --upgrade pip
  pip install -r "$initial_dir/dp_matrix_factorization/requirements.txt"
popd

echo "Requirements installed; beginning factorization"
matfac_name="prefix_sum_factorization"
root_dir="/tmp/matfac"
# Run simple settings for the binary. We coupled the directory we write these
# factorizations to to the directory we read from in the aggregator builder,
# given the flag configuration below.
bazel run dp_matrix_factorization:factorize_prefix_sum -- --strategy=identity --log_2_observations=11 --root_output_dir="$root_dir" --experiment_name="$matfac_name"
# Train for 2048 rounds. With these settings, on a 12-core machine with no
# accelerators, rounds  take ~ 60 sec on average, so training would complete in
# ~34 hrs. Metrics should log every 20 rounds by default. The binary should be
# robust to failures and restarts, so training can be e.g. stopped and resumed.
echo "Factorization computed; beginning training."
bazel run dp_matrix_factorization/dp_ftrl:run_stackoverflow -- --aggregator_method=opt_prefix_sum_matrix --total_rounds=2048 --total_epochs=1 --clients_per_round=100 --root_output_dir="$root_dir" --experiment_name=stackoverflow_test_run --matrix_root_path="$root_dir/$matfac_name"

# Exit the virtualenv's Python environment.
deactivate
# Clean up tempdir.
rm -rf "${tmp_dir}"
