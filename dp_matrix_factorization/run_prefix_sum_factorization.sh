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
  pip install -r "$initial_dir/requirements.txt"
# Return to project root.
popd

echo "Requirements installed; beginning factorization"
# Run simple settings for the binary.
bazel run dp_matrix_factorization:factorize_prefix_sum --strategy=identity --rtol=1e-5 --root_output_dir=/tmp/matfac --experiment_name=test_matrix_factorization
# Exit the virtualenv's Python environment.
deactivate
# Clean up tempdir.
rm -rf "${tmp_dir}"
