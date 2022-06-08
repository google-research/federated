# DP Matrix Factorization for Streaming Linear Operators.

This directory contains the code for improving expected reconstruction error for
differentially private linear operators via the matrix mechanism.

The code here implements both gradient and fixed-point based algorithms to
compute optimal factorizations, and integrates these factorizations with
federated learning for the purpose of training machine learning models.

The code in this directory generated the data used in the paper: "Improved
Differential Privacy for SGD via Optimal Private Linear Operators on Adaptive
Streams", https://arxiv.org/abs/2202.08312.

## Usage

This code can be downloaded and verified to be correctly configured by the
following procedure:

1.  Cloning the
    [federated_research](https://github.com/google-research/federated) research
    repository.
1.  Ensuring [Bazel](https://bazel.build/) is installed.
1.  Navigating to the `federated_research` directory immediately above
    `dp_matrix_factorization`, which contains a WORKSPACE file and serves as the
    Bazel project root.
1.  Creating and activating a Python virtual environment, e.g. `virtualenv -p
    python3 dp_matfac_venv && source dp_matfac_venv/bin/activate`.
1.  Installing this project's dependencies to that virtual environment, via `pip
    install -r dp_matrix_factorization/requirments.txt`.
1.  Running all the tests associated to this project, via `bazel test
    dp_matrix_factorization/...`.

A single experiment can be run via a standalone shell script, invoking

```shell
bash dp_matrix_factorization/run_stackoverflow.sh
```

This script will construct a virtualenv and install the appropriate
dependencies, then compute a matrix factorization corresponding to the prefix
sum matrix S in the paper references above, and proceed to run 2048 rounds of
federated StackOverflow training with this factorization and some default
settings (e.g., 100 clients per round). Training on a 12-core machine with no
accelerators has been observed to take approximately 1 minute per round.
