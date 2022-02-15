# Distributed Differential Privacy

## Overview

This directory contains the code for the papers "The Distributed Discrete
Gaussian Mechanism for Federated Learning with Secure Aggregation" ICML 2021
[[PDF](https://arxiv.org/pdf/2102.06387)][[arXiv](https://arxiv.org/abs/2102.06387)]
and "The Skellam Mechanism for Differentially Private Federated Learning"
NeurIPS 2021
[[PDF](https://arxiv.org/pdf/2110.04995.pdf)][[arXiv](https://arxiv.org/abs/2110.04995)]
.

## Dependencies

-   Bazel (build tool for running the scripts)
-   Python dependencies: see `requirements.txt`.
-   Note: You may need the
    [nightly build](https://pypi.org/project/tensorflow-federated-nightly/) of
    Tensorflow Federated (TFF): `pip install tensorflow-federated-nightly`.

## Directory Structure

Run scripts:

-   `dme_run.py`: Runs the distributed mean estimation experiment.
-   `fl_run.py`: Runs the FL experiments on different tasks
    -   [Federated EMNIST](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/emnist/load_data)
    -   [Stack Overflow](https://www.tensorflow.org/federated/api_docs/python/tff/simulation/datasets/stackoverflow/load_data)

Shared modules:

-   `accounting_utils.py`: Util functions for computing relevant privacy
    parameters.
-   `ddpquery_utils.py`: Util functions for constructing DPQuery objects (from
    [TF Privacy](https://github.com/tensorflow/privacy)) with quantization.
-   `compression_utils.py`: Util functions for encoding client data (e.g.
    rotation, scaling, randomized rounding).
-   `compression_query.py`: Packs the compression util functions in a `DPQuery`
    object that can wrap the operations of another `DPQuery` object.
-   `discrete_gaussian_utils.py`: Implements sampling for the discrete Gaussian
    distribution.
-   `distributed_discrete_gaussian_query.py`: Implements the distributed
    discrete Gaussian mechanism as a `DPQuery` object from TensorFlow Privacy
    for simulations.
-   `distributed_skellam_query.py`: Implements the distributed Skellam mechanism
    as a `DPQuery` object from TensorFlow Privacy for simulations.
-   `modular_clipping_factory.py`: implements the TensorFlow Federated
    `tff.aggregators.AggregationProcessFactory` interface for applying modular
    clipping before and after aggregation. Used by the FL experiments.

## Running Experiments

There are two main components:

-   Distributed Mean Estimation (`dme_run.py`)
-   Federated Learning (`fl_run.py`)

### Distributed Mean Estimation

First, set the experiment parameters (e.g. bits, `n`, and `d`) in the Python
script `dme_run.py` for both the CLI flags and the constants in the `main()`
function.

#### Sequential Execution

Run the script with:

```
bazel run :dme_run
```

It also provides some basic plotting with the `--show_plot` flag. Note that by
default the above script runs everything sequentially.

#### Parallel Execution

For settings with larger `n` and `d`, different random dataset initializations
can be parallelized over different processes with a for-loop in `bash`:

```
for i in `seq 10`; do
  bazel run :dme_run -- --repeat=1 --run_id=$i --tag=my_test_run & done; wait
```

Then, combine and plot the results of the repeated parallel runs at
`/tmp/ddg_dme_outputs/my_test_run/` with:

```
python3 dme_merge_repeats.py /tmp/ddg_dme_outputs/my_test_run/
```

### Federated Learning

Run the `fl_run.py` script to start training:

```
bazel run :fl_run -- <flags>
```

The task flags are defined in the format of:

```
--task=<task_name> --<task_name>_<task_args1>=... --<task_name>_<task_args2>=...
```

where `task_name` is one of the options from `utils/task_utils.py`, and
`task_args` are defined according to the task constructors at
`tff.simulation.baselines.*`.

The optimizer flags can be set as:

```
--server_optimizer=sgd --server_learning_rate=<slr> --server_sgd_momentum=0.9 \
--client_optimizer=sgd --client_learning_rate=<clr>
```

Example command to train on Federated EMNIST:

```
bazel run :fl_run -- \
    --task=emnist_character \
    --server_optimizer=sgd \
    --server_learning_rate=1 \
    --server_sgd_momentum=0.9 \
    --client_optimizer=sgd \
    --client_learning_rate=0.03 \
    --client_batch_size=20 \
    --experiment_name=my_emnist_test \
    --epsilon=10 \
    --l2_norm_clip=0.03 \
    --dp_mechanism=ddgauss \
    --logtostderr
```

For Stack Overflow tasks, the following flags should also be set:

-   `--num_validation_examples=10000`
-   `--max_elements_per_client=256` (next word prediction)
-   `--max_elements_per_client=1000` (tag prediction)

Please refer to the paper for hyperparameter settings and set the flags
accordingly. Note that the training scripts by default do not configure hardware
accelerators.

## Result Highlights

Distributed Discrete Gaussian achieves comparable utility to central
differential privacy with the Gaussian mechanism under the same privacy budget.

**Left:** Stack Overflow next-word prediction. **Right:** Large-cohort (1000
clients) and low-noise (utility-preserving) settings.

<img src="images/ddg-sonwp-summary-k4.png" height="200px"><img src="images/ddg-sonwp-n1000-rounds.png" height="200px">

## Troubleshooting

-   If errors occur with calls to the TFF util libraries (mainly used by the FL
    training scripts), try using a TFF nightly build.

## Citations

```
@inproceedings{kairouz2021distributed,
  title={The distributed discrete gaussian mechanism for federated learning with secure aggregation},
  author={Kairouz, Peter and Liu, Ziyu and Steinke, Thomas},
  booktitle={International Conference on Machine Learning},
  pages={5201--5212},
  year={2021},
  organization={PMLR}
}

@article{agarwal2021skellam,
  title={The skellam mechanism for differentially private federated learning},
  author={Agarwal, Naman and Kairouz, Peter and Liu, Ziyu},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}

```
