# Federated Reconstruction Models

This directory contains code for partially local federated learning with
reconstruction, as introduced in
["Federated Reconstruction: Partially Local Federated Learning"](https://arxiv.org/abs/2102.03448).
The library includes general-purpose utilities for partially local federated
learning as well as experimental code for the paper.

## Using this Directory

This library uses [TensorFlow Federated](https://www.tensorflow.org/federated).
For a more general look at using TensorFlow Federated for research, see
[Using TFF for Federated Learning Research](https://www.tensorflow.org/federated/tff_for_research).

Some pip packages are required by this library and can be installed by running
`pip install -r requirements.txt`. Code has been tested with Python 3.8.

We also require [Bazel](https://www.bazel.build/) in order to run the code.
Please see the guide
[here](https://docs.bazel.build/versions/master/install.html) for installation
instructions.

## Directory Structure

The base directory `reconstruction/` contains general-purpose libraries for
partially local federated learning in TensorFlow Federated. `shared/` contains
utilities for running experiments shared across tasks. `movielens/` and
`stackoverflow/` each contain task-specific utilities for loading data,
preparing models, and running training loops.

## Running Experiments

`shared/federated_trainer.py` is the main binary for running experiments–it
accepts many flags for configuring experiments, e.g. for choosing different
tasks or hyperparameter tuning. These flags are documented in that file.

A command for running a MovieLens experiment with `FedRecon` with baseline
hyperparameters:

```bash
bazel run shared:federated_trainer -- --task=movielens_mf --root_output_dir=/tmp/output --experiment_name=ml_mf --client_optimizer=sgd --client_learning_rate=0.5 \
--server_optimizer=sgd --server_learning_rate=1.0 --reconstruction_optimizer=sgd --reconstruction_learning_rate=0.1 --client_batch_size=5 --clients_per_round=100 \
--total_rounds=500 --rounds_per_eval=5 --rounds_per_checkpoint=25 --recon_epochs_max=5 --recon_steps_max=50 --post_recon_steps_max=50 --split_dataset=True
```

A command for running a Stack Overflow experiment with `FedRecon` with baseline
hyperparameters:

```bash
bazel run shared:federated_trainer -- --task=stackoverflow_nwp --root_output_dir=/tmp/output --experiment_name=so_nwp --client_optimizer=sgd --client_learning_rate=0.3 \
--server_optimizer=yogi --server_learning_rate=0.1 --server_yogi_initial_accumulator_value=0.0 --server_yogi_beta2=0.99 --server_yogi_epsilon=0.001 --reconstruction_optimizer=sgd \
--reconstruction_learning_rate=0.1 --client_batch_size=16 --clients_per_round=200 --total_rounds=2500 --rounds_per_eval=10 --rounds_per_checkpoint=50 --so_nwp_max_elements_per_user=1024 \
--so_nwp_num_oov_buckets=500 --recon_epochs_max=5 --recon_steps_max=100 --post_recon_steps_max=100 --split_dataset=True
```

Be sure to use a different experiment name for each experiment, otherwise
checkpoints from previous runs will be loaded.

## Extending the Library

### Adding New Tasks and Datasets

`movielens/federated_movielens.py` and
`stackoverflow/federated_stackoverflow.py` provide examples on setting up tasks
in this codebase. Each has a `run_federated` function which is called by
`shared/federated_trainer.py`. This function loads the model and data and builds
training and evaluation processes to run over a training loop. Additional tasks
and datasets can use the same pattern, modifying `shared/federated_trainer.py`
to call `run_federated` for any new tasks with task-specific flags.

### Evaluating New Algorithms

Each `run_federated` function for each task is passed builders for
reconstruction training and evaluation processes, but e.g. other
`tff.templates.IterativeProcess`s can be used for training instead (see
`training_process.py` for the federated reconstruction training process
implementation, which can be extended). See `shared/federated_trainer.py` for an
example of using a different evaluation computation for `Finetuning`. The
existing flags in `shared/federated_trainer.py` can be set to instantiate a
wider variety of algorithms even without code changes; for example, we can
perform `FederatedAveraging` by setting `jointly_train_variables=True,
split_dataset=False, global_variables_only=True,
reconstruction_learning_rate=0.0`.

## Citations

If building upon this library or paper, use the following bibtex:

```latex
@misc{singhal2021federated,
      title={Federated Reconstruction: Partially Local Federated Learning},
      author={Karan Singhal and Hakim Sidahmed and Zachary Garrett and Shanshan Wu and Keith Rush and Sushant Prakash},
      year={2021},
      eprint={2102.03448},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
