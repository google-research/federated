# Federated Dual Encoder

## Overview

This directory contains an implementation of federated dual encoder as well as
source code for running experiments with centralized training and TFF
simulation. The libraries can be used to build customized federated dual encoder
models with the flexibility to apply spreadout loss and/or spreadout embeddings
on top of softmax loss or hinge loss.

## Running Experiments

`movielens/model_launcher_centralized.py` is the binary to run the baseline
centralized model. `movielens/movielens_data_gen.py` is the binary to generate
the input data for the centralized training. `movielens/model_launcher_tff.py`
is the binary to run the TFF simulation for the federated dual encoder.

### Running centralized training of a dual encoder

*   Generate Movielens datasets and save in TFRecord.

```bash
bazel run -- \
movielens:movielens_data_gen \
 --movielens_data_dir=<dir of the raw movielens data> \
 --output_dir=<dir to store processed input data>
```

*   Start centralized training.

```bash
bazel run -- \
movielens:model_launcher_centralized \
 --training_data_filepattern=<path to the training data (including the file name)> \
 --testing_data_filepattern=<path to the testing data (including the file name)> \
 --logdir=<dir to save training and testing logs> \
 --spreadout_lambda=0.001 \
 --spreadout_context_lambda=0.001 \
 --spreadout_label_lambda=0.001 \
 --spreadout_cross_lambda=0.001 \
 --output_embeddings=True \
 --use_global_similarity=True
```

### Running TFF simulation of the federated dual encoder

Start TFF simulation.

```bash
bazel run -- \
movielens:model_launcher_tff \
 --input_data_dir=<dir of the raw movielens data> \
 --logdir=<dir to store the log files> \
 --num_clients=100 \
 --num_rounds=200 \
 --batch_size=16 \
 --num_local_epochs=1 \
 --max_examples_per_user=300 \
 --shuffle_across_users=False \
 --spreadout_lambda=0.001 \
 --spreadout_context_lambda=0.001 \
 --spreadout_label_lambda=0.001 \
 --spreadout_cross_lambda=0.001 \
 --output_embeddings=True \
 --use_global_similarity=True \
 --server_optimizer='adam' \
 --server_learning_rate=1.0 \
 --client_optimizer='sgd' \
 --client_learning_rate=0.01
```
