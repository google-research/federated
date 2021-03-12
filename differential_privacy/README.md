# Federated learning experiments with differential privacy.

This directory contains source code for training of six federated optimization
tasks with differential privacy including quantile-based adaptive clipping as
described in
["Differentially Private Learning with Adaptive Clipping" (2021)](https://arxiv.org/abs/1905.03871).

This library uses [TensorFlow Federated](https://www.tensorflow.org/federated).
For a more general look at using TensorFlow Federated for research, see
[Using TFF for Federated Learning Research](https://www.tensorflow.org/federated/tff_for_research).

Some pip packages are required by this library, and may need to be installed:

```
pip install absl-py
pip install attr
pip install numpy
pip install tensorflow
pip install tensorflow-privacy
pip install tensorflow-federated
```

## Example usage

```
bazel run run_federated -- \
  --client_optimizer=sgd \
  --server_optimizer=sgd \
  --server_sgd_momentum=0.9 \
  --clients_per_round=100 \
  --uniform_weighting=True \
  --clip=0.1 \
  --target_unclipped_quantile=0.5 \
  --adaptive_clip_learning_rate=0.2 \
  --noise_multiplier=0.1 \
  --task=stackoverflow_nwp \
  --client_learning_rate=0.3 \
  --server_learning_rate=3 \
  --total_rounds=1500 \
  --client_batch_size=16 \
  --root_output_dir=/tmp/dp \
  --experiment_name=so_nwp
```
