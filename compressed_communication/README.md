# Federated learning with compressed client communication.

## Summary

This directory provides the experiment code for "Optimizing the Communication-
Accuracy Trade-off in Federated Learning with Rate-Distortion Theory"
(https://arxiv.org/abs/2201.02664).

## Requirements

This library uses [TensorFlow Federated](https://www.tensorflow.org/federated)
(TFF). Some pip packages (including TFF) are required by this library, and may
need to be installed. See the [requirements](requirements.txt) file for details.
We also require [Bazel](https://www.bazel.build/) in order to run the code.
Please see the guide [here](https://bazel.build/install) for installation
instructions.

## Library overview

This library has two main libraries, `aggregators` and `broadcasters`. The
aggregators are used to compress client-to-server communication, while the
broadcasters are used to govern server-to-client communication. These
aggregators are generally focused on specific aspects of compression (eg.
implementing run-length encoding) and can all be composed using TFF's
aggregation composition.

In [`builder.py`](builder.py), we provide a suite of higher-level TFF
aggregators that perform a number of different compression operations. These
including our own compression aggregators, as well as others for benchmarking
purposes (including [DRIVE](https://arxiv.org/abs/2105.08339),
[QSGD](https://arxiv.org/abs/1610.02132), and
[TernGrad](https://arxiv.org/abs/1705.07878)). For more details on these
aggregation schemes, see our accompanying
[paper](https://arxiv.org/abs/2201.02664).

## Running experiments

To use the compression aggregators above in federated learning simulations, we
have provided a flag based library, [`trainer`](trainer.py). This can be run via
Bazel, and configured with flags that determine which compression method to use,
which optimizer to use, which FL task to run, and other details. For example, we
could run

```
bazel run :trainer -- --task=emnist_character --total_rounds=100 --aggregator=quantize_entropy_code --step_size=0.5 --rounding_type=stochastic
--client_optimizer=sgd --client_learning_rate=0.1 --train_batch_size=20
--server_optimizer=sgd --server_learning_rate=1.0 --clients_per_train_round=10 --experiment_name=emnist_compression
```

This command will run the EMNIST character prediction task for 100 rounds, using
our quantization method with a quantization step-size of 0.5. The command above
also configures client and server optimizers and learning rates, as well as
aspects of the federated training (eg. using 10 clients per round). For a full
list of flags, please see the code.
