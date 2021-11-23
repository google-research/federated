# FL Generalization

This directory contains source code for reproducing results in paper
"[What Do We Mean by Generalization in Federated Learning](https://arxiv.org/abs/2110.14216)".
This includes evaluating generalization performance in federated learning with
different optimizers on various models and tasks.

```
@article{yuan2021we,
  title={What Do We Mean by Generalization in Federated Learning?},
  author={Yuan, Honglin and Morningstar, Warren and Ning, Lin and Singhal, Karan},
  journal={NeurIPS 2021 Workshop on New Frontiers in Federated Learning (NeurIPS NFFL 2021)},
  year={2021}
}
```

## Using this directory

This library uses [TensorFlow Federated](https://www.tensorflow.org/federated).
For a more general look at using TensorFlow Federated for research, see
[Using TFF for Federated Learning Research](https://www.tensorflow.org/federated/tff_for_research).

Some pip packages are required by this library, and may need to be installed:

```
!pip install absl-py attrs clu numpy pandas sklearn
!pip install tensorflow_datasets tensorflow-probability
!pip uninstall tensorflow keras
!pip install tf-nightly tfa-nightly tensorflow-federated-nightly
```

We require [Bazel](https://www.bazel.build/) in order to run the code. Please
see the guide [here](https://docs.bazel.build/versions/master/install.html) for
installation instructions.

## Directory structure

This directory is broken up into three directories.

-   Directory `task` contains the task specification for four tasks:

    -   CIFAR10/100 image classification
    -   EMNIST-like (or generally MNIST-like task)
    -   Shakespeare next character prediction
    -   Stackoverflow next work prediction.

    -   Directory `synthesization` contains libraries and binary for creating
        federated dataset from centralized dataset with various algorithms
        (label-based dirichlet, coarse-label-based dirichlet, GMM on pretrained
        embedding).

    -   Directory `utils` contains various utility libraries to facilitate
        ClientData manipulation (in particular intra-client, inter-client
        split), serialization, federated and centralized training, evaluation
        (including percentiles), and synthesization.

The binaries running these tasks can be found at `trainer_centralized.py` and
`trainer_federated.py`. These binaries will, according to `absl` flags, run any
of the four task-specific centralized or federated training libraries.
