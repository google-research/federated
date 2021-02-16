# Utilities for federated learning research

This directory contains general utilities used by the other directories under
`research/`. Examples include utilities for saving checkpoints, and configuring
experiments via command-line flags. For examples of federated learning
experiments using these utilities, see the
[`research/optimization`](https://github.com/google-research/federated/blob/master/optimization)
directory.

Warning: These utilities are considered experimental, and may be removed or
changed at any time. Some of the utilties may be upstreamed to
[Tensorflow Federated (TFF)](https://github.com/tensorflow/federated) to become
stable APIs. However,
[like other code in this repository](https://github.com/google-research/federated#recommended-usage),
direct dependencies on `utils` and attempts to use it as a package are
discouraged. The recommended usage for `utils/` is to fork the necessary piece
of code for your own research projects.
