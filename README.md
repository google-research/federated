# Federated Research

Federated Research is a collection of research projects related to
[Federated Learning](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html)
and
[Federated Analytics](https://ai.googleblog.com/2020/05/federated-analytics-collaborative-data.html).
Federated learning is an approach to machine learning where a shared global
model is trained across many participating clients that keep their training data
locally. Federated analytics is the practice of applying data science methods to
the analysis of raw data that is stored locally on users’ devices.

Many of the projects contained in this repository use
[TensorFlow Federated (TFF)](https://www.tensorflow.org/federated), an
open-source framework for machine learning and other computations on
decentralized data. For an overview and introduction to TFF, please see the
[list of tutorials](https://www.tensorflow.org/federated/tutorials/federated_learning_for_image_classification).
For information on using TFF for research, see
[TFF for research](https://www.tensorflow.org/federated/tff_for_research).

## Recommended Usage

The main purpose of this repository is for reproducing experimental results in
related papers. None of the projects (or subfolders) here is intended to be a
resusable framework or package.

*   The recommended usage for this repository is to `git clone` and follow the
    instruction in each indedpendent project to run the code, usually with
    `bazel`.

There is a special module `utils/` that is widely used as a dependency for
projects in this repository. Some of the functions in `utils/` are in the
process of upstreaming to the
[TFF package](https://github.com/google-research/federated). However, `utils/`
is not promised to be a stable API and the code may change in any time.

*   The recommended usage for `utils/` is to fork the necessary piece of code
    for your own research projects.
*   If you find `utils/` and maybe other projects helpful as a module that your
    projects want to depend on (and you accept the risk of depending on
    potentially unstable and unsupported code), you can use `git submodule` and
    add the module to your python path. See
    [this example](https://github.com/michaelreneer/experiment).

## Contributing

This repository contains Google-affiliated research projects related to
federated learning and analytics. If you are working with Google collaborators
and would like to feature your research project here, please review the
[contribution guidelines](CONTRIBUTING.md) for coding style, best practices,
etc.

### Pull Requests

We currently do not accept pull requests for this repository. If you have
feature requests or encounter a bug, please file an issue to the project owners.

## Issues

Please use [GitHub issues](https://github.com/google-research/federated/issues)
to communicate with project owners for requests and bugs. Add `[project/folder
name]` in the issue title so that we can easily find the best person to respond.

## Questions

If you have questions related to TensorFlow Federated, please direct your
questions to [Stack Overflow](https://stackoverflow.com) using the
[tensorflow-federated](https://stackoverflow.com/questions/tagged/tensorflow-federated)
tag.

If you would like more information on federated learning, please see the
following
[introduction to federated learning](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html).
For a more in-depth discussion of recent progress in federated learning and open
problems, see
[Advances and Open Problems in Federated Learning](https://arxiv.org/abs/1912.04977).
