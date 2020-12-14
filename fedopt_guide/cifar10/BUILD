load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "centralized_cifar10",
    srcs = ["centralized_cifar10.py"],
    srcs_version = "PY3",
    deps = [
        "//util:centralized_training_loop",
        "//utils/datasets:cifar10_dataset",
        "//utils/models:resnet_models",
    ],
)

py_library(
    name = "federated_cifar10",
    srcs = ["federated_cifar10.py"],
    srcs_version = "PY3",
    deps = [
        "//utils:training_loop",
        "//utils:training_utils",
        "//utils/datasets:cifar10_dataset",
        "//utils/models:resnet_models",
    ],
)
