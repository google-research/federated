load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "centralized_cifar100",
    srcs = ["centralized_cifar100.py"],
    srcs_version = "PY3",
    deps = [
        "//utils:centralized_training_loop",
        "//utils/datasets:cifar100_dataset",
        "//utils/models:resnet_models",
    ],
)

py_library(
    name = "federated_cifar100",
    srcs = ["federated_cifar100.py"],
    srcs_version = "PY3",
    deps = [
        "//optimization/shared:training_specs",
        "//utils/datasets:cifar100_dataset",
        "//utils/models:resnet_models",
    ],
)
