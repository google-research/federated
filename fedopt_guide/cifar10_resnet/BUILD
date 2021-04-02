load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_test(
    name = "federated_cifar10_test",
    timeout = "long",
    srcs = ["federated_cifar10_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "manual",
        "nokokoro",
        "nopresubmit",
        "requires-gpu-nvidia",
    ],
    deps = [":federated_cifar10"],
)

py_test(
    name = "centralized_cifar10_test",
    srcs = ["centralized_cifar10_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "manual",
        "nokokoro",
        "nopresubmit",
    ],
    deps = [":centralized_cifar10"],
)

py_library(
    name = "federated_cifar10",
    srcs = ["federated_cifar10.py"],
    srcs_version = "PY3",
    deps = [
        "//fedopt_guide:training_loop",
        "//utils/datasets:cifar10_dataset",
        "//utils/models:resnet_models",
    ],
)

py_library(
    name = "centralized_cifar10",
    srcs = ["centralized_cifar10.py"],
    srcs_version = "PY3",
    deps = [
        "//utils:centralized_training_loop",
        "//utils/datasets:cifar10_dataset",
        "//utils/models:resnet_models",
    ],
)

py_binary(
    name = "federated_trainer",
    srcs = ["federated_trainer.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":federated_trainer_lib"],
)

py_library(
    name = "federated_trainer_lib",
    srcs = ["federated_trainer.py"],
    srcs_version = "PY3",
    deps = [
        ":federated_cifar10",
        "//optimization/shared:optimizer_utils",
        "//utils:utils_impl",
    ],
)

py_binary(
    name = "centralized_trainer",
    srcs = ["centralized_trainer.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":centralized_cifar10",
        "//optimization/shared:optimizer_utils",
        "//utils:utils_impl",
    ],
)
