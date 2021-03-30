load("@rules_python//python:defs.bzl", "py_binary", "py_library")

package(default_visibility = ["//google/posterior_averaging/main:__subpackages__"])

licenses(["notice"])

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
        "//optimization/cifar100:federated_cifar100",
        "//optimization/emnist:federated_emnist",
        "//optimization/emnist_ae:federated_emnist_ae",
        "//optimization/shakespeare:federated_shakespeare",
        "//optimization/shared:optimizer_utils",
        "//optimization/shared:training_specs",
        "//optimization/stackoverflow:federated_stackoverflow",
        "//optimization/stackoverflow_lr:federated_stackoverflow_lr",
        "//posterior_averaging/shared:fed_pa_schedule",
        "//utils:training_loop",
        "//utils:utils_impl",
    ],
)
