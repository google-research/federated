load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(default_visibility = ["//google/optimization/main:__subpackages__"])

licenses(["notice"])

py_binary(
    name = "centralized_trainer",
    srcs = ["centralized_trainer.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":centralized_trainer_lib"],
)

py_library(
    name = "centralized_trainer_lib",
    srcs = ["centralized_trainer.py"],
    srcs_version = "PY3",
    deps = [
        "//optimization/cifar100:centralized_cifar100",
        "//optimization/emnist:centralized_emnist",
        "//optimization/emnist_ae:centralized_emnist_ae",
        "//optimization/shakespeare:centralized_shakespeare",
        "//optimization/shared:optimizer_utils",
        "//optimization/stackoverflow:centralized_stackoverflow",
        "//optimization/stackoverflow_lr:centralized_stackoverflow_lr",
        "//utils:utils_impl",
    ],
)

py_test(
    name = "centralized_tasks_test",
    size = "large",
    srcs = ["centralized_tasks_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "manual",
        "nokokoro",
        "nopresubmit",
    ],
    deps = [
        "//optimization/cifar100:centralized_cifar100",
        "//optimization/emnist:centralized_emnist",
        "//optimization/emnist_ae:centralized_emnist_ae",
        "//optimization/shakespeare:centralized_shakespeare",
        "//optimization/stackoverflow:centralized_stackoverflow",
        "//optimization/stackoverflow_lr:centralized_stackoverflow_lr",
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
        "//optimization/cifar100:federated_cifar100",
        "//optimization/emnist:federated_emnist",
        "//optimization/emnist_ae:federated_emnist_ae",
        "//optimization/shakespeare:federated_shakespeare",
        "//optimization/shared:fed_avg_schedule",
        "//optimization/shared:optimizer_utils",
        "//optimization/shared:training_specs",
        "//optimization/stackoverflow:federated_stackoverflow",
        "//optimization/stackoverflow_lr:federated_stackoverflow_lr",
        "//utils:training_loop",
        "//utils:utils_impl",
    ],
)

py_test(
    name = "federated_tasks_test",
    size = "large",
    srcs = ["federated_tasks_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "manual",
        "nokokoro",
        "nopresubmit",
    ],
    deps = [
        "//optimization/cifar100:federated_cifar100",
        "//optimization/emnist:federated_emnist",
        "//optimization/emnist_ae:federated_emnist_ae",
        "//optimization/shakespeare:federated_shakespeare",
        "//optimization/shared:training_specs",
        "//optimization/stackoverflow:federated_stackoverflow",
        "//optimization/stackoverflow_lr:federated_stackoverflow_lr",
        "//utils:training_loop",
    ],
)
