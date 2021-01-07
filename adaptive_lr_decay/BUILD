load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "adaptive_fed_avg",
    srcs = ["adaptive_fed_avg.py"],
    srcs_version = "PY3",
    deps = [":callbacks"],
)

py_test(
    name = "adaptive_fed_avg_test",
    size = "large",
    srcs = ["adaptive_fed_avg_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "manual",
        "nokokoro",
    ],
    deps = [
        ":adaptive_fed_avg",
        ":callbacks",
    ],
)

py_library(
    name = "callbacks",
    srcs = ["callbacks.py"],
    srcs_version = "PY3",
)

py_test(
    name = "callbacks_test",
    size = "small",
    srcs = ["callbacks_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":callbacks"],
)

py_binary(
    name = "federated_trainer",
    srcs = ["federated_trainer.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":adaptive_fed_avg",
        ":callbacks",
        "//optimization/cifar100:federated_cifar100",
        "//optimization/emnist:federated_emnist",
        "//optimization/emnist_ae:federated_emnist_ae",
        "//optimization/shakespeare:federated_shakespeare",
        "//optimization/shared:optimizer_utils",
        "//optimization/shared:training_specs",
        "//optimization/stackoverflow:federated_stackoverflow",
        "//optimization/stackoverflow_lr:federated_stackoverflow_lr",
        "//utils:training_loop",
        "//utils:utils_impl",
    ],
)
