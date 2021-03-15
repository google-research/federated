load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "fed_avg_local_adaptivity",
    srcs = ["fed_avg_local_adaptivity.py"],
    srcs_version = "PY3",
    deps = ["//utils:tensor_utils"],
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
        ":fed_avg_local_adaptivity",
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

py_test(
    name = "fed_avg_local_adaptivity_test",
    srcs = ["fed_avg_local_adaptivity_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":fed_avg_local_adaptivity",
        "//optimization/shared/keras_optimizers:yogi",
    ],
)
