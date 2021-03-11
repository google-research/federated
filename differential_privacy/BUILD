load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_binary(
    name = "run_federated",
    srcs = ["run_federated.py"],
    python_version = "PY3",
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
        "//utils:training_loop",
        "//utils:utils_impl",
    ],
)
