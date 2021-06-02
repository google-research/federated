load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_binary(
    name = "run_federated",
    srcs = ["run_federated.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//optimization/tasks:cifar100",
        "//optimization/tasks:emnist",
        "//optimization/tasks:emnist_ae",
        "//optimization/tasks:shakespeare",
        "//optimization/tasks:stackoverflow_nwp",
        "//optimization/tasks:stackoverflow_tp",
        "//optimization/tasks:training_specs",
        "//utils:utils_impl",
        "//utils/optimizers:optimizer_utils",
    ],
)

py_library(
    name = "secret_sharer",
    srcs = ["secret_sharer.py"],
    srcs_version = "PY3",
)

py_test(
    name = "secret_sharer_test",
    srcs = ["secret_sharer_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":secret_sharer"],
)
