load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "centralized_emnist_ae",
    srcs = ["centralized_emnist_ae.py"],
    srcs_version = "PY3",
    deps = [
        "//utils:centralized_training_loop",
        "//utils/datasets:emnist_dataset",
        "//utils/models:emnist_ae_models",
    ],
)

py_library(
    name = "federated_emnist_ae",
    srcs = ["federated_emnist_ae.py"],
    srcs_version = "PY3",
    deps = [
        "//optimization/shared:training_specs",
        "//utils/datasets:emnist_dataset",
        "//utils/models:emnist_ae_models",
    ],
)
