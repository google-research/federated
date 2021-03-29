load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "centralized_shakespeare",
    srcs = ["centralized_shakespeare.py"],
    srcs_version = "PY3",
    deps = [
        "//optimization/shared:keras_metrics",
        "//utils:centralized_training_loop",
        "//utils/datasets:shakespeare_dataset",
        "//utils/models:shakespeare_models",
    ],
)

py_library(
    name = "federated_shakespeare",
    srcs = ["federated_shakespeare.py"],
    srcs_version = "PY3",
    deps = [
        "//optimization/shared:keras_metrics",
        "//optimization/shared:training_specs",
        "//utils/datasets:shakespeare_dataset",
        "//utils/models:shakespeare_models",
    ],
)
