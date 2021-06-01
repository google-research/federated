load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "centralized_stackoverflow",
    srcs = ["centralized_stackoverflow.py"],
    srcs_version = "PY3",
    deps = [
        "//utils:centralized_training_loop",
        "//utils:keras_metrics",
        "//utils/datasets:stackoverflow_word_prediction",
        "//utils/models:stackoverflow_models",
    ],
)

py_library(
    name = "federated_stackoverflow",
    srcs = ["federated_stackoverflow.py"],
    srcs_version = "PY3",
    deps = [
        "//optimization/shared:training_specs",
        "//utils:keras_metrics",
        "//utils/datasets:stackoverflow_word_prediction",
        "//utils/models:stackoverflow_models",
    ],
)
