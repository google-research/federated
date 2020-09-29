load("@rules_python//python:defs.bzl", "py_binary", "py_library")

package(default_visibility = ["//gans/experiments/emnist:__subpackages__"])

licenses(["notice"])

py_binary(
    name = "filter_examples",
    srcs = ["filter_examples.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//gans/experiments/emnist:emnist_data_utils",
        "//gans/experiments/emnist/classifier:emnist_classifier_model",
        "//utils:utils_impl",
    ],
)

py_binary(
    name = "filter_users",
    srcs = ["filter_users.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//gans/experiments/emnist:emnist_data_utils",
        "//gans/experiments/emnist/classifier:emnist_classifier_model",
        "//utils:utils_impl",
    ],
)

py_library(
    name = "filtered_emnist_data_utils",
    srcs = ["filtered_emnist_data_utils.py"],
    srcs_version = "PY3",
    deps = ["//gans/experiments/emnist:emnist_data_utils"],
)
