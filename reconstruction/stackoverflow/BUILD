load("@rules_python//python:defs.bzl", "py_library", "py_test")

licenses(["notice"])

package(default_visibility = ["//visibility:public"])

py_library(
    name = "models",
    srcs = ["models.py"],
    srcs_version = "PY3",
    deps = [
        "//reconstruction:keras_utils",
        "//reconstruction:reconstruction_model",
    ],
)

py_test(
    name = "models_test",
    srcs = ["models_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":models"],
)

py_library(
    name = "federated_stackoverflow",
    srcs = ["federated_stackoverflow.py"],
    srcs_version = "PY3",
    deps = [
        ":models",
        ":stackoverflow_dataset",
        "//optimization/shared:keras_metrics",
        "//reconstruction/shared:federated_trainer_utils",
        "//utils:training_loop",
        "//utils/datasets:stackoverflow_word_prediction",
    ],
)

py_library(
    name = "stackoverflow_dataset",
    srcs = ["stackoverflow_dataset.py"],
    srcs_version = "PY3",
    deps = ["//utils/datasets:stackoverflow_word_prediction"],
)

py_test(
    name = "stackoverflow_dataset_test",
    srcs = ["stackoverflow_dataset_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":stackoverflow_dataset"],
)
