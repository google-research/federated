load("@rules_python//python:defs.bzl", "py_library", "py_test")

# Models and data processing for MovieLens rating prediction.

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
    size = "medium",
    srcs = ["models_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":models",
        "//reconstruction:reconstruction_utils",
        "//reconstruction:training_process",
    ],
)

py_library(
    name = "movielens_dataset",
    srcs = ["movielens_dataset.py"],
    srcs_version = "PY3",
)

py_test(
    name = "movielens_dataset_test",
    srcs = ["movielens_dataset_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":movielens_dataset"],
)

py_library(
    name = "federated_movielens",
    srcs = ["federated_movielens.py"],
    srcs_version = "PY3",
    deps = [
        ":models",
        ":movielens_dataset",
        "//reconstruction/shared:federated_trainer_utils",
        "//utils:training_loop",
    ],
)
