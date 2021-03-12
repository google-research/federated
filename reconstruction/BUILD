load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "reconstruction",
    srcs = ["__init__.py"],
    srcs_version = "PY3",
    deps = [
        ":evaluation_computation",
        ":keras_utils",
        ":reconstruction_model",
        ":reconstruction_utils",
        ":training_process",
    ],
)

py_library(
    name = "reconstruction_model",
    srcs = ["reconstruction_model.py"],
    srcs_version = "PY3",
)

py_library(
    name = "training_process",
    srcs = ["training_process.py"],
    srcs_version = "PY3",
    deps = [
        ":keras_utils",
        ":reconstruction_model",
        ":reconstruction_utils",
        "//utils:tensor_utils",
    ],
)

py_test(
    name = "training_process_test",
    size = "large",
    srcs = ["training_process_test.py"],
    python_version = "PY3",
    shard_count = 10,
    srcs_version = "PY3",
    deps = [
        ":keras_utils",
        ":reconstruction_model",
        ":reconstruction_utils",
        ":training_process",
    ],
)

py_library(
    name = "evaluation_computation",
    srcs = ["evaluation_computation.py"],
    srcs_version = "PY3",
    deps = [
        ":keras_utils",
        ":reconstruction_model",
        ":reconstruction_utils",
    ],
)

py_test(
    name = "evaluation_computation_test",
    size = "medium",
    srcs = ["evaluation_computation_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":evaluation_computation",
        ":keras_utils",
        ":reconstruction_model",
        ":reconstruction_utils",
    ],
)

py_library(
    name = "reconstruction_utils",
    srcs = ["reconstruction_utils.py"],
    srcs_version = "PY3",
    deps = [":reconstruction_model"],
)

py_test(
    name = "reconstruction_utils_test",
    srcs = ["reconstruction_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":keras_utils",
        ":reconstruction_utils",
    ],
)

py_library(
    name = "keras_utils",
    srcs = ["keras_utils.py"],
    srcs_version = "PY3",
    deps = [":reconstruction_model"],
)

py_test(
    name = "keras_utils_test",
    srcs = ["keras_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":keras_utils",
        ":reconstruction_model",
    ],
)
