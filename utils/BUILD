load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "keras_metrics",
    srcs = ["keras_metrics.py"],
    srcs_version = "PY3",
)

py_test(
    name = "keras_metrics_test",
    size = "small",
    srcs = ["keras_metrics_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":keras_metrics"],
)

py_library(
    name = "task_utils",
    srcs = ["task_utils.py"],
    srcs_version = "PY3",
)

py_test(
    name = "task_utils_test",
    srcs = ["task_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["nokokoro"],
    deps = [":task_utils"],
)

py_library(
    name = "tensor_utils",
    srcs = ["tensor_utils.py"],
    srcs_version = "PY3",
)

py_test(
    name = "tensor_utils_test",
    srcs = ["tensor_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":tensor_utils"],
)

py_library(
    name = "training_utils",
    srcs = ["training_utils.py"],
    srcs_version = "PY3",
    deps = [":utils_impl"],
)

py_test(
    name = "training_utils_test",
    srcs = ["training_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["nokokoro"],
    deps = [":training_utils"],
)

py_library(
    name = "utils_impl",
    srcs = ["utils_impl.py"],
    srcs_version = "PY3",
)

py_test(
    name = "utils_impl_test",
    srcs = ["utils_impl_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":utils_impl"],
)
