load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "centralized_training_loop",
    srcs = ["centralized_training_loop.py"],
    srcs_version = "PY3",
    deps = [":utils_impl"],
)

py_test(
    name = "centralized_training_loop_test",
    srcs = ["centralized_training_loop_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":centralized_training_loop"],
)

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
    name = "training_loop",
    srcs = ["training_loop.py"],
    srcs_version = "PY3",
)

py_test(
    name = "training_loop_test",
    size = "large",
    srcs = ["training_loop_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":training_loop"],
)

py_library(
    name = "utils_impl",
    srcs = ["utils_impl.py"],
    srcs_version = "PY3",
)

py_library(
    name = "tensor_utils",
    srcs = ["tensor_utils.py"],
    srcs_version = "PY3",
)

py_test(
    name = "utils_impl_test",
    srcs = ["utils_impl_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":utils_impl"],
)

py_test(
    name = "tensor_utils_test",
    srcs = ["tensor_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":tensor_utils"],
)
