# Federated dual encoder.

load("@rules_python//python:defs.bzl", "py_library", "py_test")

licenses(["notice"])

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [":__subpackages__"],
)

py_library(
    name = "encoders",
    srcs = ["encoders.py"],
    srcs_version = "PY3",
    deps = [":keras_layers"],
)

py_test(
    name = "encoders_test",
    srcs = ["encoders_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":encoders"],
)

py_library(
    name = "keras_layers",
    srcs = ["keras_layers.py"],
    srcs_version = "PY3",
)

py_test(
    name = "keras_layers_test",
    srcs = ["keras_layers_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":keras_layers"],
)

py_library(
    name = "losses",
    srcs = ["losses.py"],
    srcs_version = "PY3",
    deps = [":model_utils"],
)

py_test(
    name = "losses_test",
    srcs = ["losses_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":losses",
        ":model_utils",
    ],
)

py_library(
    name = "metrics",
    srcs = ["metrics.py"],
    srcs_version = "PY3",
    deps = [":model_utils"],
)

py_test(
    name = "metrics_test",
    srcs = ["metrics_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":metrics",
        ":model_utils",
    ],
)

py_library(
    name = "model_utils",
    srcs = ["model_utils.py"],
    srcs_version = "PY3",
)

py_test(
    name = "model_utils_test",
    srcs = ["model_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":model_utils"],
)

py_library(
    name = "run_utils",
    srcs = ["run_utils.py"],
    srcs_version = "PY3",
)

py_test(
    name = "run_utils_test",
    timeout = "long",
    srcs = ["run_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":run_utils"],
)
