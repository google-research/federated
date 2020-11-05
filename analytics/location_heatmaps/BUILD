load("@rules_python//python:defs.bzl", "py_binary", "py_library")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

py_library(
    name = "mechanisms",
    srcs = ["mechanisms.py"],
    srcs_version = "PY3",
)

py_library(
    name = "metrics",
    srcs = ["metrics.py"],
    srcs_version = "PY3",
)

py_library(
    name = "plotting",
    srcs = ["plotting.py"],
    srcs_version = "PY3",
    deps = [
        ":geo_utils",
        ":metrics",
    ],
)

py_library(
    name = "geo_utils",
    srcs = ["geo_utils.py"],
)

py_binary(
    name = "run_experiment",
    srcs = ["run_experiment.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":geo_utils",
        ":mechanisms",
        ":metrics",
        ":plotting",
    ],
)
