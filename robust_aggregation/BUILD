load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

py_library(
    name = "robust_federated_aggregation",
    srcs = ["robust_federated_aggregation.py"],
    srcs_version = "PY3",
)

py_test(
    name = "robust_federated_aggregation_test",
    srcs = ["robust_federated_aggregation_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":robust_federated_aggregation",
        "//utils:tensor_utils",
    ],
)
