load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

py_library(
    name = "attacked_fedavg",
    srcs = ["attacked_fedavg.py"],
    srcs_version = "PY3",
)

py_test(
    name = "attacked_fedavg_test",
    size = "large",
    srcs = ["attacked_fedavg_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":attacked_fedavg"],
)

py_binary(
    name = "emnist_with_targeted_attack",
    srcs = ["emnist_with_targeted_attack.py"],
    python_version = "PY3",
    deps = [":attacked_fedavg"],
)
