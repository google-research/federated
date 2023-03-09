load("@rules_python//python:defs.bzl", "py_binary", "py_library")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])

py_binary(
    name = "run_federated",
    srcs = ["run_federated.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":run_federated_lib"],
)

py_library(
    name = "run_federated_lib",
    srcs = ["run_federated.py"],
    srcs_version = "PY3",
    deps = [
        "//utils:task_utils",
        "//utils:training_utils",
        "//utils:utils_impl",
        "//utils/optimizers:optimizer_utils",
    ],
)
