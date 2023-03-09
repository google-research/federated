load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])

py_library(
    name = "fed_avg_local_adaptivity",
    srcs = ["fed_avg_local_adaptivity.py"],
    srcs_version = "PY3",
    deps = ["//utils:tensor_utils"],
)

py_binary(
    name = "federated_trainer",
    srcs = ["federated_trainer.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":federated_trainer_lib"],
)

py_library(
    name = "federated_trainer_lib",
    srcs = ["federated_trainer.py"],
    srcs_version = "PY3",
    deps = [
        ":fed_avg_local_adaptivity",
        "//utils:task_utils",
        "//utils:training_utils",
        "//utils:utils_impl",
        "//utils/optimizers:optimizer_utils",
    ],
)

py_test(
    name = "fed_avg_local_adaptivity_test",
    srcs = ["fed_avg_local_adaptivity_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":fed_avg_local_adaptivity",
        "//utils/optimizers:yogi",
    ],
)
