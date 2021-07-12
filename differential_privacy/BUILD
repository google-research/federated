load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_binary(
    name = "federated_stackoverflow_with_secrets",
    srcs = ["federated_stackoverflow_with_secrets.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":secret_sharer",
        "//utils:task_utils",
        "//utils:training_utils",
        "//utils:utils_impl",
        "//utils/datasets:stackoverflow_word_prediction",
        "//utils/models:stackoverflow_models",
        "//utils/optimizers:optimizer_utils",
    ],
)

py_binary(
    name = "run_federated",
    srcs = ["run_federated.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//utils:task_utils",
        "//utils:training_utils",
        "//utils:utils_impl",
        "//utils/optimizers:optimizer_utils",
    ],
)

py_library(
    name = "secret_sharer",
    srcs = ["secret_sharer.py"],
    srcs_version = "PY3",
)

py_test(
    name = "secret_sharer_test",
    srcs = ["secret_sharer_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":secret_sharer"],
)
