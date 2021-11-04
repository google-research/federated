load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(default_visibility = [
    "//intelligence/federated/research/experiments/stackoverflow_with_secrets:__subpackages__",
    "//google/dp_opt:__subpackages__",
])

licenses(["notice"])

py_binary(
    name = "run_stackoverflow_with_secrets",
    srcs = ["run_stackoverflow_with_secrets.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":run_stackoverflow_with_secrets_lib"],
)

py_test(
    name = "run_stackoverflow_with_secrets_test",
    srcs = ["run_stackoverflow_with_secrets_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":run_stackoverflow_with_secrets_lib"],
)

py_library(
    name = "run_stackoverflow_with_secrets_lib",
    srcs = ["run_stackoverflow_with_secrets.py"],
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
