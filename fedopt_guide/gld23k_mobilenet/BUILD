load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_test(
    name = "dataset_test",
    srcs = ["dataset_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "manual",
        "nokokoro",
        "nopresubmit",
    ],
    deps = [":dataset"],
)

py_test(
    name = "mobilenet_v2_test",
    size = "small",
    timeout = "moderate",
    srcs = ["mobilenet_v2_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":mobilenet_v2"],
)

py_test(
    name = "federated_main_test",
    timeout = "long",
    srcs = ["federated_main_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "manual",
        "nokokoro",
        "nopresubmit",
    ],
    deps = [":federated_main"],
)

py_test(
    name = "centralized_main_test",
    srcs = ["centralized_main_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "manual",
        "nokokoro",
        "nopresubmit",
    ],
    deps = [":centralized_main"],
)

py_library(
    name = "dataset",
    srcs = ["dataset.py"],
    srcs_version = "PY3",
)

py_library(
    name = "mobilenet_v2",
    srcs = ["mobilenet_v2.py"],
    srcs_version = "PY3",
)

py_library(
    name = "federated_main",
    srcs = ["federated_main.py"],
    srcs_version = "PY3",
    deps = [
        ":dataset",
        ":mobilenet_v2",
        "//fedopt_guide:training_loop",
    ],
)

py_library(
    name = "centralized_main",
    srcs = ["centralized_main.py"],
    srcs_version = "PY3",
    deps = [
        ":dataset",
        ":mobilenet_v2",
        "//utils:centralized_training_loop",
    ],
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
        ":dataset",
        ":federated_main",
        "//optimization/shared:optimizer_utils",
        "//utils:utils_impl",
    ],
)

py_binary(
    name = "centralized_trainer",
    srcs = ["centralized_trainer.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":centralized_trainer_lib"],
)

py_library(
    name = "centralized_trainer_lib",
    srcs = ["centralized_trainer.py"],
    srcs_version = "PY3",
    deps = [
        ":centralized_main",
        ":dataset",
        "//optimization/shared:optimizer_utils",
        "//utils:utils_impl",
    ],
)
