load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_test(
    name = "transformer_models_test",
    srcs = ["transformer_models_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":transformer_models"],
)

py_library(
    name = "transformer_models",
    srcs = ["transformer_models.py"],
    srcs_version = "PY3",
    deps = ["//utils/models:stackoverflow_models"],
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
        "//optimization/shared:optimizer_utils",
        "//utils:utils_impl",
    ],
)

py_library(
    name = "centralized_main",
    srcs = ["centralized_main.py"],
    srcs_version = "PY3",
    deps = [
        ":transformer_models",
        "//optimization/shared:keras_metrics",
        "//utils:centralized_training_loop",
        "//utils/datasets:stackoverflow_word_prediction",
    ],
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
    name = "federated_main",
    srcs = ["federated_main.py"],
    srcs_version = "PY3",
    deps = [
        ":transformer_models",
        "//fedopt_guide:training_loop",
        "//optimization/shared:keras_metrics",
        "//utils/datasets:stackoverflow_word_prediction",
    ],
)

py_test(
    name = "federated_main_test",
    size = "large",
    timeout = "long",
    srcs = ["federated_main_test.py"],
    python_version = "PY3",
    shard_count = 4,
    srcs_version = "PY3",
    tags = [
        "manual",
        "nokokoro",
        "nopresubmit",
    ],
    deps = [":federated_main"],
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
        ":federated_main",
        "//optimization/shared:optimizer_utils",
        "//utils:utils_impl",
    ],
)
