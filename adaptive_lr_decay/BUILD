load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

licenses(["notice"])

py_library(
    name = "adaptive_fed_avg",
    srcs = ["adaptive_fed_avg.py"],
    srcs_version = "PY3",
    deps = [":callbacks"],
)

py_test(
    name = "adaptive_fed_avg_test",
    size = "large",
    srcs = ["adaptive_fed_avg_test.py"],
    python_version = "PY3",
    shard_count = 2,
    srcs_version = "PY3",
    tags = [
        "manual",
        "nokokoro",
    ],
    deps = [
        ":adaptive_fed_avg",
        ":callbacks",
    ],
)

py_library(
    name = "callbacks",
    srcs = ["callbacks.py"],
    srcs_version = "PY3",
)

py_test(
    name = "callbacks_test",
    size = "small",
    srcs = ["callbacks_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":callbacks"],
)

py_binary(
    name = "federated_trainer",
    srcs = ["federated_trainer.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":adaptive_fed_avg",
        ":callbacks",
        "//utils:task_utils",
        "//utils:training_utils",
        "//utils:utils_impl",
        "//utils/optimizers:optimizer_utils",
    ],
)
