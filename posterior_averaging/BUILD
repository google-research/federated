load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

licenses(["notice"])

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
        ":fed_pa_schedule",
        "//utils:task_utils",
        "//utils:training_utils",
        "//utils:utils_impl",
        "//utils/optimizers:optimizer_utils",
    ],
)

py_library(
    name = "fed_pa_schedule",
    srcs = ["fed_pa_schedule.py"],
    srcs_version = "PY3",
    deps = ["//utils:tensor_utils"],
)

py_test(
    name = "fed_pa_schedule_test",
    size = "large",
    srcs = ["fed_pa_schedule_test.py"],
    python_version = "PY3",
    shard_count = 2,
    srcs_version = "PY3",
    tags = ["nokokoro"],
    deps = [":fed_pa_schedule"],
)
