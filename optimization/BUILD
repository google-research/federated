load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(default_visibility = ["//google/optimization:__subpackages__"])

licenses(["notice"])

py_library(
    name = "fed_avg_client_opt",
    srcs = ["fed_avg_client_opt.py"],
    srcs_version = "PY3",
    deps = ["//utils:tensor_utils"],
)

py_test(
    name = "fed_avg_client_opt_test",
    size = "large",
    srcs = ["fed_avg_client_opt_test.py"],
    python_version = "PY3",
    shard_count = 2,
    srcs_version = "PY3",
    deps = [":fed_avg_client_opt"],
)

py_library(
    name = "fed_avg_schedule",
    srcs = ["fed_avg_schedule.py"],
    srcs_version = "PY3",
    deps = ["//utils:tensor_utils"],
)

py_test(
    name = "fed_avg_schedule_test",
    size = "large",
    srcs = ["fed_avg_schedule_test.py"],
    python_version = "PY3",
    shard_count = 2,
    srcs_version = "PY3",
    deps = [":fed_avg_schedule"],
)

py_binary(
    name = "trainer",
    srcs = ["trainer.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":trainer_lib"],
)

py_library(
    name = "trainer_lib",
    srcs = ["trainer.py"],
    srcs_version = "PY3",
    deps = [
        ":fed_avg_schedule",
        "//optimization/tasks:cifar100",
        "//optimization/tasks:emnist",
        "//optimization/tasks:emnist_ae",
        "//optimization/tasks:shakespeare",
        "//optimization/tasks:stackoverflow_nwp",
        "//optimization/tasks:stackoverflow_tp",
        "//optimization/tasks:training_specs",
        "//utils:utils_impl",
        "//utils/optimizers:optimizer_utils",
    ],
)

py_test(
    name = "trainer_test",
    size = "large",
    srcs = ["trainer_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "manual",
        "nokokoro",
        "nopresubmit",
    ],
    deps = [
        "//optimization/tasks:cifar100",
        "//optimization/tasks:emnist",
        "//optimization/tasks:emnist_ae",
        "//optimization/tasks:shakespeare",
        "//optimization/tasks:stackoverflow_nwp",
        "//optimization/tasks:stackoverflow_tp",
        "//optimization/tasks:training_specs",
    ],
)
