load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

licenses(["notice"])

py_library(
    name = "aggregation",
    srcs = ["aggregation.py"],
    srcs_version = "PY3",
)

py_test(
    name = "aggregation_test",
    size = "small",
    srcs = ["aggregation_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":aggregation"],
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
        ":aggregation",
        ":data_utils",
        ":file_utils",
        ":flag_utils",
        ":warmup",
        "//utils:task_utils",
        "//utils:training_utils",
    ],
)

py_library(
    name = "data_utils",
    srcs = ["data_utils.py"],
    srcs_version = "PY3",
)

py_test(
    name = "data_utils_test",
    size = "small",
    srcs = ["data_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":data_utils"],
)

py_library(
    name = "file_utils",
    srcs = ["file_utils.py"],
    srcs_version = "PY3",
)

py_test(
    name = "file_utils_test",
    size = "small",
    srcs = ["file_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":file_utils"],
)

py_library(
    name = "flag_utils",
    srcs = ["flag_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//utils/optimizers:lars",
        "//utils/optimizers:shampoo",
        "//utils/optimizers:yogi",
    ],
)

py_test(
    name = "flag_utils_test",
    size = "small",
    srcs = ["flag_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":flag_utils"],
)

py_library(
    name = "simulation_specs",
    srcs = ["simulation_specs.py"],
    srcs_version = "PY3",
)

py_library(
    name = "warmup",
    srcs = ["warmup.py"],
    srcs_version = "PY3",
)

py_test(
    name = "warmup_test",
    size = "small",
    srcs = ["warmup_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":warmup"],
)
