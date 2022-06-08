load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

# Enables linear compression aggregators in Federated Learning as a means to achieve optimal privacy-accuracy-communication tradeoffs.

licenses(["notice"])

py_library(
    name = "count_sketching",
    srcs = ["count_sketching.py"],
    srcs_version = "PY3",
    deps = [":count_sketching_utils"],
)

py_library(
    name = "ddp_compression",
    srcs = ["ddp_compression.py"],
    srcs_version = "PY3",
    deps = [
        ":count_sketching",
        ":count_sketching_utils",
    ],
)

py_test(
    name = "ddp_compression_test",
    size = "large",
    srcs = ["ddp_compression_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":ddp_compression"],
)

py_test(
    name = "count_sketching_test",
    srcs = ["count_sketching_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":count_sketching",
        ":count_sketching_utils",
    ],
)

py_library(
    name = "count_sketching_utils",
    srcs = ["count_sketching_utils.py"],
    srcs_version = "PY3",
)

py_test(
    name = "count_sketching_utils_test",
    srcs = ["count_sketching_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":count_sketching_utils"],
)

py_binary(
    name = "fl_compression_run",
    srcs = ["fl_compression_run.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":count_sketching",
        ":count_sketching_utils",
        ":ddp_compression",
        "//utils:task_utils",
        "//utils:training_utils",
        "//utils:utils_impl",
        "//utils/optimizers:optimizer_utils",
    ],
)
