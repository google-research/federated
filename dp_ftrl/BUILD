load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "dp_fedavg",
    srcs = ["dp_fedavg.py"],
    srcs_version = "PY3",
    deps = [
        ":optimizer_utils",
        "//utils:tensor_utils",
    ],
)

py_library(
    name = "tree_aggregation",
    srcs = ["tree_aggregation.py"],
    srcs_version = "PY3",
)

py_library(
    name = "optimizer_utils",
    srcs = ["optimizer_utils.py"],
    srcs_version = "PY3",
    deps = [":tree_aggregation"],
)

py_library(
    name = "training_loop",
    srcs = ["training_loop.py"],
    srcs_version = "PY3",
    deps = [
        ":dp_fedavg",
        "//utils:utils_impl",
    ],
)

py_test(
    name = "tree_aggregation_test",
    size = "large",
    srcs = ["tree_aggregation_test.py"],
    python_version = "PY3",
    shard_count = 10,
    srcs_version = "PY3",
    deps = [":tree_aggregation"],
)

py_test(
    name = "dp_fedavg_test",
    size = "large",
    srcs = ["dp_fedavg_test.py"],
    python_version = "PY3",
    shard_count = 10,
    srcs_version = "PY3",
    deps = [
        ":dp_fedavg",
        ":optimizer_utils",
        ":tree_aggregation",
    ],
)

py_test(
    name = "optimizer_utils_test",
    srcs = ["optimizer_utils_test.py"],
    python_version = "PY3",
    shard_count = 10,
    srcs_version = "PY3",
    deps = [":optimizer_utils"],
)

py_test(
    name = "training_loop_test",
    srcs = ["training_loop_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":training_loop"],
)

py_binary(
    name = "run_stackoverflow",
    srcs = ["run_stackoverflow.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":run_stackoverflow_lib"],
)

py_library(
    name = "run_stackoverflow_lib",
    srcs = ["run_stackoverflow.py"],
    srcs_version = "PY3",
    deps = [
        ":dp_fedavg",
        ":optimizer_utils",
        ":training_loop",
        "//optimization/shared:keras_metrics",
        "//utils/datasets:stackoverflow_word_prediction",
        "//utils/models:stackoverflow_models",
    ],
)

py_binary(
    name = "run_emnist",
    srcs = ["run_emnist.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":dp_fedavg",
        ":optimizer_utils",
        ":training_loop",
    ],
)
