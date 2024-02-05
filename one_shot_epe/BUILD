load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
)

licenses(["notice"])

py_library(
    name = "dot_product_utils",
    srcs = ["dot_product_utils.py"],
    srcs_version = "PY3",
)

py_test(
    name = "dot_product_utils_test",
    srcs = ["dot_product_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":dot_product_utils"],
)

py_library(
    name = "canary_insertion",
    srcs = ["canary_insertion.py"],
    srcs_version = "PY3",
)

py_test(
    name = "canary_insertion_test",
    srcs = ["canary_insertion_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":canary_insertion"],
)

py_library(
    name = "empirical_privacy_estimation_lib",
    srcs = ["empirical_privacy_estimation_lib.py"],
)

py_test(
    name = "empirical_privacy_estimation_test",
    srcs = ["empirical_privacy_estimation_test.py"],
    shard_count = 20,
    deps = [":empirical_privacy_estimation_lib"],
)

py_library(
    name = "fed_avg_with_canaries",
    srcs = ["fed_avg_with_canaries.py"],
    srcs_version = "PY3",
    deps = [":dot_product_utils"],
)

py_test(
    name = "fed_avg_with_canaries_test",
    srcs = ["fed_avg_with_canaries_test.py"],
    python_version = "PY3",
    shard_count = 4,
    srcs_version = "PY3",
    deps = [
        ":canary_insertion",
        ":dot_product_utils",
        ":fed_avg_with_canaries",
    ],
)

py_binary(
    name = "train",
    srcs = ["train.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":canary_insertion",
        ":fed_avg_with_canaries",
        ":train_lib",
        "//utils:task_utils",
        "//utils/optimizers:optimizer_utils",
    ],
)

py_library(
    name = "train_lib",
    srcs = ["train_lib.py"],
    srcs_version = "PY3",
    deps = [
        ":dot_product_utils",
        "//utils:utils_impl",
    ],
)

py_test(
    name = "train_lib_test",
    srcs = ["train_lib_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":dot_product_utils",
        ":train_lib",
    ],
)
