load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":dp_matrix_factorization_experiment_packages",
    ],
)

package_group(
    name = "dp_matrix_factorization_experiment_packages",
    packages = ["//dp_matrix_factorization/dp_ftrl/..."],
)

licenses(["notice"])

py_library(
    name = "aggregator_builder",
    srcs = ["aggregator_builder.py"],
    srcs_version = "PY3",
    deps = ["//dp_matrix_factorization:tff_aggregator"],
)

py_test(
    name = "aggregator_builder_test",
    size = "large",
    srcs = ["aggregator_builder_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":aggregator_builder",
        "//dp_matrix_factorization:matrix_constructors",
    ],
)

py_library(
    name = "dp_fedavg",
    srcs = ["dp_fedavg.py"],
    srcs_version = "PY3",
)

py_library(
    name = "training_loop",
    srcs = ["training_loop.py"],
    srcs_version = "PY3",
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
        "//dp_matrix_factorization:tff_aggregator",
    ],
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
        ":aggregator_builder",
        ":dp_fedavg",
        ":training_loop",
        "//dp_matrix_factorization/dp_ftrl/datasets:stackoverflow_word_prediction",
        "//dp_matrix_factorization/dp_ftrl/models:stackoverflow_models",
        "//utils:keras_metrics",
    ],
)

py_test(
    name = "run_stackoverflow_test",
    size = "large",
    srcs = ["run_stackoverflow_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":aggregator_builder",
        ":run_stackoverflow_lib",
        "//dp_matrix_factorization:matrix_constructors",
    ],
)
