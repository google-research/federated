load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":dp_matrix_factorization_experiment_packages",
        "//multi_epoch_dp_matrix_factorization:dp_matrix_factorization_packages",
        "//multi_epoch_dp_matrix_factorization:dp_matrix_factorization_users",
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
    deps = [
        "//multi_epoch_dp_matrix_factorization:matrix_io",
        "//multi_epoch_dp_matrix_factorization:tff_aggregator",
    ],
)

py_test(
    name = "aggregator_builder_test",
    size = "large",
    srcs = ["aggregator_builder_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":aggregator_builder",
        "//multi_epoch_dp_matrix_factorization:matrix_constructors",
        "//multi_epoch_dp_matrix_factorization:matrix_io",
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
    deps = ["//utils:utils_impl"],
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
        "//multi_epoch_dp_matrix_factorization:tff_aggregator",
    ],
)

py_test(
    name = "training_loop_test",
    srcs = ["training_loop_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":dp_fedavg",
        ":training_loop",
    ],
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
        "//multi_epoch_dp_matrix_factorization/dp_ftrl/datasets:stackoverflow_word_prediction",
        "//multi_epoch_dp_matrix_factorization/dp_ftrl/models:stackoverflow_models",
        "//multi_epoch_dp_matrix_factorization/multiple_participations:contrib_matrix_builders",
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
        ":run_stackoverflow_lib",
        "//multi_epoch_dp_matrix_factorization:matrix_constructors",
        "//multi_epoch_dp_matrix_factorization:matrix_io",
    ],
)
