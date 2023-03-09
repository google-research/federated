# Utilities for training federated dual encoders for next movie id prediction using MovieLens data.

load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

licenses(["notice"])

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:private"],
)

py_library(
    name = "movielens_data_gen_lib",
    srcs = ["movielens_data_gen.py"],
    srcs_version = "PY3",
)

py_binary(
    name = "movielens_data_gen",
    srcs = ["movielens_data_gen.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":movielens_data_gen_lib",
    ],
)

py_test(
    name = "movielens_data_gen_test",
    srcs = ["movielens_data_gen_test.py"],
    data = glob(["testdata/*"]),
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":movielens_data_gen_lib"],
)

py_library(
    name = "model_launcher_centralized_lib",
    srcs = ["model_launcher_centralized.py"],
    srcs_version = "PY3",
    deps = [
        ":model",
        ":movielens_data_gen_lib",
        "//dual_encoder:model_utils",
    ],
)

py_binary(
    name = "model_launcher_centralized",
    srcs = ["model_launcher_centralized.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":model_launcher_centralized_lib",
    ],
)

py_test(
    name = "model_launcher_centralized_test",
    srcs = ["model_launcher_centralized_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":model_launcher_centralized_lib"],
)

py_library(
    name = "model_launcher_tff_lib",
    srcs = ["model_launcher_tff.py"],
    srcs_version = "PY3",
    deps = [
        ":model",
        ":movielens_data_gen_lib",
        "//dual_encoder:model_utils",
        "//dual_encoder:run_utils",
        "//utils/optimizers:optimizer_utils",
    ],
)

py_binary(
    name = "model_launcher_tff",
    srcs = ["model_launcher_tff.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":model_launcher_tff_lib",
    ],
)

py_library(
    name = "model",
    srcs = ["model.py"],
    srcs_version = "PY3",
    deps = [
        "//dual_encoder:encoders",
        "//dual_encoder:keras_layers",
        "//dual_encoder:losses",
        "//dual_encoder:metrics",
        "//dual_encoder:model_utils",
    ],
)

py_test(
    name = "model_test",
    srcs = ["model_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":model",
        "//dual_encoder:encoders",
        "//dual_encoder:model_utils",
    ],
)
