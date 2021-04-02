load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

py_binary(
    name = "run_experiment",
    srcs = ["run_experiment.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":sparsity",
        "//utils:training_loop",
        "//utils:utils_impl",
        "//utils/datasets:emnist_dataset",
        "//utils/models:emnist_models",
    ],
)

py_library(
    name = "sparsity",
    srcs = ["sparsity.py"],
    srcs_version = "PY3",
)

py_test(
    name = "sparsity_test",
    srcs = ["sparsity_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":sparsity"],
)
