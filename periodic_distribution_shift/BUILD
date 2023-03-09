load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//periodic_distribution_shift:__subpackages__"],
)

licenses(["notice"])

py_test(
    name = "fedavg_temporal_kmeans_test",
    srcs = ["fedavg_temporal_kmeans_test.py"],
    deps = [
        ":fedavg_temporal_kmeans",
        "//periodic_distribution_shift/models:keras_utils_dual_branch_kmeans",
    ],
)

py_library(
    name = "fedavg_temporal_kmeans",
    srcs = ["fedavg_temporal_kmeans.py"],
    deps = ["//utils:tensor_utils"],
)

py_library(
    name = "validation_utils",
    srcs = ["validation_utils.py"],
    deps = ["//periodic_distribution_shift/tasks:dist_shift_task"],
)

py_library(
    name = "train_loop_kmeans",
    srcs = ["train_loop_kmeans.py"],
    deps = [":fedavg_temporal_kmeans"],
)

py_binary(
    name = "main_trainer",
    srcs = ["main_trainer.py"],
    deps = [
        ":fedavg_temporal_kmeans",
        ":train_loop_kmeans",
        ":validation_utils",
        "//periodic_distribution_shift/datasets:client_sampling",
        "//periodic_distribution_shift/tasks:task_utils",
        "//utils:training_utils",
        "//utils:utils_impl",
        "//utils/optimizers:optimizer_utils",
    ],
)
