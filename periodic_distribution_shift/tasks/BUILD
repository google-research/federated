load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//periodic_distribution_shift:__subpackages__"])

licenses(["notice"])

py_library(
    name = "dist_shift_task_data",
    srcs = ["dist_shift_task_data.py"],
)

py_library(
    name = "dist_shift_task",
    srcs = ["dist_shift_task.py"],
    deps = [":dist_shift_task_data"],
)

py_library(
    name = "stackoverflow_nwp_tasks",
    srcs = ["stackoverflow_nwp_tasks.py"],
    deps = [
        ":dist_shift_task",
        ":dist_shift_task_data",
        "//periodic_distribution_shift/datasets:stackoverflow_nwp_preprocessing",
        "//periodic_distribution_shift/models:keras_utils_dual_branch_kmeans_lm",
    ],
)

py_library(
    name = "task_utils",
    srcs = ["task_utils.py"],
    deps = [
        ":cifar_classification_tasks",
        ":dist_shift_task",
        ":emnist_classification_tasks",
        ":stackoverflow_nwp_tasks",
    ],
)

py_library(
    name = "emnist_classification_tasks",
    srcs = ["emnist_classification_tasks.py"],
    deps = [
        ":dist_shift_task",
        ":dist_shift_task_data",
        "//periodic_distribution_shift/datasets:emnist_preprocessing",
        "//periodic_distribution_shift/models:keras_utils_dual_branch_kmeans",
    ],
)

py_library(
    name = "cifar_classification_tasks",
    srcs = ["cifar_classification_tasks.py"],
    deps = [
        ":dist_shift_task",
        ":dist_shift_task_data",
        "//periodic_distribution_shift/datasets:cifar_classification_preprocessing",
        "//periodic_distribution_shift/models:dual_branch_resnet_models",
        "//periodic_distribution_shift/models:keras_utils_dual_branch_kmeans",
        "//utils/datasets:cifar10_dataset",
    ],
)

py_test(
    name = "dist_shift_task_data_test",
    srcs = ["dist_shift_task_data_test.py"],
    deps = [":dist_shift_task_data"],
)

py_test(
    name = "dist_shift_task_test",
    srcs = ["dist_shift_task_test.py"],
    deps = [
        ":dist_shift_task",
        ":dist_shift_task_data",
    ],
)

py_test(
    name = "emnist_classification_tasks_test",
    srcs = ["emnist_classification_tasks_test.py"],
    tags = [
        "requires-net:external",
    ],
    deps = [
        ":dist_shift_task",
        ":emnist_classification_tasks",
    ],
)
