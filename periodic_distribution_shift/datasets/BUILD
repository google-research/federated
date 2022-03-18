load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//periodic_distribution_shift:__subpackages__"])

licenses(["notice"])

py_library(
    name = "stackoverflow_nwp_preprocessing",
    srcs = ["stackoverflow_nwp_preprocessing.py"],
)

py_test(
    name = "stackoverflow_nwp_preprocessing_test",
    srcs = ["stackoverflow_nwp_preprocessing_test.py"],
    deps = [":stackoverflow_nwp_preprocessing"],
)

py_library(
    name = "cifar_classification_preprocessing",
    srcs = ["cifar_classification_preprocessing.py"],
)

py_library(
    name = "emnist_preprocessing",
    srcs = ["emnist_preprocessing.py"],
)

py_library(
    name = "client_sampling",
    srcs = ["client_sampling.py"],
    deps = [
        ":cifar_classification_preprocessing",
        ":emnist_preprocessing",
        ":stackoverflow_nwp_preprocessing",
        "//utils/datasets:cifar10_dataset",
    ],
)

py_test(
    name = "cifar_classification_preprocessing_test",
    srcs = ["cifar_classification_preprocessing_test.py"],
    deps = [":cifar_classification_preprocessing"],
)

py_test(
    name = "emnist_preprocessing_test",
    srcs = ["emnist_preprocessing_test.py"],
    deps = [":emnist_preprocessing"],
)

py_test(
    name = "client_sampling_test",
    timeout = "long",
    srcs = ["client_sampling_test.py"],
    shard_count = 5,
    tags = [
        "requires-net:external",
    ],
    deps = [":client_sampling"],
)
