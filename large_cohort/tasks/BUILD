load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "cifar100",
    srcs = ["cifar100.py"],
    srcs_version = "PY3",
    deps = [
        "//large_cohort:simulation_specs",
        "//utils/datasets:cifar100_dataset",
        "//utils/models:resnet_models",
    ],
)

py_library(
    name = "emnist",
    srcs = ["emnist.py"],
    srcs_version = "PY3",
    deps = [
        "//large_cohort:simulation_specs",
        "//utils/datasets:emnist_dataset",
        "//utils/models:emnist_models",
    ],
)

py_library(
    name = "emnist_autoencoder",
    srcs = ["emnist_autoencoder.py"],
    srcs_version = "PY3",
    deps = [
        "//large_cohort:simulation_specs",
        "//utils/datasets:emnist_dataset",
        "//utils/models:emnist_ae_models",
    ],
)

py_library(
    name = "shakespeare",
    srcs = ["shakespeare.py"],
    srcs_version = "PY3",
    deps = [
        "//large_cohort:simulation_specs",
        "//utils:keras_metrics",
        "//utils/datasets:shakespeare_dataset",
        "//utils/models:shakespeare_models",
    ],
)

py_library(
    name = "stackoverflow_tag_prediction",
    srcs = ["stackoverflow_tag_prediction.py"],
    srcs_version = "PY3",
    deps = [
        "//large_cohort:simulation_specs",
        "//utils/datasets:stackoverflow_tag_prediction",
        "//utils/models:stackoverflow_lr_models",
    ],
)

py_library(
    name = "stackoverflow_word_prediction",
    srcs = ["stackoverflow_word_prediction.py"],
    srcs_version = "PY3",
    deps = [
        "//large_cohort:simulation_specs",
        "//utils:keras_metrics",
        "//utils/datasets:stackoverflow_word_prediction",
        "//utils/models:stackoverflow_models",
    ],
)
