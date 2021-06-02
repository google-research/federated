load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "cifar100",
    srcs = ["cifar100.py"],
    srcs_version = "PY3",
    deps = [
        ":training_specs",
        "//utils/datasets:cifar100_dataset",
        "//utils/models:resnet_models",
    ],
)

py_library(
    name = "emnist",
    srcs = ["emnist.py"],
    srcs_version = "PY3",
    deps = [
        ":training_specs",
        "//utils/datasets:emnist_dataset",
        "//utils/models:emnist_models",
    ],
)

py_library(
    name = "emnist_ae",
    srcs = ["emnist_ae.py"],
    srcs_version = "PY3",
    deps = [
        ":training_specs",
        "//utils/datasets:emnist_dataset",
        "//utils/models:emnist_ae_models",
    ],
)

py_library(
    name = "shakespeare",
    srcs = ["shakespeare.py"],
    srcs_version = "PY3",
    deps = [
        ":training_specs",
        "//utils:keras_metrics",
        "//utils/datasets:shakespeare_dataset",
        "//utils/models:shakespeare_models",
    ],
)

py_library(
    name = "stackoverflow_nwp",
    srcs = ["stackoverflow_nwp.py"],
    srcs_version = "PY3",
    deps = [
        ":training_specs",
        "//utils:keras_metrics",
        "//utils/datasets:stackoverflow_word_prediction",
        "//utils/models:stackoverflow_models",
    ],
)

py_library(
    name = "stackoverflow_tp",
    srcs = ["stackoverflow_tp.py"],
    srcs_version = "PY3",
    deps = [
        ":training_specs",
        "//utils/datasets:stackoverflow_tag_prediction",
        "//utils/models:stackoverflow_lr_models",
    ],
)

py_library(
    name = "training_specs",
    srcs = ["training_specs.py"],
    srcs_version = "PY3",
)
