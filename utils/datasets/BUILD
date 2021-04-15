load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "cifar10_dataset",
    srcs = ["cifar10_dataset.py"],
    srcs_version = "PY3",
)

py_test(
    name = "cifar10_dataset_test",
    size = "large",
    srcs = ["cifar10_dataset_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "manual",
        "nokokoro",
        "nopresubmit",
    ],
    deps = [":cifar10_dataset"],
)

py_library(
    name = "cifar100_dataset",
    srcs = ["cifar100_dataset.py"],
    srcs_version = "PY3",
)

py_test(
    name = "cifar100_dataset_test",
    size = "medium",
    srcs = ["cifar100_dataset_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":cifar100_dataset"],
)

py_library(
    name = "emnist_dataset",
    srcs = ["emnist_dataset.py"],
    srcs_version = "PY3",
)

py_test(
    name = "emnist_dataset_test",
    size = "medium",
    srcs = ["emnist_dataset_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":emnist_dataset"],
)

py_library(
    name = "infinite_emnist",
    srcs = ["infinite_emnist.py"],
    srcs_version = "PY3",
)

py_test(
    name = "infinite_emnist_test",
    size = "medium",
    srcs = ["infinite_emnist_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "manual",
        "nokokoro",
    ],
    deps = [":infinite_emnist"],
)

py_library(
    name = "shakespeare_dataset",
    srcs = ["shakespeare_dataset.py"],
    srcs_version = "PY3",
)

py_test(
    name = "shakespeare_dataset_test",
    size = "medium",
    srcs = ["shakespeare_dataset_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":shakespeare_dataset"],
)

py_library(
    name = "stackoverflow_word_prediction",
    srcs = ["stackoverflow_word_prediction.py"],
    srcs_version = "PY3",
)

py_test(
    name = "stackoverflow_word_prediction_test",
    size = "medium",
    srcs = ["stackoverflow_word_prediction_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":stackoverflow_word_prediction"],
)

py_library(
    name = "stackoverflow_tag_prediction",
    srcs = ["stackoverflow_tag_prediction.py"],
    srcs_version = "PY3",
)

py_test(
    name = "stackoverflow_tag_prediction_test",
    size = "medium",
    srcs = ["stackoverflow_tag_prediction_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":stackoverflow_tag_prediction"],
)
