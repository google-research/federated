load("@rules_python//python:defs.bzl", "py_library", "py_test")

licenses(["notice"])

package(default_visibility = ["//bandits:__subpackages__"])

py_library(
    name = "task_utils",
    srcs = ["task_utils.py"],
    deps = ["//bandits:bandits_utils"],
)

py_library(
    name = "emnist_preprocessing",
    srcs = ["emnist_preprocessing.py"],
)

py_library(
    name = "stackoverflow",
    srcs = ["stackoverflow.py"],
    deps = [
        ":tag_prediction_preprocessing",
        ":task_utils",
    ],
)

py_library(
    name = "emnist",
    srcs = ["emnist.py"],
    deps = [
        ":emnist_preprocessing",
        ":task_utils",
    ],
)

py_library(
    name = "tag_prediction_preprocessing",
    srcs = ["tag_prediction_preprocessing.py"],
)

py_test(
    name = "task_utils_test",
    srcs = ["task_utils_test.py"],
    deps = [
        ":task_utils",
        "//bandits:bandits_utils",
    ],
)

py_test(
    name = "stackoverflow_test",
    srcs = ["stackoverflow_test.py"],
    deps = [
        ":stackoverflow",
        ":task_utils",
        "//bandits:bandits_utils",
        "//bandits/algs:epsilon_greedy",
    ],
)

py_test(
    name = "emnist_preprocessing_test",
    srcs = ["emnist_preprocessing_test.py"],
    deps = [":emnist_preprocessing"],
)

py_test(
    name = "tag_prediction_preprocessing_test",
    srcs = ["tag_prediction_preprocessing_test.py"],
    deps = [":tag_prediction_preprocessing"],
)

py_test(
    name = "emnist_test",
    srcs = ["emnist_test.py"],
    deps = [
        ":emnist",
        ":task_utils",
        "//bandits:bandits_utils",
        "//bandits/algs:epsilon_greedy",
    ],
)
