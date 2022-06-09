load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//personalization_benchmark:__subpackages__"])

licenses(["notice"])

py_library(
    name = "emnist",
    srcs = ["emnist.py"],
    deps = ["//personalization_benchmark/cross_device:constants"],
)

py_library(
    name = "stackoverflow",
    srcs = ["stackoverflow.py"],
    deps = [
        ":emnist",
        "//personalization_benchmark/cross_device:constants",
    ],
)

py_library(
    name = "landmark",
    srcs = ["landmark.py"],
    deps = [
        ":emnist",
        ":mobilenet_v2",
        "//personalization_benchmark/cross_device:constants",
    ],
)

py_library(
    name = "mobilenet_v2",
    srcs = ["mobilenet_v2.py"],
)

py_test(
    name = "emnist_test",
    srcs = ["emnist_test.py"],
    deps = [
        ":emnist",
        "//personalization_benchmark/cross_device:constants",
    ],
)

py_test(
    name = "stackoverflow_test",
    srcs = ["stackoverflow_test.py"],
    deps = [
        ":stackoverflow",
        "//personalization_benchmark/cross_device:constants",
    ],
)

py_test(
    name = "landmark_test",
    srcs = ["landmark_test.py"],
    deps = [
        ":landmark",
        "//personalization_benchmark/cross_device:constants",
    ],
)

py_test(
    name = "mobilenet_v2_test",
    srcs = ["mobilenet_v2_test.py"],
    deps = [":mobilenet_v2"],
)
