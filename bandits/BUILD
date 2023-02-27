load("@rules_python//python:defs.bzl", "py_library", "py_test")

licenses(["notice"])

package(default_visibility = ["//bandits:__subpackages__"])

py_library(
    name = "keras_optimizer_utils",
    srcs = ["keras_optimizer_utils.py"],
)

py_library(
    name = "bandits_utils",
    srcs = ["bandits_utils.py"],
)

py_library(
    name = "file_utils",
    srcs = ["file_utils.py"],
)

py_test(
    name = "file_utils_test",
    srcs = ["file_utils_test.py"],
    deps = [":file_utils"],
)

py_test(
    name = "bandits_utils_test",
    srcs = ["bandits_utils_test.py"],
    deps = [":bandits_utils"],
)

py_test(
    name = "keras_optimizer_utils_test",
    srcs = ["keras_optimizer_utils_test.py"],
    deps = [":keras_optimizer_utils"],
)
