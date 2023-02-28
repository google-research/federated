load("@rules_python//python:defs.bzl", "py_library", "py_test")

licenses(["notice"])

package(default_visibility = ["//bandits:__subpackages__"])

py_library(
    name = "task_utils",
    srcs = ["task_utils.py"],
    deps = ["//bandits:bandits_utils"],
)

py_test(
    name = "task_utils_test",
    srcs = ["task_utils_test.py"],
    deps = [
        ":task_utils",
        "//bandits:bandits_utils",
    ],
)
