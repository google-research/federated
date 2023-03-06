load("@rules_python//python:defs.bzl", "py_library", "py_test")

licenses(["notice"])

package(default_visibility = ["//bandits:__subpackages__"])

py_library(
    name = "epsilon_greedy",
    srcs = ["epsilon_greedy.py"],
    deps = ["//bandits:bandits_utils"],
)

py_library(
    name = "falcon",
    srcs = ["falcon.py"],
    deps = ["//bandits:bandits_utils"],
)

py_library(
    name = "softmax_sampling",
    srcs = ["softmax_sampling.py"],
    deps = ["//bandits:bandits_utils"],
)

py_test(
    name = "falcon_test",
    srcs = ["falcon_test.py"],
    deps = [
        ":falcon",
        "//bandits:bandits_utils",
    ],
)

py_test(
    name = "epsilon_greedy_test",
    srcs = ["epsilon_greedy_test.py"],
    deps = [
        ":epsilon_greedy",
        "//bandits:bandits_utils",
        "//bandits/tasks:task_utils",
    ],
)

py_test(
    name = "softmax_sampling_test",
    srcs = ["softmax_sampling_test.py"],
    deps = [
        ":softmax_sampling",
        "//bandits:bandits_utils",
    ],
)
