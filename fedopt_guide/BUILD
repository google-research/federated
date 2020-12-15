load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//fedopt_guide:__subpackages__"])

licenses(["notice"])

py_library(
    name = "training_loop",
    srcs = ["training_loop.py"],
    srcs_version = "PY3",
    deps = [
        "//utils:csv_manager",
        "//utils:utils_impl",
    ],
)

py_test(
    name = "training_loop_test",
    size = "large",
    srcs = ["training_loop_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//fedopt_guide:training_loop",
        "//utils:csv_manager",
        "//utils:training_utils",
    ],
)
