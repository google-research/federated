load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "q_ffl",
    srcs = ["q_ffl.py"],
    srcs_version = "PY3",
)

py_test(
    name = "q_ffl_test",
    size = "large",
    srcs = ["q_ffl_test.py"],
    python_version = "PY3",
    shard_count = 2,
    srcs_version = "PY3",
    deps = [":q_ffl"],
)
