load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "fed_pa_schedule",
    srcs = ["fed_pa_schedule.py"],
    srcs_version = "PY3",
    deps = [
        "//optimization/shared:fed_avg_schedule",
        "//utils:tensor_utils",
    ],
)
