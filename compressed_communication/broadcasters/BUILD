load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_visibility = ["//compressed_communication:__subpackages__"],
)

licenses(["notice"])

py_library(
    name = "histogram_model",
    srcs = ["histogram_model.py"],
    srcs_version = "PY3",
)

py_test(
    name = "histogram_model_test",
    srcs = ["histogram_model_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":histogram_model"],
)
