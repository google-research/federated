load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_visibility = ["//compressed_communication:__subpackages__"],
)

licenses(["notice"])

py_library(
    name = "quantize_utils",
    srcs = ["quantize_utils.py"],
    srcs_version = "PY3",
)

py_test(
    name = "quantize_utils_test",
    srcs = ["quantize_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":quantize_utils"],
)
