load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_test(
    name = "transformer_models_test",
    srcs = ["transformer_models_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":transformer_models"],
)

py_library(
    name = "transformer_models",
    srcs = ["transformer_models.py"],
    srcs_version = "PY3",
    deps = ["//utils/models:stackoverflow_models"],
)
