load("@rules_python//python:defs.bzl", "py_library")

licenses(["notice"])

py_library(
    name = "synthesizer_config",
    srcs = ["synthesizer_config.py"],
    deps = [":config_utils"],
)

py_library(
    name = "config_utils",
    srcs = ["config_utils.py"],
)

py_library(
    name = "federated_config",
    srcs = ["federated_config.py"],
    deps = [":config_utils"],
)

py_library(
    name = "centralized_config",
    srcs = ["centralized_config.py"],
    deps = [":config_utils"],
)
