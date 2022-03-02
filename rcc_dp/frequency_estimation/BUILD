load("@rules_python//python:defs.bzl", "py_library", "py_test")

licenses(["notice"])

py_library(
    name = "config",
    srcs = ["config.py"],
)

py_library(
    name = "experiment",
    srcs = ["experiment.py"],
    deps = [
        ":miracle",
        ":rhr",
        ":ss",
        ":unbias",
        "//rcc_dp:modify_pi",
    ],
)

py_library(
    name = "experiment_coding_cost",
    srcs = ["experiment_coding_cost.py"],
    deps = [
        ":miracle",
        ":rhr",
        ":ss",
        ":unbias",
        "//rcc_dp:modify_pi",
    ],
)

py_test(
    name = "experiment_test",
    srcs = ["experiment_test.py"],
    deps = [
        ":config",
        ":experiment",
        ":experiment_coding_cost",
    ],
)

py_library(
    name = "miracle",
    srcs = ["miracle.py"],
)

py_library(
    name = "unbias",
    srcs = ["unbias.py"],
    deps = [":ss"],
)

py_library(
    name = "ss",
    srcs = ["ss.py"],
)

py_library(
    name = "rhr",
    srcs = ["rhr.py"],
    deps = [":ss"],
)
