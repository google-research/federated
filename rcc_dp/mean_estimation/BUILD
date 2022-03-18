load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_visibility = ["//rcc_dp:__subpackages__"],
)

licenses(["notice"])

py_library(
    name = "config",
    srcs = ["config.py"],
)

py_library(
    name = "experiment",
    srcs = ["experiment.py"],
    deps = [
        ":get_parameters",
        ":miracle",
        ":privunit",
        ":sqkr",
        "//rcc_dp:modify_pi",
    ],
)

py_library(
    name = "experiment_coding_cost",
    srcs = ["experiment_coding_cost.py"],
    deps = [
        ":get_parameters",
        ":miracle",
        ":privunit",
        ":sqkr",
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
    name = "get_parameters",
    srcs = ["get_parameters.py"],
    deps = [
        ":optimize_unbias",
        ":privunit",
    ],
)

py_test(
    name = "get_parameters_test",
    srcs = ["get_parameters_test.py"],
    deps = [
        ":get_parameters",
        ":miracle",
        "//rcc_dp:modify_pi",
    ],
)

py_library(
    name = "miracle",
    srcs = ["miracle.py"],
)

py_test(
    name = "miracle_test",
    srcs = ["miracle_test.py"],
    deps = [
        ":get_parameters",
        ":miracle",
    ],
)

py_library(
    name = "optimize_unbias",
    srcs = ["optimize_unbias.py"],
    deps = [":privunit"],
)

py_library(
    name = "privunit",
    srcs = ["privunit.py"],
)

py_test(
    name = "privunit_test",
    srcs = ["privunit_test.py"],
    deps = [":privunit"],
)

py_library(
    name = "sqkr",
    srcs = ["sqkr.py"],
)

py_test(
    name = "sqkr_test",
    srcs = ["sqkr_test.py"],
    deps = [":sqkr"],
)
