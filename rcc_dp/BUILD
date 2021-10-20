load("@rules_python//python:defs.bzl", "py_library", "py_test")

licenses(["notice"])

py_library(
    name = "config",
    srcs = ["config.py"],
    srcs_version = "PY3",
)

py_library(
    name = "get_parameters",
    srcs = ["get_parameters.py"],
    srcs_version = "PY3",
    deps = [
        ":optimize_unbias",
        ":privunit",
    ],
)

py_library(
    name = "experiment",
    srcs = ["experiment.py"],
    srcs_version = "PY3",
    deps = [
        ":get_parameters",
        ":miracle",
        ":modify_pi",
        ":privunit",
        ":sqkr",
    ],
)

py_library(
    name = "miracle",
    srcs = ["miracle.py"],
    srcs_version = "PY3",
)

py_library(
    name = "modify_pi",
    srcs = ["modify_pi.py"],
    srcs_version = "PY3",
)

py_library(
    name = "optimize_unbias",
    srcs = ["optimize_unbias.py"],
    srcs_version = "PY3",
    deps = [":privunit"],
)

py_library(
    name = "privunit",
    srcs = ["privunit.py"],
    srcs_version = "PY3",
)

py_library(
    name = "sqkr",
    srcs = ["sqkr.py"],
)

py_test(
    name = "experiment_test",
    size = "large",
    srcs = ["experiment_test.py"],
    python_version = "PY3",
    deps = [
        ":config",
        ":experiment",
    ],
)

py_test(
    name = "get_parameters_test",
    srcs = ["get_parameters_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["nokokoro"],  # Runtime too high for continuous integration.
    deps = [
        ":get_parameters",
        ":miracle",
        ":modify_pi",
    ],
)

py_test(
    name = "miracle_test",
    srcs = ["miracle_test.py"],
    python_version = "PY3",
    deps = [
        ":get_parameters",
        ":miracle",
    ],
)

py_test(
    name = "modify_pi_test",
    srcs = ["modify_pi_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":get_parameters",
        ":miracle",
        ":modify_pi",
    ],
)

py_test(
    name = "privunit_test",
    srcs = ["privunit_test.py"],
    python_version = "PY3",
    deps = [":privunit"],
)

py_test(
    name = "sqkr_test",
    srcs = ["sqkr_test.py"],
    python_version = "PY3",
    deps = [":sqkr"],
)
