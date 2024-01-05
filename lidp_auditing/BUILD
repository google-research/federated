# Build all for the code.

load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

licenses(["notice"])

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [":lidp_auditing_packages"],
)

package_group(
    name = "lidp_auditing_packages",
    packages = ["//lidp_auditing/..."],
)

py_library(
    name = "constants_lib",
    srcs = ["constants.py"],
)

py_library(
    name = "data_lib",
    srcs = ["data.py"],
    deps = [":constants_lib"],
)

py_test(
    name = "data_test",
    timeout = "long",
    srcs = ["data_test.py"],
    deps = [
        ":constants_lib",
        ":data_lib",
    ],
)

py_library(
    name = "models_lib",
    srcs = ["models.py"],
    deps = [":constants_lib"],
)

py_library(
    name = "utils_lib",
    srcs = ["utils.py"],
)

py_library(
    name = "auditing_eval_lib",
    srcs = ["auditing_eval.py"],
    deps = [
        ":constants_lib",
        ":utils_lib",
    ],
)

py_library(
    name = "auditing_trainer_lib",
    srcs = ["auditing_trainer.py"],
    deps = [
        ":auditing_eval_lib",
        ":constants_lib",
        ":utils_lib",
    ],
)

py_binary(
    name = "main_central",
    srcs = ["main_central.py"],
    deps = [
        ":auditing_trainer_lib",
        ":constants_lib",
        ":data_lib",
        ":models_lib",
        ":utils_lib",
    ],
)
