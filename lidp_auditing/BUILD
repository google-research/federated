# Build all for the code.

load("@rules_python//python:defs.bzl", "py_library", "py_test")

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
