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
    name = "asymptotic_lib",
    srcs = ["asymptotic.py"],
)

py_test(
    name = "asymptotic_test",
    timeout = "long",
    srcs = ["asymptotic_test.py"],
    deps = [":asymptotic_lib"],
)

py_library(
    name = "asymptotic_vectorized_lib",
    srcs = ["asymptotic_vectorized.py"],
)

py_test(
    name = "asymptotic_vectorized_test",
    timeout = "long",
    srcs = ["asymptotic_vectorized_test.py"],
    deps = [
        ":asymptotic_lib",
        ":asymptotic_vectorized_lib",
    ],
)
