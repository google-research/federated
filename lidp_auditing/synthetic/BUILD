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
    name = "generate_data_lib",
    srcs = ["generate_data.py"],
    deps = ["//third_party/py/numba"],
)

py_test(
    name = "generate_data_test",
    timeout = "long",
    srcs = ["generate_data_test.py"],
    deps = [":generate_data_lib"],
)

py_library(
    name = "process_data_lib",
    srcs = ["process_data.py"],
    deps = [
        "//lidp_auditing/confidence_estimators:asymptotic_lib",
        "//lidp_auditing/confidence_estimators:asymptotic_vectorized_lib",
        "//third_party/py/fsspec",  # build-cleaner: keep
    ],
)

py_test(
    name = "process_data_test",
    timeout = "long",
    srcs = ["process_data_test.py"],
    deps = [
        ":generate_data_lib",
        ":process_data_lib",
    ],
)

py_binary(
    name = "run_synthetic",
    srcs = ["run_synthetic.py"],
    deps = [
        ":generate_data_lib",
        ":process_data_lib",
    ],
)
