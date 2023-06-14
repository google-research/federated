load("@rules_license//rules:license.bzl", "license")
load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("//tools/build_defs/license:license.bzl", "license")

package(
    default_applicable_licenses = ["//multi_epoch_dp_matrix_factorization:license"],
    default_visibility = [
        ":dp_matrix_factorization_packages",
        ":dp_matrix_factorization_users",
    ],
)

package_group(
    name = "dp_matrix_factorization_packages",
    packages = ["//multi_epoch_dp_matrix_factorization/..."],
)

package_group(
    name = "dp_matrix_factorization_users",
    packages = [
        "//multi_epoch_dp_matrix_factorization/...",  # Placeholder
    ],
)

license(
    name = "license",
    package_name = "multi_epoch_dp_matrix_factorization",
)

licenses(["notice"])

py_library(
    name = "fixed_point_library",
    srcs = ["fixed_point_library.py"],
    srcs_version = "PY3",
)

py_test(
    name = "fixed_point_library_test",
    srcs = ["fixed_point_library_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":fixed_point_library"],
)

py_library(
    name = "loops",
    srcs = ["loops.py"],
    srcs_version = "PY3",
    deps = [
        ":fixed_point_library",
        ":matrix_constructors",
        ":solvers",
    ],
)

py_test(
    name = "loops_test",
    srcs = ["loops_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":constraint_builders",
        ":loops",
        ":matrix_constructors",
    ],
)

py_library(
    name = "matrix_constructors",
    srcs = ["matrix_constructors.py"],
    srcs_version = "PY3",
    deps = [
        ":constraint_builders",
        ":matrix_factorization_query",
    ],
)

py_test(
    name = "matrix_constructors_test",
    srcs = ["matrix_constructors_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":matrix_constructors"],
)

py_library(
    name = "constraint_builders",
    srcs = ["constraint_builders.py"],
    srcs_version = "PY3",
)

py_test(
    name = "constraint_builders_test",
    srcs = ["constraint_builders_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":constraint_builders"],
)

py_library(
    name = "tff_aggregator",
    srcs = ["tff_aggregator.py"],
    srcs_version = "PY3",
    deps = [
        ":matrix_constructors",
        ":matrix_factorization_query",
    ],
)

py_test(
    name = "tff_aggregator_test",
    srcs = ["tff_aggregator_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":matrix_constructors",
        ":matrix_factorization_query",
        ":tff_aggregator",
    ],
)

py_library(
    name = "solvers",
    srcs = ["solvers.py"],
    srcs_version = "PY3",
)

py_test(
    name = "solvers_test",
    srcs = ["solvers_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":constraint_builders",
        ":matrix_constructors",
        ":solvers",
    ],
)

py_library(
    name = "matrix_io",
    srcs = ["matrix_io.py"],
    srcs_version = "PY3",
)

py_library(
    name = "matrix_factorization_query",
    srcs = ["matrix_factorization_query.py"],
    deps = ["//third_party/py/dp_accounting"],
)

py_library(
    name = "accounting",
    srcs = ["accounting.py"],
    deps = ["//third_party/py/dp_accounting"],
)

py_test(
    name = "matrix_io_test",
    srcs = ["matrix_io_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":matrix_io"],
)

py_test(
    name = "matrix_factorization_query_test",
    srcs = ["matrix_factorization_query_test.py"],
    deps = [
        ":matrix_factorization_query",
        "//third_party/py/tensorflow_privacy/privacy/dp_query:gaussian_query",
        "//third_party/py/tensorflow_privacy/privacy/dp_query:tree_aggregation_query",
    ],
)
