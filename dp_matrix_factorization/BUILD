load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

licenses(["notice"])

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        ":dp_matrix_factorization_packages",
    ],
)

package_group(
    name = "dp_matrix_factorization_packages",
    packages = ["//dp_matrix_factorization/..."],
)

py_binary(
    name = "factorize_prefix_sum",
    srcs = ["factorize_prefix_sum.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":initializers",
        ":loops",
    ],
)

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
    name = "initializers",
    srcs = ["initializers.py"],
    srcs_version = "PY3",
    deps = [":matrix_constructors"],
)

py_test(
    name = "initializers_test",
    srcs = ["initializers_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":initializers"],
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
    name = "matrix_factorization_query",
    srcs = ["matrix_factorization_query.py"],
    srcs_version = "PY3",
)

py_test(
    name = "matrix_factorization_query_test",
    srcs = ["matrix_factorization_query_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":matrix_factorization_query"],
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
