load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        "//multi_epoch_dp_matrix_factorization:dp_matrix_factorization_packages",
        "//multi_epoch_dp_matrix_factorization:dp_matrix_factorization_users",
    ],
)

licenses(["notice"])

py_test(
    name = "primal_optimization_test",
    srcs = ["primal_optimization_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":contrib_matrix_builders",
        ":lagrange_terms",
        ":optimization",
        ":primal_optimization",
    ],
)

py_library(
    name = "primal_optimization",
    srcs = ["primal_optimization.py"],
    srcs_version = "PY3",
    deps = ["//third_party/py/typing_extensions"],
)

py_binary(
    name = "primal_experiments",
    srcs = ["primal_experiments.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":primal_experiments_lib"],
)

py_library(
    name = "primal_experiments_lib",
    srcs = ["primal_experiments.py"],
    srcs_version = "PY3",
    deps = [
        ":contrib_matrix_builders",
        ":lagrange_terms",
        ":optimization",
        ":primal_optimization",
        "//multi_epoch_dp_matrix_factorization:loops",
        "//multi_epoch_dp_matrix_factorization:matrix_constructors",
        "//multi_epoch_dp_matrix_factorization:matrix_io",
    ],
)

py_test(
    name = "primal_experiments_test",
    srcs = ["primal_experiments_test.py"],
    srcs_version = "PY3",
    deps = [
        ":primal_experiments_lib",
        "//multi_epoch_dp_matrix_factorization:matrix_io",
    ],
)

py_library(
    name = "optimization",
    srcs = ["optimization.py"],
    srcs_version = "PY3",
    deps = [
        ":lagrange_terms",
        "//multi_epoch_dp_matrix_factorization:loops",
        "//third_party/py/jax:experimental",
        "//third_party/py/optax",
    ],
)

py_test(
    name = "optimization_test",
    srcs = ["optimization_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":contrib_matrix_builders",
        ":lagrange_terms",
        ":optimization",
        "//third_party/py/optax",
    ],
)

py_library(
    name = "lt_initializers",
    srcs = ["lt_initializers.py"],
    srcs_version = "PY3",
    deps = [
        ":contrib_matrix_builders",
        ":lagrange_terms",
    ],
)

py_library(
    name = "lagrange_terms",
    srcs = ["lagrange_terms.py"],
    srcs_version = "PY3",
    deps = ["//third_party/py/flax:core"],
)

py_test(
    name = "lagrange_terms_test",
    srcs = ["lagrange_terms_test.py"],
    srcs_version = "PY3",
    deps = [":lagrange_terms"],
)

py_library(
    name = "contrib_matrix_builders",
    srcs = ["contrib_matrix_builders.py"],
    srcs_version = "PY3",
)

py_test(
    name = "contrib_matrix_builders_test",
    srcs = ["contrib_matrix_builders_test.py"],
    srcs_version = "PY3",
    deps = [":contrib_matrix_builders"],
)

py_binary(
    name = "factorize_multi_epoch_prefix_sum",
    srcs = ["factorize_multi_epoch_prefix_sum.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":factorize_multi_epoch_prefix_sum_lib"],
)

py_library(
    name = "factorize_multi_epoch_prefix_sum_lib",
    srcs = ["factorize_multi_epoch_prefix_sum.py"],
    srcs_version = "PY3",
    deps = [
        ":contrib_matrix_builders",
        ":lagrange_terms",
        ":lt_initializers",
        ":optimization",
        "//multi_epoch_dp_matrix_factorization:matrix_constructors",
        "//multi_epoch_dp_matrix_factorization:matrix_io",
        "//utils:training_utils",
        "//third_party/py/optax",
    ],
)

py_test(
    name = "factorize_multi_epoch_prefix_sum_test",
    srcs = ["factorize_multi_epoch_prefix_sum_test.py"],
    srcs_version = "PY3",
    deps = [
        ":factorize_multi_epoch_prefix_sum_lib",
        "//multi_epoch_dp_matrix_factorization:matrix_io",
    ],
)
