load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        "//multi_epoch_dp_matrix_factorization:dp_matrix_factorization_packages",
        "//multi_epoch_dp_matrix_factorization:dp_matrix_factorization_users",
    ],
)

licenses(["notice"])

py_library(
    name = "stackoverflow_word_prediction",
    srcs = ["stackoverflow_word_prediction.py"],
    srcs_version = "PY3",
)

py_test(
    name = "stackoverflow_word_prediction_test",
    size = "medium",
    srcs = ["stackoverflow_word_prediction_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":stackoverflow_word_prediction"],
)
