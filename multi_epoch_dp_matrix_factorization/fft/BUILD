# Package exploring FFT noise pre-generated noise in Matrix Factorization DPFTRL.

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
    name = "generate_noise",
    srcs = ["generate_noise.py"],
)

py_test(
    name = "generate_noise_test",
    srcs = ["generate_noise_test.py"],
    deps = [":generate_noise"],
)
