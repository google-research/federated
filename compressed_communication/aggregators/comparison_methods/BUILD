load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_visibility = ["//compressed_communication:__subpackages__"],
)

licenses(["notice"])

py_library(
    name = "drive",
    srcs = ["drive.py"],
)

py_test(
    name = "drive_test",
    srcs = ["drive_test.py"],
    deps = [":drive"],
)

py_library(
    name = "one_bit_sgd",
    srcs = ["one_bit_sgd.py"],
)

py_test(
    name = "one_bit_sgd_test",
    srcs = ["one_bit_sgd_test.py"],
    deps = [":one_bit_sgd"],
)

py_library(
    name = "qsgd",
    srcs = ["qsgd.py"],
    deps = ["//compressed_communication/aggregators/utils:quantize_utils"],
)

py_test(
    name = "qsgd_test",
    srcs = ["qsgd_test.py"],
    deps = [":qsgd"],
)

py_library(
    name = "terngrad",
    srcs = ["terngrad.py"],
)

py_test(
    name = "terngrad_test",
    srcs = ["terngrad_test.py"],
    deps = [":terngrad"],
)

py_library(
    name = "three_lc",
    srcs = ["three_lc.py"],
    deps = ["//compressed_communication/aggregators/utils:quantize_utils"],
)

py_test(
    name = "three_lc_test",
    srcs = ["three_lc_test.py"],
    deps = [":three_lc"],
)

py_library(
    name = "top_k",
    srcs = ["top_k.py"],
)

py_test(
    name = "top_k_test",
    srcs = ["top_k_test.py"],
    deps = [":top_k"],
)
