load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_visibility = ["//compressed_communication:__subpackages__"],
)

licenses(["notice"])

py_library(
    name = "cross_entropy",
    srcs = ["cross_entropy.py"],
)

py_test(
    name = "cross_entropy_test",
    srcs = ["cross_entropy_test.py"],
    deps = [":cross_entropy"],
)

py_library(
    name = "elias_gamma_encode",
    srcs = ["elias_gamma_encode.py"],
)

py_test(
    name = "elias_gamma_encode_test",
    srcs = ["elias_gamma_encode_test.py"],
    deps = [":elias_gamma_encode"],
)

py_library(
    name = "entropy",
    srcs = ["entropy.py"],
    deps = [":cross_entropy"],
)

py_test(
    name = "entropy_test",
    srcs = ["entropy_test.py"],
    deps = [":entropy"],
)

py_library(
    name = "group",
    srcs = ["group.py"],
)

py_test(
    name = "group_test",
    srcs = ["group_test.py"],
    deps = [":group"],
)

py_library(
    name = "histogram_weights",
    srcs = ["histogram_weights.py"],
)

py_test(
    name = "histogram_weights_test",
    srcs = ["histogram_weights_test.py"],
    python_version = "PY3",
    deps = [":histogram_weights"],
)

py_library(
    name = "min_max_mean_weights",
    srcs = ["min_max_mean_weights.py"],
)

py_test(
    name = "min_max_mean_weights_test",
    srcs = ["min_max_mean_weights_test.py"],
    python_version = "PY3",
    deps = [":min_max_mean_weights"],
)

py_library(
    name = "quantize",
    srcs = ["quantize.py"],
    deps = ["//compressed_communication/aggregators/utils:quantize_utils"],
)

py_test(
    name = "quantize_test",
    srcs = ["quantize_test.py"],
    deps = [":quantize"],
)

py_library(
    name = "quantize_encode",
    srcs = ["quantize_encode.py"],
    deps = [
        ":elias_gamma_encode",
        "//compressed_communication/aggregators/utils:quantize_utils",
    ],
)

py_library(
    name = "quantize_encode_client_lambda",
    srcs = ["quantize_encode_client_lambda.py"],
    deps = [
        ":elias_gamma_encode",
        "//compressed_communication/aggregators/utils:quantize_utils",
    ],
)

py_test(
    name = "quantize_encode_client_lambda_test",
    srcs = ["quantize_encode_client_lambda_test.py"],
    deps = [":quantize_encode_client_lambda"],
)

py_test(
    name = "quantize_encode_test",
    srcs = ["quantize_encode_test.py"],
    deps = [":quantize_encode"],
)

py_library(
    name = "stdev_weights",
    srcs = ["stdev_weights.py"],
)

py_test(
    name = "stdev_weights_test",
    srcs = ["stdev_weights_test.py"],
    deps = [":stdev_weights"],
)

py_library(
    name = "stochastic_quantize",
    srcs = ["stochastic_quantize.py"],
    deps = ["//compressed_communication/aggregators/utils:quantize_utils"],
)

py_test(
    name = "stochastic_quantize_test",
    srcs = ["stochastic_quantize_test.py"],
    deps = [":stochastic_quantize"],
)

py_library(
    name = "zero_run_length",
    srcs = ["zero_run_length.py"],
    deps = [
        ":cross_entropy",
        ":entropy",
    ],
)

py_test(
    name = "zero_run_length_test",
    srcs = ["zero_run_length_test.py"],
    deps = [":zero_run_length"],
)
