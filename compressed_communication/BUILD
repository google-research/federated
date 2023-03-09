load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])

py_library(
    name = "builder",
    srcs = ["builder.py"],
    srcs_version = "PY3",
    deps = [
        ":builder_configs",
        "//compressed_communication/aggregators:entropy",
        "//compressed_communication/aggregators:group",
        "//compressed_communication/aggregators:histogram_weights",
        "//compressed_communication/aggregators:quantize",
        "//compressed_communication/aggregators:quantize_encode",
        "//compressed_communication/aggregators:quantize_encode_client_lambda",
        "//compressed_communication/aggregators/comparison_methods:drive",
        "//compressed_communication/aggregators/comparison_methods:one_bit_sgd",
        "//compressed_communication/aggregators/comparison_methods:qsgd",
        "//compressed_communication/aggregators/comparison_methods:terngrad",
        "//compressed_communication/aggregators/comparison_methods:three_lc",
        "//compressed_communication/aggregators/comparison_methods:top_k",
    ],
)

py_library(
    name = "builder_configs",
    srcs = ["builder_configs.py"],
)

py_test(
    name = "builder_test",
    srcs = ["builder_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":builder",
        "//compressed_communication/aggregators:entropy",
        "//compressed_communication/aggregators:group",
        "//compressed_communication/aggregators:histogram_weights",
        "//compressed_communication/aggregators:quantize",
        "//compressed_communication/aggregators:quantize_encode",
        "//compressed_communication/aggregators:quantize_encode_client_lambda",
        "//compressed_communication/aggregators/comparison_methods:drive",
        "//compressed_communication/aggregators/comparison_methods:one_bit_sgd",
        "//compressed_communication/aggregators/comparison_methods:qsgd",
        "//compressed_communication/aggregators/comparison_methods:terngrad",
        "//compressed_communication/aggregators/comparison_methods:three_lc",
        "//compressed_communication/aggregators/comparison_methods:top_k",
    ],
)

py_library(
    name = "configs",
    srcs = ["configs.py"],
    srcs_version = "PY3",
)

py_binary(
    name = "trainer",
    srcs = ["trainer.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":trainer_lib"],
)

py_library(
    name = "trainer_lib",
    srcs = ["trainer.py"],
    srcs_version = "PY3",
    deps = [
        ":builder",
        ":configs",
        "//utils:task_utils",
        "//utils:training_utils",
        "//utils:utils_impl",
        "//utils/optimizers:optimizer_utils",
    ],
)
