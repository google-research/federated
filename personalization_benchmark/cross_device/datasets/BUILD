load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(default_visibility = ["//personalization_benchmark:__subpackages__"])

licenses(["notice"])

py_library(
    name = "emnist",
    srcs = ["emnist.py"],
    deps = ["//personalization_benchmark/cross_device:constants"],
)

py_library(
    name = "stackoverflow",
    srcs = ["stackoverflow.py"],
    deps = [
        ":emnist",
        "//personalization_benchmark/cross_device:constants",
    ],
)

py_library(
    name = "landmark",
    srcs = ["landmark.py"],
    deps = [
        ":emnist",
        ":mobilenet_v2",
        "//personalization_benchmark/cross_device:constants",
    ],
)

py_library(
    name = "mobilenet_v2",
    srcs = ["mobilenet_v2.py"],
)

py_library(
    name = "transformer_models",
    srcs = ["transformer_models.py"],
    deps = ["//utils/models:stackoverflow_models"],
)

py_library(
    name = "ted_multi",
    srcs = ["ted_multi.py"],
    deps = [
        ":emnist",
        ":transformer_models",
        "//personalization_benchmark/cross_device:constants",
        "//utils:keras_metrics",
        "//utils/datasets:stackoverflow_word_prediction",
    ],
)

py_test(
    name = "emnist_test",
    srcs = ["emnist_test.py"],
    deps = [
        ":emnist",
        "//personalization_benchmark/cross_device:constants",
    ],
)

py_test(
    name = "stackoverflow_test",
    srcs = ["stackoverflow_test.py"],
    deps = [
        ":stackoverflow",
        "//personalization_benchmark/cross_device:constants",
    ],
)

py_test(
    name = "landmark_test",
    srcs = ["landmark_test.py"],
    deps = [
        ":landmark",
        "//personalization_benchmark/cross_device:constants",
    ],
)

py_test(
    name = "mobilenet_v2_test",
    srcs = ["mobilenet_v2_test.py"],
    deps = [":mobilenet_v2"],
)

py_test(
    name = "transformer_models_test",
    srcs = ["transformer_models_test.py"],
    deps = [":transformer_models"],
)

py_test(
    name = "ted_multi_test",
    srcs = ["ted_multi_test.py"],
    deps = [
        ":ted_multi",
        "//personalization_benchmark/cross_device:constants",
    ],
)

py_binary(
    name = "write_ted_multi_to_sql_client_data",
    srcs = ["write_ted_multi_to_sql_client_data.py"],
)
