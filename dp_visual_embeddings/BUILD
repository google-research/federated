load("@rules_python//python:defs.bzl", "py_library", "py_test")

licenses(["notice"])

package(default_visibility = ["//dp_visual_embeddings:__subpackages__"])

py_library(
    name = "tff_input",
    srcs = ["tff_input.py"],
)

py_library(
    name = "metrics",
    srcs = ["metrics.py"],
    deps = ["//dp_visual_embeddings/models:keras_utils"],
)

py_library(
    name = "losses",
    srcs = ["losses.py"],
)

py_test(
    name = "tff_input_test",
    srcs = ["tff_input_test.py"],
    deps = [":tff_input"],
)

py_test(
    name = "losses_test",
    srcs = ["losses_test.py"],
    deps = [":losses"],
)

py_test(
    name = "metrics_test",
    srcs = ["metrics_test.py"],
    deps = [":metrics"],
)
