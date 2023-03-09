load("@rules_python//python:defs.bzl", "py_library", "py_test")

licenses(["notice"])

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//dp_visual_embeddings:__subpackages__"],
)

py_library(
    name = "federated_partial",
    srcs = ["federated_partial.py"],
    deps = [
        "//dp_visual_embeddings/models:embedding_model",
        "//dp_visual_embeddings/models:keras_utils",
    ],
)

py_library(
    name = "process_with_pretrained",
    srcs = ["process_with_pretrained.py"],
    deps = ["//dp_visual_embeddings/models:keras_utils"],
)

py_test(
    name = "federated_partial_test",
    srcs = ["federated_partial_test.py"],
    deps = [
        ":federated_partial",
        "//dp_visual_embeddings/models:keras_utils",
    ],
)

py_test(
    name = "process_with_pretrained_test",
    srcs = ["process_with_pretrained_test.py"],
    deps = [
        ":federated_partial",
        ":process_with_pretrained",
        "//dp_visual_embeddings/models:keras_utils",
    ],
)
