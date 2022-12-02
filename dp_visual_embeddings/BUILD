load("@rules_python//python:defs.bzl", "py_library", "py_test")

licenses(["notice"])

package(default_visibility = ["//dp_visual_embeddings:__subpackages__"])

py_library(
    name = "tff_input",
    srcs = ["tff_input.py"],
)

py_test(
    name = "tff_input_test",
    srcs = ["tff_input_test.py"],
    deps = [":tff_input"],
)
