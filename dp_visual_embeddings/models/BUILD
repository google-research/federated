load("@rules_python//python:defs.bzl", "py_library", "py_test")

licenses(["notice"])

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//dp_visual_embeddings:__subpackages__"],
)

py_library(
    name = "build_model",
    srcs = ["build_model.py"],
    deps = [
        ":keras_lenet",
        ":keras_mobilenet_v2",
        ":keras_resnet",
        ":keras_utils",
    ],
)

py_library(
    name = "keras_mobilenet_v2",
    srcs = ["keras_mobilenet_v2.py"],
    deps = [":keras_utils"],
)

py_library(
    name = "keras_resnet",
    srcs = ["keras_resnet.py"],
    deps = [":keras_utils"],
)

py_library(
    name = "keras_utils",
    srcs = ["keras_utils.py"],
    deps = [":embedding_model"],
)

py_library(
    name = "keras_lenet",
    srcs = ["keras_lenet.py"],
    deps = [":keras_utils"],
)

py_library(
    name = "embedding_model",
    srcs = ["embedding_model.py"],
)

py_test(
    name = "build_model_test",
    srcs = ["build_model_test.py"],
    deps = [
        ":build_model",
        ":keras_utils",
    ],
)

py_test(
    name = "keras_utils_test",
    srcs = ["keras_utils_test.py"],
    deps = [
        ":embedding_model",
        ":keras_utils",
    ],
)

py_test(
    name = "keras_mobilenet_v2_test",
    srcs = ["keras_mobilenet_v2_test.py"],
    shard_count = 2,
    deps = [":keras_mobilenet_v2"],
)

py_test(
    name = "keras_resnet_test",
    srcs = ["keras_resnet_test.py"],
    deps = [
        ":keras_resnet",
        ":keras_utils",
    ],
)

py_test(
    name = "model_size_test",
    srcs = ["model_size_test.py"],
    tags = [
        "manual",
        "notap",
    ],
    deps = [
        ":keras_mobilenet_v2",
        ":keras_resnet",
    ],
)

py_test(
    name = "keras_lenet_test",
    srcs = ["keras_lenet_test.py"],
    deps = [
        ":keras_lenet",
        ":keras_utils",
    ],
)
