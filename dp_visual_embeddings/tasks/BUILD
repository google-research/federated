load("@rules_python//python:defs.bzl", "py_library", "py_test")

licenses(["notice"])

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//dp_visual_embeddings:__subpackages__"],
)

py_library(
    name = "emnist_task",
    srcs = ["emnist_task.py"],
    deps = [
        ":task_data",
        ":task_utils",
        "//dp_visual_embeddings:tff_input",
        "//dp_visual_embeddings/models:build_model",
        "//dp_visual_embeddings/models:keras_utils",
    ],
)

py_library(
    name = "inaturalist_task",
    srcs = ["inaturalist_task.py"],
    deps = [
        ":task_data",
        ":task_utils",
        "//dp_visual_embeddings:tff_input",
        "//dp_visual_embeddings/models:build_model",
        "//dp_visual_embeddings/models:keras_utils",
    ],
)

py_library(
    name = "build_task",
    srcs = ["build_task.py"],
    deps = [
        ":emnist_task",
        ":inaturalist_task",
        ":landmark_task",
        ":task_utils",
        "//dp_visual_embeddings/models:build_model",
    ],
)

py_library(
    name = "task_data",
    srcs = ["task_data.py"],
)

py_library(
    name = "task_utils",
    srcs = ["task_utils.py"],
    deps = [
        ":task_data",
        "//dp_visual_embeddings:losses",
        "//dp_visual_embeddings:metrics",
        "//dp_visual_embeddings/models:embedding_model",
        "//dp_visual_embeddings/models:keras_utils",
        "//dp_visual_embeddings/utils:exporting_program_state_manager",
    ],
)

py_library(
    name = "landmark_task",
    srcs = ["landmark_task.py"],
    deps = [
        ":task_data",
        ":task_utils",
        "//dp_visual_embeddings:tff_input",
        "//dp_visual_embeddings/models:build_model",
        "//dp_visual_embeddings/models:keras_utils",
    ],
)

py_test(
    name = "landmark_task_test",
    size = "large",
    srcs = ["landmark_task_test.py"],
    shard_count = 3,
    tags = [
        "notap",
        "requires-net:external",
    ],
    deps = [
        ":landmark_task",
        "//dp_visual_embeddings:metrics",
    ],
)

py_test(
    name = "emnist_task_test",
    size = "large",
    srcs = ["emnist_task_test.py"],
    shard_count = 3,
    tags = [
        "notap",
        "requires-net:external",
    ],
    deps = [
        ":emnist_task",
        "//dp_visual_embeddings:metrics",
    ],
)

py_test(
    name = "build_task_test",
    size = "large",
    srcs = ["build_task_test.py"],
    tags = [
        "notap",
        "requires-net:external",
    ],
    deps = [
        ":build_task",
        ":task_utils",
    ],
)

py_test(
    name = "task_utils_test",
    srcs = ["task_utils_test.py"],
    deps = [
        ":task_data",
        ":task_utils",
        "//dp_visual_embeddings/models:keras_utils",
    ],
)

py_test(
    name = "task_data_test",
    srcs = ["task_data_test.py"],
    deps = [":task_data"],
)
