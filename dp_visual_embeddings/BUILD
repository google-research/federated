load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

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

py_library(
    name = "centralized_training_loop",
    srcs = ["centralized_training_loop.py"],
    deps = [
        "//dp_visual_embeddings/tasks:task_utils",
        "//dp_visual_embeddings/utils:export",
        "//dp_visual_embeddings/utils:file_utils",
    ],
)

py_library(
    name = "trainer",
    srcs = ["trainer.py"],
    deps = [
        "//dp_visual_embeddings/algorithms:federated_partial",
        "//dp_visual_embeddings/algorithms:process_with_pretrained",
        "//dp_visual_embeddings/models:embedding_model",
        "//dp_visual_embeddings/models:keras_utils",
        "//dp_visual_embeddings/tasks:task_utils",
        "//dp_visual_embeddings/utils:export",
    ],
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

py_test(
    name = "centralized_training_loop_test",
    srcs = ["centralized_training_loop_test.py"],
    deps = [":centralized_training_loop"],
)

py_test(
    name = "trainer_test",
    size = "large",
    srcs = ["trainer_test.py"],
    shard_count = 3,
    tags = [
        "notap",
        "requires-net:external",
    ],
    deps = [
        ":trainer",
        "//dp_visual_embeddings/models:build_model",
        "//dp_visual_embeddings/tasks:emnist_task",
    ],
)

py_binary(
    name = "run_centralized",
    srcs = ["run_centralized.py"],
    deps = [
        ":centralized_training_loop",
        "//dp_visual_embeddings/models:build_model",
        "//dp_visual_embeddings/tasks:build_task",
    ],
)

py_binary(
    name = "run_federated",
    srcs = ["run_federated.py"],
    deps = [
        ":trainer",
        "//dp_visual_embeddings/models:build_model",
        "//dp_visual_embeddings/tasks:build_task",
        "//dp_visual_embeddings/utils:file_utils",
    ],
)
