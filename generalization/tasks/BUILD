load("@rules_python//python:defs.bzl", "py_library", "py_test")

licenses(["notice"])

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        "//generalization:__subpackages__",
    ],
)

py_library(
    name = "training_specs",
    srcs = ["training_specs.py"],
    srcs_version = "PY3",
)

py_library(
    name = "stackoverflow_word",
    srcs = ["stackoverflow_word.py"],
    srcs_version = "PY3",
    deps = [
        ":training_specs",
        "//generalization/utils:client_data_utils",
        "//generalization/utils:eval_metric_distribution",
        "//generalization/utils:trainer_utils",
        "//utils:keras_metrics",
        "//utils:utils_impl",
        "//utils/datasets:stackoverflow_word_prediction",
        "//utils/models:stackoverflow_models",
    ],
)

py_test(
    name = "stackoverflow_word_test",
    srcs = ["stackoverflow_word_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "requires-net:external",
    ],
    deps = [":stackoverflow_word"],
)

py_test(
    name = "emnist_character_test",
    srcs = ["emnist_character_test.py"],
    python_version = "PY3",
    shard_count = 5,
    srcs_version = "PY3",
    tags = [
        "requires-net:external",
    ],
    deps = [":emnist_character"],
)

py_library(
    name = "cifar100_image",
    srcs = ["cifar100_image.py"],
    srcs_version = "PY3",
    deps = [
        ":training_specs",
        "//generalization/utils:client_data_utils",
        "//generalization/utils:eval_metric_distribution",
        "//generalization/utils:resnet_models",
        "//generalization/utils:sql_client_data_utils",
        "//generalization/utils:trainer_utils",
        "//utils:utils_impl",
        "//utils/datasets:cifar100_dataset",
    ],
)

py_library(
    name = "emnist_character",
    srcs = ["emnist_character.py"],
    srcs_version = "PY3",
    deps = [
        ":training_specs",
        "//generalization/utils:client_data_utils",
        "//generalization/utils:eval_metric_distribution",
        "//generalization/utils:resnet_models",
        "//generalization/utils:sql_client_data_utils",
        "//generalization/utils:trainer_utils",
        "//utils:utils_impl",
        "//utils/datasets:emnist_dataset",
    ],
)

py_library(
    name = "shakespeare_character",
    srcs = ["shakespeare_character.py"],
    deps = [
        ":training_specs",
        "//generalization/utils:client_data_utils",
        "//generalization/utils:eval_metric_distribution",
        "//generalization/utils:trainer_utils",
        "//utils:keras_metrics",
        "//utils:utils_impl",
        "//utils/datasets:shakespeare_dataset",
        "//utils/models:shakespeare_models",
    ],
)
