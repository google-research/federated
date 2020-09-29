load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(default_visibility = ["//gans/experiments/emnist:__subpackages__"])

licenses(["notice"])

py_library(
    name = "emnist_data_utils",
    srcs = ["emnist_data_utils.py"],
    srcs_version = "PY3",
)

py_test(
    name = "emnist_data_utils_test",
    srcs = ["emnist_data_utils_test.py"],
    local = True,
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "manual",
        "nokokoro",
        "notap",
    ],
    deps = [":emnist_data_utils"],
)

py_binary(
    name = "train",
    srcs = ["train.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":emnist_data_utils",
        "//gans:gan_losses",
        "//gans:gan_training_tf_fns",
        "//gans:tff_gans",
        "//gans:training_loops",
        "//gans/experiments/emnist/classifier:emnist_classifier_model",
        "//gans/experiments/emnist/eval:emnist_eval_util",
        "//gans/experiments/emnist/models:convolutional_gan_networks",
        "//gans/experiments/emnist/preprocessing:filtered_emnist_data_utils",
        "//utils:utils_impl",
    ],
)

py_binary(
    name = "run_experiments",
    srcs = ["run_experiments.py"],
    data = [":train"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = ["//utils:utils_impl"],
)
