load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(default_visibility = ["//personalization_benchmark:__subpackages__"])

licenses(["notice"])

py_binary(
    name = "finetuning_trainer",
    srcs = ["finetuning_trainer.py"],
    deps = [":finetuning_trainer_lib"],
)

py_binary(
    name = "hypcluster_trainer",
    srcs = ["hypcluster_trainer.py"],
    deps = [":hypcluster_trainer_lib"],
)

py_library(
    name = "hypcluster_trainer_lib",
    srcs = ["hypcluster_trainer.py"],
    deps = [
        ":constants",
        "//personalization_benchmark/cross_device/algorithms:hypcluster_eval",
        "//personalization_benchmark/cross_device/algorithms:hypcluster_train",
        "//personalization_benchmark/cross_device/algorithms:optimizer_flag_utils",
        "//personalization_benchmark/cross_device/datasets:emnist",
        "//personalization_benchmark/cross_device/datasets:landmark",
        "//personalization_benchmark/cross_device/datasets:stackoverflow",
        "//personalization_benchmark/cross_device/datasets:ted_multi",
        "//utils:training_utils",
        "//utils:utils_impl",
    ],
)

py_library(
    name = "finetuning_trainer_lib",
    srcs = ["finetuning_trainer.py"],
    deps = [
        ":constants",
        "//personalization_benchmark/cross_device/algorithms:finetuning_utils",
        "//personalization_benchmark/cross_device/datasets:emnist",
        "//personalization_benchmark/cross_device/datasets:landmark",
        "//personalization_benchmark/cross_device/datasets:stackoverflow",
        "//personalization_benchmark/cross_device/datasets:ted_multi",
        "//utils:training_utils",
        "//utils:utils_impl",
        "//utils/optimizers:optimizer_utils",
    ],
)

py_library(
    name = "constants",
    srcs = ["constants.py"],
)

py_test(
    name = "finetuning_trainer_test",
    size = "large",
    srcs = ["finetuning_trainer_test.py"],
    deps = [":finetuning_trainer_lib"],
)

py_test(
    name = "hypcluster_trainer_test",
    size = "large",
    srcs = ["hypcluster_trainer_test.py"],
    deps = [":hypcluster_trainer_lib"],
)
