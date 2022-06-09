load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//personalization_benchmark:__subpackages__"])

licenses(["notice"])

py_library(
    name = "finetuning_utils",
    srcs = ["finetuning_utils.py"],
)

py_library(
    name = "hypcluster_train",
    srcs = ["hypcluster_train.py"],
    deps = [":hypcluster_utils"],
)

py_library(
    name = "hypcluster_eval",
    srcs = ["hypcluster_eval.py"],
    deps = [
        ":hypcluster_utils",
        "//personalization_benchmark/cross_device:constants",
    ],
)

py_library(
    name = "hypcluster_utils",
    srcs = ["hypcluster_utils.py"],
)

py_library(
    name = "checkpoint_utils",
    srcs = ["checkpoint_utils.py"],
)

py_library(
    name = "optimizer_flag_utils",
    srcs = ["optimizer_flag_utils.py"],
)

py_test(
    name = "finetuning_utils_test",
    srcs = ["finetuning_utils_test.py"],
    deps = [":finetuning_utils"],
)

py_test(
    name = "hypcluster_train_test",
    srcs = ["hypcluster_train_test.py"],
    deps = [":hypcluster_train"],
)

py_test(
    name = "hypcluster_utils_test",
    srcs = ["hypcluster_utils_test.py"],
    deps = [":hypcluster_utils"],
)

py_test(
    name = "hypcluster_eval_test",
    srcs = ["hypcluster_eval_test.py"],
    deps = [":hypcluster_eval"],
)

py_test(
    name = "checkpoint_utils_test",
    srcs = ["checkpoint_utils_test.py"],
    deps = [":checkpoint_utils"],
)

py_test(
    name = "optimizer_flag_utils_test",
    srcs = ["optimizer_flag_utils_test.py"],
    deps = [":optimizer_flag_utils"],
)
