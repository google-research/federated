load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

licenses(["notice"])

py_binary(
    name = "global_trainer",
    srcs = ["global_trainer.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":global_trainer_lib"],
)

py_library(
    name = "global_trainer_lib",
    srcs = ["global_trainer.py"],
    srcs_version = "PY3",
    deps = [
        ":hypcluster_eval",
        ":hypcluster_train",
        ":optimizer_flag_utils",
        "//data_poor_fl/pseudo_client_tasks:emnist_pseudo_client",
        "//utils:task_utils",
        "//utils:training_utils",
        "//utils:utils_impl",
    ],
)

py_test(
    name = "global_trainer_test",
    srcs = ["global_trainer_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":global_trainer_lib"],
)

py_library(
    name = "hypcluster_eval",
    srcs = ["hypcluster_eval.py"],
    deps = [":hypcluster_utils"],
)

py_test(
    name = "hypcluster_eval_test",
    srcs = ["hypcluster_eval_test.py"],
    deps = [":hypcluster_eval"],
)

py_library(
    name = "hypcluster_train",
    srcs = ["hypcluster_train.py"],
    deps = [":hypcluster_utils"],
)

py_test(
    name = "hypcluster_train_test",
    srcs = ["hypcluster_train_test.py"],
    deps = [":hypcluster_train"],
)

py_binary(
    name = "hypcluster_trainer",
    srcs = ["hypcluster_trainer.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":hypcluster_trainer_lib",
    ],
)

py_library(
    name = "hypcluster_trainer_lib",
    srcs = ["hypcluster_trainer.py"],
    srcs_version = "PY3",
    deps = [
        ":hypcluster_eval",
        ":hypcluster_train",
        ":optimizer_flag_utils",
        ":personalization_utils",
        "//data_poor_fl/pseudo_client_tasks:emnist_pseudo_client",
        "//utils:training_utils",
        "//utils:utils_impl",
    ],
)

py_test(
    name = "hypcluster_trainer_test",
    srcs = ["hypcluster_trainer_test.py"],
    deps = [":hypcluster_trainer_lib"],
)

py_library(
    name = "hypcluster_utils",
    srcs = ["hypcluster_utils.py"],
)

py_test(
    name = "hypcluster_utils_test",
    srcs = ["hypcluster_utils_test.py"],
    deps = [":hypcluster_utils"],
)

py_library(
    name = "optimizer_flag_utils",
    srcs = ["optimizer_flag_utils.py"],
    srcs_version = "PY3",
)

py_test(
    name = "optimizer_flag_utils_test",
    srcs = ["optimizer_flag_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":optimizer_flag_utils"],
)

py_binary(
    name = "personalization_trainer",
    srcs = ["personalization_trainer.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":personalization_trainer_lib",
    ],
)

py_library(
    name = "personalization_trainer_lib",
    srcs = ["personalization_trainer.py"],
    srcs_version = "PY3",
    deps = [
        ":personalization_utils",
        "//data_poor_fl/pseudo_client_tasks:emnist_pseudo_client",
        "//utils:training_utils",
        "//utils:utils_impl",
        "//utils/optimizers:optimizer_utils",
    ],
)

py_test(
    name = "personalization_trainer_test",
    srcs = ["personalization_trainer_test.py"],
    deps = [":personalization_trainer_lib"],
)

py_library(
    name = "personalization_utils",
    srcs = ["personalization_utils.py"],
)

py_test(
    name = "personalization_utils_test",
    srcs = ["personalization_utils_test.py"],
    deps = [":personalization_utils"],
)
