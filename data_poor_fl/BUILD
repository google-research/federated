load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

licenses(["notice"])

py_binary(
    name = "global_trainer",
    srcs = ["global_trainer.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":global_trainer_lib"],
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
    data = [
        ":emnist_train_num_examples.csv",
    ],
    srcs_version = "PY3",
    deps = [
        ":personalization_utils",
        ":pseudo_client_data",
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
    name = "global_trainer_lib",
    srcs = ["global_trainer.py"],
    data = [
        "//data_poor_fl:emnist_train_num_examples.csv",
    ],
    srcs_version = "PY3",
    deps = [
        ":optimizer_flag_utils",
        ":pseudo_client_data",
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

py_library(
    name = "pseudo_client_data",
    srcs = ["pseudo_client_data.py"],
    srcs_version = "PY3",
)

py_library(
    name = "personalization_utils",
    srcs = ["personalization_utils.py"],
)

py_test(
    name = "pseudo_client_data_test",
    srcs = ["pseudo_client_data_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":pseudo_client_data"],
)

py_test(
    name = "personalization_utils_test",
    srcs = ["personalization_utils_test.py"],
    deps = [":personalization_utils"],
)
