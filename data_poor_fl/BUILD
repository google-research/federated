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
    data = [
        "//data_poor_fl:emnist_train_num_examples.csv",
    ],
    srcs_version = "PY3",
    deps = [
        ":pseudo_client_data",
        "//utils:training_utils",
        "//utils:utils_impl",
        "//utils/optimizers:optimizer_utils",
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
    name = "pseudo_client_data",
    srcs = ["pseudo_client_data.py"],
    srcs_version = "PY3",
)

py_test(
    name = "pseudo_client_data_test",
    srcs = ["pseudo_client_data_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":pseudo_client_data"],
)
