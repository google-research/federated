load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

licenses(["notice"])

py_library(
    name = "coordinate_aggregators",
    srcs = ["coordinate_aggregators.py"],
)

py_test(
    name = "coordinate_aggregators_test",
    srcs = ["coordinate_aggregators_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":coordinate_aggregators"],
)

py_library(
    name = "coordinate_finalizers",
    srcs = ["coordinate_finalizers.py"],
)

py_test(
    name = "coordinate_finalizers_test",
    srcs = ["coordinate_finalizers_test.py"],
    deps = [":coordinate_finalizers"],
)

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
    name = "hypcluster",
    srcs = ["hypcluster.py"],
    deps = [
        ":coordinate_aggregators",
        ":coordinate_finalizers",
    ],
)

py_test(
    name = "hypcluster_test",
    srcs = ["hypcluster_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":hypcluster"],
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
    data = [
        ":emnist_train_num_examples.csv",
    ],
    srcs_version = "PY3",
    deps = [
        ":hypcluster",
        ":optimizer_flag_utils",
        ":personalization_utils",
        ":pseudo_client_data",
        "//utils:training_utils",
        "//utils:utils_impl",
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
    name = "personalization_utils",
    srcs = ["personalization_utils.py"],
)

py_test(
    name = "personalization_utils_test",
    srcs = ["personalization_utils_test.py"],
    deps = [":personalization_utils"],
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

py_test(
    name = "hypcluster_trainer_test",
    srcs = ["hypcluster_trainer_test.py"],
    deps = [":hypcluster_trainer_lib"],
)
