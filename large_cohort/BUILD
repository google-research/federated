load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "aggregation",
    srcs = ["aggregation.py"],
    srcs_version = "PY3",
)

py_test(
    name = "aggregation_test",
    size = "small",
    srcs = ["aggregation_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":aggregation"],
)

py_binary(
    name = "trainer",
    srcs = ["trainer.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":trainer_lib"],
)

py_library(
    name = "trainer_lib",
    srcs = ["trainer.py"],
    srcs_version = "PY3",
    deps = [
        ":aggregation",
        ":data_utils",
        ":eval_utils",
        ":file_utils",
        ":flag_utils",
        ":simulation_specs",
        ":warmup",
        "//large_cohort/tasks:cifar100",
        "//large_cohort/tasks:emnist",
        "//large_cohort/tasks:emnist_autoencoder",
        "//large_cohort/tasks:emnist_lite",
        "//large_cohort/tasks:shakespeare",
        "//large_cohort/tasks:stackoverflow_tag_prediction",
        "//large_cohort/tasks:stackoverflow_word_prediction",
    ],
)

py_library(
    name = "data_utils",
    srcs = ["data_utils.py"],
    srcs_version = "PY3",
)

py_test(
    name = "data_utils_test",
    size = "small",
    srcs = ["data_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":data_utils"],
)

py_library(
    name = "eval_utils",
    srcs = ["eval_utils.py"],
    srcs_version = "PY3",
    deps = [":simulation_specs"],
)

py_test(
    name = "eval_utils_test",
    size = "small",
    srcs = ["eval_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":eval_utils",
        ":simulation_specs",
    ],
)

py_library(
    name = "file_utils",
    srcs = ["file_utils.py"],
    srcs_version = "PY3",
)

py_test(
    name = "file_utils_test",
    size = "small",
    srcs = ["file_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":file_utils"],
)

py_library(
    name = "flag_utils",
    srcs = ["flag_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//utils/optimizers:lars",
        "//utils/optimizers:shampoo",
        "//utils/optimizers:yogi",
    ],
)

py_test(
    name = "flag_utils_test",
    size = "small",
    srcs = ["flag_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":flag_utils"],
)

py_library(
    name = "simulation_specs",
    srcs = ["simulation_specs.py"],
    srcs_version = "PY3",
)

py_library(
    name = "warmup",
    srcs = ["warmup.py"],
    srcs_version = "PY3",
)

py_test(
    name = "warmup_test",
    size = "small",
    srcs = ["warmup_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":warmup"],
)
