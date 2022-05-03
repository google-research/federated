load("@rules_python//python:defs.bzl", "py_library", "py_test")

licenses(["notice"])

package(default_visibility = [
    "//generalization:__subpackages__",
])

py_library(
    name = "centralized_training_loop",
    srcs = ["centralized_training_loop.py"],
    srcs_version = "PY3",
    deps = [":metric_utils"],
)

py_test(
    name = "centralized_training_loop_test",
    size = "large",
    srcs = ["centralized_training_loop_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":centralized_training_loop",
        ":metric_utils",
    ],
)

py_library(
    name = "client_data_utils",
    srcs = ["client_data_utils.py"],
    srcs_version = "PY3",
)

py_test(
    name = "client_data_utils_test",
    srcs = ["client_data_utils_test.py"],
    python_version = "PY3",
    shard_count = 10,
    srcs_version = "PY3",
    deps = [":client_data_utils"],
)

py_library(
    name = "eval_metric_distribution",
    srcs = ["eval_metric_distribution.py"],
)

py_test(
    name = "eval_metric_distribution_test",
    srcs = ["eval_metric_distribution_test.py"],
    deps = [":eval_metric_distribution"],
)

py_library(
    name = "federated_training_loop",
    srcs = ["federated_training_loop.py"],
    srcs_version = "PY3",
    deps = [":metric_utils"],
)

py_test(
    name = "federated_training_loop_test",
    srcs = ["federated_training_loop_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":federated_training_loop",
        ":metric_utils",
    ],
)

py_library(
    name = "fed_avg_schedule",
    srcs = ["fed_avg_schedule.py"],
    srcs_version = "PY3",
    deps = ["//utils:tensor_utils"],
)

py_test(
    name = "fed_avg_schedule_test",
    size = "large",
    srcs = ["fed_avg_schedule_test.py"],
    python_version = "PY3",
    shard_count = 10,
    srcs_version = "PY3",
    deps = [":fed_avg_schedule"],
)

py_library(
    name = "logging_utils",
    srcs = ["logging_utils.py"],
)

py_library(
    name = "metric_utils",
    srcs = ["metric_utils.py"],
    srcs_version = "PY3",
    deps = ["//utils:utils_impl"],
)

py_test(
    name = "metric_utils_test",
    srcs = ["metric_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":metric_utils"],
)

py_library(
    name = "resnet_models",
    srcs = ["resnet_models.py"],
)

py_test(
    name = "resnet_models_test",
    srcs = ["resnet_models_test.py"],
    shard_count = 3,
    deps = [":resnet_models"],
)

py_library(
    name = "sql_client_data_utils",
    srcs = ["sql_client_data_utils.py"],
    srcs_version = "PY3",
)

py_test(
    name = "sql_client_data_utils_test",
    srcs = ["sql_client_data_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":sql_client_data_utils"],
)

py_library(
    name = "tf_gaussian_mixture",
    srcs = ["tf_gaussian_mixture.py"],
    srcs_version = "PY3",
)

py_test(
    name = "tf_gaussian_mixture_test",
    srcs = ["tf_gaussian_mixture_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":tf_gaussian_mixture"],
)

py_library(
    name = "trainer_utils",
    srcs = ["trainer_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":client_data_utils",
        ":eval_metric_distribution",
        "//generalization/tasks:training_specs",
    ],
)

py_test(
    name = "trainer_utils_test",
    srcs = ["trainer_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":eval_metric_distribution",
        ":trainer_utils",
    ],
)
