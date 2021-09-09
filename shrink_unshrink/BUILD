load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

py_binary(
    name = "emnist_fedavg_main",
    srcs = ["emnist_fedavg_main.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":simple_fedavg_tf",
        ":simple_fedavg_tff",
    ],
)

py_binary(
    name = "trainer",
    srcs = ["trainer.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":models",
        ":shrink_unshrink_tff",
        ":simple_fedavg_tf",
        ":simple_fedavg_tff",
        "//utils:task_utils",
        "//utils:training_utils",
        "//utils:utils_impl",
        "//utils/optimizers:optimizer_utils",
    ],
)

py_library(
    name = "simple_fedavg_tf",
    srcs = ["simple_fedavg_tf.py"],
    srcs_version = "PY3",
)

py_library(
    name = "simple_fedavg_tff",
    srcs = ["simple_fedavg_tff.py"],
    srcs_version = "PY3",
    deps = [
        ":shrink_unshrink_tff",
        ":simple_fedavg_tf",
    ],
)

py_library(
    name = "shrink_unshrink_tff",
    srcs = ["shrink_unshrink_tff.py"],
    srcs_version = "PY3",
    deps = [":simple_fedavg_tf"],
)

py_library(
    name = "models",
    srcs = ["models.py"],
    srcs_version = "PY3",
)

py_test(
    name = "simple_fedavg_test",
    size = "medium",
    srcs = ["simple_fedavg_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":simple_fedavg_tf",
        ":simple_fedavg_tff",
    ],
)

py_test(
    name = "simple_fedavg_tf_test",
    srcs = ["simple_fedavg_tf_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = ["nokokoro"],
    deps = [
        ":models",
        ":simple_fedavg_tf",
        ":simple_fedavg_tff",
    ],
)

py_test(
    name = "shrink_unshrink_tff_test",
    srcs = ["shrink_unshrink_tff_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":models",
        ":shrink_unshrink_tff",
        ":simple_fedavg_tf",
        ":simple_fedavg_tff",
    ],
)
