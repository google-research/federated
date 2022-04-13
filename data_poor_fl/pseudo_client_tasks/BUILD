load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//data_poor_fl:__subpackages__"])

licenses(["notice"])

py_library(
    name = "emnist_pseudo_client",
    srcs = ["emnist_pseudo_client.py"],
    data = [
        ":emnist_test_num_examples.csv",
        ":emnist_train_num_examples.csv",
    ],
    srcs_version = "PY3",
    deps = [":pseudo_client_data"],
)

py_test(
    name = "emnist_pseudo_client_test",
    srcs = ["emnist_pseudo_client_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        "manual",
        "notap",
    ],
    deps = [":emnist_pseudo_client"],
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
