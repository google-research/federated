load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(
    default_applicable_licenses = ["//third_party/federated_research:package_license"],
    default_visibility = ["//bandits:__subpackages__"],
)

licenses(["notice"])

py_library(
    name = "keras_optimizer_utils",
    srcs = ["keras_optimizer_utils.py"],
)

py_library(
    name = "bandits_utils",
    srcs = ["bandits_utils.py"],
)

py_library(
    name = "file_utils",
    srcs = ["file_utils.py"],
)

py_library(
    name = "bandits_process",
    srcs = ["bandits_process.py"],
    deps = [":bandits_utils"],
)

py_library(
    name = "trainer",
    srcs = ["trainer.py"],
    deps = [
        ":bandits_process",
        ":bandits_utils",
        "//bandits/algs:epsilon_greedy",
        "//bandits/algs:falcon",
        "//bandits/algs:softmax_sampling",
        "//bandits/tasks:emnist",
        "//bandits/tasks:stackoverflow",
        "//bandits/tasks:task_utils",
    ],
)

py_test(
    name = "file_utils_test",
    srcs = ["file_utils_test.py"],
    deps = [":file_utils"],
)

py_test(
    name = "bandits_utils_test",
    srcs = ["bandits_utils_test.py"],
    deps = [":bandits_utils"],
)

py_test(
    name = "keras_optimizer_utils_test",
    srcs = ["keras_optimizer_utils_test.py"],
    deps = [":keras_optimizer_utils"],
)

py_test(
    name = "bandits_process_test",
    srcs = ["bandits_process_test.py"],
    deps = [
        ":bandits_process",
        ":bandits_utils",
        "//bandits/algs:epsilon_greedy",
        "//bandits/tasks:emnist",
        "//bandits/tasks:task_utils",
    ],
)

py_test(
    name = "trainer_test",
    srcs = ["trainer_test.py"],
    deps = [
        ":bandits_utils",
        ":keras_optimizer_utils",
        ":trainer",
        "//bandits/tasks:task_utils",
    ],
)

py_binary(
    name = "run_federated",
    srcs = ["run_federated.py"],
    deps = [
        ":file_utils",
        ":keras_optimizer_utils",
        ":trainer",
    ],
)
