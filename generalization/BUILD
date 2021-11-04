load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

licenses(["notice"])

py_library(
    name = "trainer_federated_lib",
    srcs = ["trainer_federated.py"],
    srcs_version = "PY3",
    deps = [
        "//generalization/tasks:cifar100_image",
        "//generalization/tasks:emnist_character",
        "//generalization/tasks:shakespeare_character",
        "//generalization/tasks:stackoverflow_word",
        "//generalization/tasks:training_specs",
        "//generalization/utils:fed_avg_schedule",
        "//generalization/utils:federated_training_loop",
        "//generalization/utils:metric_utils",
        "//utils:utils_impl",
        "//utils/optimizers:optimizer_utils",
    ],
)

py_binary(
    name = "trainer_federated",
    srcs = ["trainer_federated.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":trainer_federated_lib",
    ],
)

py_test(
    name = "trainer_federated_test",
    size = "large",
    srcs = ["trainer_federated_test.py"],
    python_version = "PY3",
    shard_count = 10,
    srcs_version = "PY3",
    tags = ["nokokoro"],
    deps = [
        "//generalization/tasks:cifar100_image",
        "//generalization/tasks:emnist_character",
        "//generalization/tasks:shakespeare_character",
        "//generalization/tasks:stackoverflow_word",
        "//generalization/tasks:training_specs",
        "//generalization/utils:federated_training_loop",
        "//generalization/utils:metric_utils",
    ],
)

py_library(
    name = "trainer_centralized_lib",
    srcs = ["trainer_centralized.py"],
    srcs_version = "PY3",
    deps = [
        "//generalization/tasks:cifar100_image",
        "//generalization/tasks:emnist_character",
        "//generalization/tasks:shakespeare_character",
        "//generalization/tasks:stackoverflow_word",
        "//generalization/tasks:training_specs",
        "//generalization/utils:centralized_training_loop",
        "//generalization/utils:metric_utils",
        "//utils:utils_impl",
        "//utils/optimizers:optimizer_utils",
    ],
)

py_binary(
    name = "trainer_centralized",
    srcs = ["trainer_centralized.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":trainer_centralized_lib",
    ],
)

py_test(
    name = "trainer_centralized_test",
    size = "large",
    srcs = ["trainer_centralized_test.py"],
    python_version = "PY3",
    shard_count = 10,
    srcs_version = "PY3",
    tags = ["nokokoro"],
    deps = [
        "//generalization/tasks:cifar100_image",
        "//generalization/tasks:emnist_character",
        "//generalization/tasks:shakespeare_character",
        "//generalization/tasks:stackoverflow_word",
        "//generalization/tasks:training_specs",
        "//generalization/utils:centralized_training_loop",
        "//generalization/utils:metric_utils",
    ],
)
