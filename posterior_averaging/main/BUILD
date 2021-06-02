load("@rules_python//python:defs.bzl", "py_binary")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_binary(
    name = "federated_trainer",
    srcs = ["federated_trainer.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//optimization/tasks:cifar100",
        "//optimization/tasks:emnist",
        "//optimization/tasks:emnist_ae",
        "//optimization/tasks:shakespeare",
        "//optimization/tasks:stackoverflow_nwp",
        "//optimization/tasks:stackoverflow_tp",
        "//optimization/tasks:training_specs",
        "//posterior_averaging/shared:fed_pa_schedule",
        "//utils:training_loop",
        "//utils:utils_impl",
        "//utils/optimizers:optimizer_utils",
    ],
)
