load("@rules_python//python:defs.bzl", "py_binary")

licenses(["notice"])

py_binary(
    name = "run_federated",
    srcs = ["run_federated.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//optimization/shared:optimizer_utils",
        "//utils:training_loop",
        "//utils:training_utils",
        "//utils:utils_impl",
        "//utils/datasets:emnist_dataset",
        "//utils/models:emnist_models",
    ],
)
