load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

licenses(["notice"])

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
        "//utils:task_utils",
        "//utils:training_utils",
        "//utils:utils_impl",
        "//utils/optimizers:optimizer_utils",
    ],
)

py_test(
    name = "trainer_test",
    srcs = ["trainer_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":trainer_lib"],
)
