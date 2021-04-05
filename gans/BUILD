load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "checkpoint_utils",
    srcs = ["checkpoint_utils.py"],
    srcs_version = "PY3",
)

py_test(
    name = "checkpoint_utils_test",
    size = "large",
    srcs = ["checkpoint_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":checkpoint_utils"],
)

py_library(
    name = "gan_losses",
    srcs = ["gan_losses.py"],
    srcs_version = "PY3",
)

py_library(
    name = "gan_training_tf_fns",
    srcs = ["gan_training_tf_fns.py"],
    srcs_version = "PY3",
    deps = [
        ":gan_losses",
        "//utils:tensor_utils",
    ],
)

py_test(
    name = "gan_training_tf_fns_test",
    srcs = ["gan_training_tf_fns_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":gan_losses",
        ":gan_training_tf_fns",
        ":one_dim_gan",
    ],
)

py_library(
    name = "one_dim_gan",
    srcs = ["one_dim_gan.py"],
    srcs_version = "PY3",
)

py_library(
    name = "tff_gans",
    srcs = ["tff_gans.py"],
    srcs_version = "PY3",
    deps = [":gan_training_tf_fns"],
)

py_test(
    name = "tff_gans_test",
    size = "large",
    srcs = ["tff_gans_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":gan_losses",
        ":gan_training_tf_fns",
        ":one_dim_gan",
        ":tff_gans",
    ],
)

py_library(
    name = "training_loops",
    srcs = ["training_loops.py"],
    srcs_version = "PY3",
    deps = [
        ":checkpoint_utils",
        ":gan_training_tf_fns",
        ":tff_gans",
    ],
)

py_test(
    name = "training_loops_test",
    size = "large",
    srcs = ["training_loops_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":gan_losses",
        ":gan_training_tf_fns",
        ":one_dim_gan",
        ":tff_gans",
        ":training_loops",
    ],
)
