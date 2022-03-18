load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(default_visibility = ["//visibility:private"])

licenses(["notice"])

py_library(
    name = "triehh_tf",
    srcs = ["triehh_tf.py"],
    srcs_version = "PY3",
    deps = ["//analytics/heavy_hitters:heavy_hitters_utils"],
)

py_library(
    name = "triehh_tff",
    srcs = ["triehh_tff.py"],
    srcs_version = "PY3",
    deps = [":triehh_tf"],
)

py_test(
    name = "triehh_tff_test",
    srcs = ["triehh_tff_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":triehh_tf",
        ":triehh_tff",
        "//analytics/heavy_hitters:heavy_hitters_testcase",
    ],
)

py_test(
    name = "triehh_tf_test",
    srcs = ["triehh_tf_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":triehh_tf",
        "//analytics/heavy_hitters:heavy_hitters_testcase",
    ],
)
