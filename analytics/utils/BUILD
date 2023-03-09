load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = ["//visibility:private"],
)

licenses(["notice"])

py_library(
    name = "non_iid_histograms",
    srcs = ["non_iid_histograms.py"],
)

py_test(
    name = "non_iid_histograms_test",
    srcs = ["non_iid_histograms_test.py"],
    python_version = "PY3",
    deps = [":non_iid_histograms"],
)
