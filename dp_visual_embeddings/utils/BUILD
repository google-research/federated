load("@rules_python//python:defs.bzl", "py_library", "py_test")

licenses(["notice"])

package(default_visibility = ["//dp_visual_embeddings:__subpackages__"])

py_library(
    name = "exporting_program_state_manager",
    srcs = ["exporting_program_state_manager.py"],
)

py_test(
    name = "exporting_program_state_manager_test",
    srcs = ["exporting_program_state_manager_test.py"],
    deps = [
        ":export",
        ":exporting_program_state_manager",
    ],
)

py_library(
    name = "file_utils",
    srcs = ["file_utils.py"],
)

py_test(
    name = "file_utils_test",
    srcs = ["file_utils_test.py"],
    deps = [":file_utils"],
)

py_library(
    name = "export",
    srcs = ["export.py"],
)

py_test(
    name = "export_test",
    srcs = ["export_test.py"],
    deps = [":export"],
)
