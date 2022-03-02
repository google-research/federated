load("@rules_python//python:defs.bzl", "py_library", "py_test")

package(
    default_visibility = [
        "//rcc_dp:__subpackages__",
    ],
)

licenses(["notice"])

py_library(
    name = "modify_pi",
    srcs = ["modify_pi.py"],
)

py_test(
    name = "modify_pi_test",
    srcs = ["modify_pi_test.py"],
    deps = [
        ":modify_pi",
        "//rcc_dp/mean_estimation:get_parameters",
        "//rcc_dp/mean_estimation:miracle",
    ],
)
