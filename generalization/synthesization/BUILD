load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

licenses(["notice"])

package(default_visibility = [
    "//generalization:__subpackages__",
])

py_library(
    name = "dirichlet",
    srcs = ["dirichlet.py"],
    srcs_version = "PY3",
    deps = ["//generalization/utils:client_data_utils"],
)

py_test(
    name = "dirichlet_test",
    srcs = ["dirichlet_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":dirichlet"],
)

py_library(
    name = "synthesizer_lib",
    srcs = ["synthesizer.py"],
    srcs_version = "PY3",
    deps = [
        ":cifar_synthesis",
        ":mnist_synthesis",
        "//generalization/utils:sql_client_data_utils",
    ],
)

py_binary(
    name = "synthesizer",
    srcs = ["synthesizer.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":cifar_synthesis",
        ":mnist_synthesis",
        "//generalization/utils:sql_client_data_utils",
    ],
)

py_library(
    name = "coarse_dirichlet",
    srcs = ["coarse_dirichlet.py"],
    srcs_version = "PY3",
    deps = ["//generalization/utils:client_data_utils"],
)

py_library(
    name = "mnist_synthesis",
    srcs = ["mnist_synthesis.py"],
    srcs_version = "PY3",
    deps = [
        ":dirichlet",
        ":gmm_embedding",
    ],
)

py_test(
    name = "coarse_dirichlet_test",
    srcs = ["coarse_dirichlet_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":coarse_dirichlet"],
)

py_library(
    name = "cifar_synthesis",
    srcs = ["cifar_synthesis.py"],
    srcs_version = "PY3",
    deps = [
        ":coarse_dirichlet",
        ":dirichlet",
        ":gmm_embedding",
    ],
)

py_library(
    name = "gmm_embedding",
    srcs = ["gmm_embedding.py"],
    srcs_version = "PY3",
    deps = [
        "//generalization/utils:client_data_utils",
        "//generalization/utils:logging_utils",
        "//generalization/utils:tf_gaussian_mixture",
    ],
)

py_test(
    name = "gmm_embedding_test",
    srcs = ["gmm_embedding_test.py"],
    python_version = "PY3",
    shard_count = 8,
    srcs_version = "PY3",
    deps = [":gmm_embedding"],
)
