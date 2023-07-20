load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(
    default_applicable_licenses = ["//:package_license"],
    default_visibility = [
        "//multi_epoch_dp_matrix_factorization:dp_matrix_factorization_packages",
        "//multi_epoch_dp_matrix_factorization:dp_matrix_factorization_users",
    ],
)

package_group(
    name = "dp_matfac_dpftrl_centralized",
    packages = ["//multi_epoch_dp_matrix_factorization/dp_ftrl/centralized/..."],
)

licenses(["notice"])

py_library(
    name = "data_loaders",
    srcs = ["data_loaders.py"],
    srcs_version = "PY3",
)

py_test(
    name = "data_loaders_test",
    timeout = "moderate",
    srcs = ["data_loaders_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":data_loaders"],
)

py_library(
    name = "grad_processor_builders",
    srcs = ["grad_processor_builders.py"],
    srcs_version = "PY3",
    deps = [
        ":gradient_processors",
        "//multi_epoch_dp_matrix_factorization:matrix_io",
        "//multi_epoch_dp_matrix_factorization:tff_aggregator",
        "//multi_epoch_dp_matrix_factorization/fft:generate_noise",
        "//multi_epoch_dp_matrix_factorization/multiple_participations:contrib_matrix_builders",
        "//third_party/py/haiku",
    ],
)

py_test(
    name = "grad_processor_builders_test",
    srcs = ["grad_processor_builders_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":grad_processor_builders",
        ":gradient_processors",
    ],
)

py_library(
    name = "gradient_processors",
    srcs = ["gradient_processors.py"],
    srcs_version = "PY3",
)

py_test(
    name = "gradient_processors_test",
    timeout = "short",
    srcs = ["gradient_processors_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":gradient_processors"],
)

py_library(
    name = "models",
    srcs = ["models.py"],
    srcs_version = "PY3",
    deps = ["//third_party/py/haiku"],
)

py_test(
    name = "models_test",
    timeout = "short",
    srcs = ["models_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":models"],
)

py_binary(
    name = "run_training",
    srcs = ["run_training.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":run_training_lib",
    ],
)

py_library(
    name = "run_training_lib",
    srcs = ["run_training.py"],
    srcs_version = "PY3",
    deps = [
        ":data_loaders",
        ":grad_processor_builders",
        ":models",
        ":training_loop",
        "//third_party/py/optax",
    ],
)

py_test(
    name = "run_training_test",
    srcs = ["run_training_test.py"],
    srcs_version = "PY3",
    deps = [
        ":grad_processor_builders",
        ":run_training_lib",
    ],
)

py_library(
    name = "training_loop",
    srcs = ["training_loop.py"],
    srcs_version = "PY3",
    deps = [
        ":gradient_processors",
        "//utils:training_utils",
        "//third_party/py/haiku",
        "//third_party/py/optax",
    ],
)

py_library(
    name = "generate_and_write_matrices",
    srcs = ["generate_and_write_matrices.py"],
    deps = [
        "//multi_epoch_dp_matrix_factorization:matrix_io",
        "//multi_epoch_dp_matrix_factorization/fft:generate_noise",
        "//multi_epoch_dp_matrix_factorization/multiple_participations:contrib_matrix_builders",
    ],
)

py_library(
    name = "xm_experiment_definitions",
    srcs = ["xm_experiment_definitions.py"],
    deps = [":xm_fft_experiment_definitions"],
)

py_library(
    name = "xm_fft_experiment_definitions",
    srcs = ["xm_fft_experiment_definitions.py"],
)

py_test(
    name = "training_loop_test",
    timeout = "moderate",
    srcs = ["training_loop_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":gradient_processors",
        ":models",
        ":training_loop",
        "//third_party/py/haiku",
        "//third_party/py/optax",
    ],
)
