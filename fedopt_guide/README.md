# FedOpt Guide experiments

Note: This directory is a work-in-progress.

This is a shared folder for developing a field guide to federated optimization.

*   The code in this folder should mimic the structure in `optimization/`.
*   Minimize the number of PRs when you try to commit your code. Note that you
    do not need to commit your code at this point if it will not be used by
    others.
*   Consider creating a subfolder for each task by `dataset_model`, for example,
    `fedopt_guide/gld23k_mobilenet`.
*   Re-use the shared code in `optimization/shared/` and `utils/` whenever
    possible, for example, `utils.training_loop` is recommended for the training
    loop. Use `tff.learning.build_federated_averaging_process` instead of
    `optimization.shared.fed_avg_schedule` to build your TFF iterative process.
