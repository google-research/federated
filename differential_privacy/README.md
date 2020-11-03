# Federated EMNIST and StackOverflow Baseline Experiments with differential privacy

Note: This directory is a work-in-progress.

## Example usage

`bazel run emnist:run_federated -- --clients_per_round 2 --uniform_weighting
--noise_multiplier 0.01 --total_rounds 20 --client_optimizer sgd
--client_learning_rate 0.02 --server_optimizer sgd --server_learning_rate 1.0
--root_output_dir /tmp/dp/emnist --experiment_name debug`
