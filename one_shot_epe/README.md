# One-shot Empirical Privacy Estimation for Federated Learning

This directory contains the code for estimating the DP epsilon during a run of
federated training via insertion of random canary clients. It is sufficient to
run the experiments on the StackOverflow and EMNIST federated datasets described
in the paper "One-shot Empirical Privacy Estimation for Federated Learning",
https://arxiv.org/pdf/2302.03098.pdf.

# Example command line invocation

bazel run train -- \
--root_output_dir=/tmp/test_output_dir \
--run_name=test_run \
--clients_per_round=10 \
--total_rounds=10 \
--client_optimizer=sgd \
--client_learning_rate=0.03 \
--server_optimizer=sgd \
--server_learning_rate=1.0 \
--server_sgd_momentum=0.9 \
--num_test_examples=1000 \
--task=emnist_character \
--train_epochs=1 \
--num_canaries=100 \
--num_unseen_canaries=100 \
--noise_multiplier=0.1 \
--sampling=shuffle
