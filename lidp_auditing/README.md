# LiDP Auditing: Unleashing the Power of Randomization in Auditing DP

This is the code to reproduce the experimental results of the NeurIPS 2023 paper
[Unleashing the Power of Randomization in Auditing Differentially Private ML](https://arxiv.org/abs/2305.18447).

Auditing differential privacy for ML involves running membership inference many
times and giving high-confidence estimates on the success of the attack (i.e.,
we try to detect the presence of a crafted datapoint, called a "canary" in the
training data).

[This paper](https://arxiv.org/abs/2305.18447) introduces a variant of DP called
"Lifted DP" (or "LiDP" in short) that is equivalent to the usual notions of DP.
It also gives a recipe to audit LiDP with multiple randomized hypothesis tests
and adaptive confidence intervals to improve the sample complexity of auditing
DP by 4 to 16 times.

## Cite

If you found this code useful, please cite the following work.

```
@incollection{pillutla-etal:lidp_auditing:neurips2023,
title = {{Unleashing the Power of Randomization in Auditing
          Differentially Private ML}},
author = {Krishna Pillutla and Galen Andrew and Peter Kairouz and
          H. Brendan McMahan and Alina Oprea and Sewoong Oh},
booktitle = {NeurIPS},
year = {2023},
}
```

## Generating the experimental results

For the synthetic experiments with the Gaussian mechanism, see the
`synthetic/README.md`.

For the experiments with real data, follow the steps below:

1.  Train 2000 models (1000 models for parameter tuning and the other 1000 for
    reporting the epsilon lower bounds). See `main_central.py` for details on
    the command line arguments.

```bash
### General arguments
output_dir="./outputs"  # NOTE: set the output directory
dataset="fashion_mnist"  # Name of the dataset
model="mlp"  # Model type: can be "linear" or "mlp"
seed=0  # Random seed, vary from 0 to 1999 (total 2000 seeds)
### Canary arguments
canary_type="random_gradient"  # Can be "random_gradient" or "static_data"
num_canaries=64  # Vary from 1 to 512
min_dimension=0  # Minimum random seed for the random canary gradient
max_dimension=1000000  # Minimum random seed for the random canary gradient
### For canary_type="static_data", uncomment the following two lines:
# min_dimension=300  # Minimum PCA direction for the canary
# max_dimension=784  # Maximum PCA direction for the canary (= data dimension)
### Learning arguments
learning_rate=0.01  # Use 0.02 for the linear model
dp_epsilon=2  # Vary from 1 to 32 in powers of 2
dp_delta="1e-5"

arguments="--experiment_name="run_${dataset}_${model}" --output_dir=${output_dir}  \
    --dataset_name=${dataset}  --model_type=${model}  \
    --canary_type=${canary_type}  \
    --min_dimension=${min_dimension}  --max_dimension=${max_dimension}  \
    --batch_size=100 --num_epochs=30 --learning_rate=${learning_rate}  \
    --dp_epsilon=${dp_epsilon} --dp_delta=${dp_delta} \
    --seed=${seed}"

# Alternate hypothesis: run with k canaries
bazel run :main_central -- ${arguments} \
    --num_canaries=${num_canaries}

# Null hypothesis: run with k-1 canaries (for training but test on k canaries)
bazel run :main_central -- ${arguments} \
    --num_canaries=$((num_canaries-1))  \
    --test_canary_add_one=True
```

This code also saves various files in ${output_dir}/run_${dataset}_${model}
tracking the test statistic of the training and test canaries.

1.  Obtain the confidence intervals from the saved logs using the confidence
    estimators from `lidp_auditing/confidence_estimators`. These instructions
    will be completed later.
