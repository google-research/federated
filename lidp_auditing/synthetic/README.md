# Synthetic Experiments with the Gaussian mechanism

We describe here how to reproduce the experimental results of Section 4 of the
[Lifted DP paper](https://arxiv.org/pdf/2305.18447.pdf) published in NeurIPS
2023.

## Step 0: Setup default parameters

We define some overall parameters, each of which contribute one point in the
plots of the paper. These parameters need to be varied as described in the
comments below to obtain the various data points used to produce the plot. The
upcoming Steps 1 and 2 need to be rerun for each set of parameters.

```bash
output_dir='path/to/output/dir'  # NOTE: change
# make sure this directory exists
out_name='CI1'  # name of the output
n=256  # num_samples: vary in [256, 1024, 4096, 16384, 65536] for the plots
k=16  # num_canaries: vary in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
dim=100  # dimension: vary in [100, 1000, int(1e4), int(1e5), int(1e6)]
beta=0.05  # Failure probability
```

## Step 1: Tune the threshold

The test statistic compares the dot products against a threshold. Our first step
is to tune this threshold. This is analogous to tuning a hyperparameter on a
validation set.

```bash
threshold_type='tune'  # denote that the threshold is to be tuned
seed=1000  # Random seed used for tuning

bazel run :run_synthetic -- \
    --threshold_type=${threshold_type} \
    --out_name=${out_name} \
    --num_samples=${num_samples}  \
    --num_canaries=${num_canaries}  \
    --dim=${dim}  \
    --beta=${beta} \
    --seed=${seed}
```

This file saves the best threshold as per each estimator in
`${output_dir}/${out_name}_n${n}_k${k}_dim${dim}_beta${beta}_threshold.csv`.

## Step 2: Use the tuned thresholds to get the epsilon estimates

Next, we run the following command to use the tuned thresholds on fresh runs.

```bash
threshold_type='saved'  # denote that the threshold is to be tuned
seed=0  # Random seed. The plots are produced from the mean and std of 25 seeds

bazel run :run_synthetic -- \
    --threshold_type=${threshold_type} \
    --out_name=${out_name} \
    --num_samples=${num_samples}  \
    --num_canaries=${num_canaries}  \
    --dim=${dim}  \
    --beta=${beta} \
    --seed=${seed}
```

This produces a csv file
`${output_dir}/${out_name}_n${n}_k${k}_dim${dim}_beta${beta}_seed{seed}_empeps.csv`.
To produce the plots, read this file with pandas. Each row contains the
confidence estimator (e.g. "4th-Order Wilson"), while each column contains a
(squared) noise multiplier. These below to the list `sigmasq_list = [58.8,
16.36, 7.788, 4.62, 2.23, 1.34, 0.4066, 0.13135]` and they correspond to
`epsilon = [0.5, 1, 1.5, 2, 3, 4, 8, 16]` respectively at `delta=1e-5`.
