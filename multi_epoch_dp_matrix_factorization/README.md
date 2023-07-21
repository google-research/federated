This repository contains the code to reproduce experiments from "Multi-Epoch
Matrix Factorization Mechanisms for Private Machine Learning" found at
[[PDF](https://arxiv.org/pdf/2211.06530.pdf)][[arXiv](https://arxiv.org/abs/2211.06530)]
and published at ICML 2023.

Multi-epoch matrix factorization or, ME-MF, is a form of MF-DPFTRL approach. Our
main contributions are:

1.  a new method for capturing sensitivity in the multi-epoch setting, (Section
    2 of the paper),

2.  showing how to optimize matrices under this sensitivity constraint (Section
    3), which leads us to

3.  the creation of new state-of-the-art mechanisms for DP-ML (as evaluated in
    Section 5). We show how to "plug-and-play" our methods into existing DP-SGD
    optimization algorithms.

# Result Highlights

We consistently outperform DP-SGD to as low as ε≈2 on both CIFAR-10 (top) and
Stackoverflow Next Word Prediction (bottom). This can be observed from the red
lines on each graph. Further, observe that we achieve near the folklore
upper-bound of full-batch gradient descent as shown on the Stackoverflow
results. Finally, it is important to note our work shows new state-of-the-art DP
mechanisms---any additional improvements that have been studied for DP-SGD (like
using public data, augmentations, and more) are likely applicable to our setting
and we are excited to see them built on our work!

<img src="images/cifar10.png" height="200px">
<img src="images/stackoverflow.png" height="200px">

In the below, we show what our matrices look like. **A** represents the
workload, here, for Stackoverflow which shows a prefix-sum workload with
learning rate cooldown and momentum factorized for (discussed in more detail
below). **B** is the decoder used for noise generation, and **C** is the encoder
whose sensitivity we bound.

<img src="images/matrices.png" height="200px">

# Directory Structure

-   The `dp_ftrl` package contains much of the run code for training models with
    our DP-FTRL-based approaches.

*   The `dp_ftrl`.`centralized` subdirectory contains the training loops and
    noise generation code for centralized training. This also contains the run
    script for generating the matrices under the file
    `dp_ftrl`.`centralized`.`generate_and_write_matrices.py`.

-   The `fft` package contains code pertaining to the FFT mechanism in the
    paper. In particular, it contains code for bounding sensitivity and for
    pre-generating noise for the FFT approach. Representing these as their
    corresponding matrix mechanism is contained in the above directory.

-   The `multiple_participations` package contains the core code for bounding
    sensitivity and optimizing our multi-epoch matrices.

-   The main top-level modules contain common utilities for all of the above
    tasks.

# Background

Throughout this document, we will refer to the DP stochastic gradient descent
(SGD) algorithm of [2] as DP-SGD and our multi-epoch matrix factorization work
as MEMF-DP-FTRL.

## Differentially Private (DP) Machine Learning (ML)

DP-SGD, the current go-to algorithm for DP-ML is based on the following 4 steps.

1.  Instead of computing a single gradient for the mean loss, compute a gradient
    for each example, termed "per-example gradients".

2.  clip each per-example gradient one to some chosen threshold, known as the l2
    clipping norm.

3.  Compute the average gradient from the per-example gradients.

4.  Add Gaussian noise *z* of standard deviation σ where σ is calibrated to
    achieve some chosen (ε, δ)-DP guarantee.

To leverage a factorized matrix for MEMF-DP-FTRL, we only need to change what
noise we add to each step of DP-SGD. This can be translated to adding the
following step 5.

*   Instead of adding a sample from a Gaussian which is isotropic across time to
    the clipped gradients, add a *slice* of a Gaussian with some specified
    covariance in the time dimension. We may add this slice in either *model
    space*, corresponding to computing the ME-MF mechanism as **Ag + Bz**, or in
    gradient space, corresponding to computing the mechanism as **A(g +
    C^{-1}z)**, where **A = BC** is a factorization of the matrix **A** formed
    by viewing the process of model training as a linear operator from streams
    of gradients to streams of models. Such **A** can express gradient descent
    with momentum and arbitrary (data-independent) learning rate schedules, but
    leaves out important practical nonlinear optimizers like Adam and Adagrad.

## DP-FTRL

The idea of leveraging noise which is *correlated through time* for
differentially private ML training is not novel; to the best of our knowledge it
first appeared in
["Practical and Private (Deep) Learning without Sampling or Shuffling"](https://arxiv.org/abs/2103.00039),
which presented DP-FTRL, an algorithm grounded in online convex optimization and
leveraging the so-called 'tree aggregation mechanism' to compute private
estimates of sums of gradients, a key quantity in FTRL algorithms (see, e.g.,
Algorithm 2 in
[this paper](https://jmlr.org/papers/volume18/14-428/14-428.pdf)).

This algorithm was able to achieve similar utility to DP-SGD in certain regimes,
*without the need to rely on amplification by sampling or shuffling*, a critical
property for training settings in which the organization training the model does
not precisely control when users participate. DP-FTRL with federated learning
was therefore able to power the training and deployment of what appears to be
the first ML model trained directly on user data with
[formal differential privacy guarantees](https://ai.googleblog.com/2022/02/federated-learning-with-formal.html).

## Matrix Factorization

DP-FTRL was extended and connected with the literature on the matrix mechanism
in ["Improved Differential Privacy for SGD via Optimal Private Linear Operators
on Adaptive
Streams"](https://proceedings.neurips.cc/paper_files/paper/2022/file/271ec4d1a9ff5e6b81a6e21d38b1ba96-Paper-Conference.pdf),
which instantiated DP-FTRL as an instance of a continuous class of private
optimization methods, showing that *any* 'embedding matrix' **C** could be used
to yield a private algorithm (assuming the noise is distributed as a Gaussian).

The mechanisms proposed here are constructed by viewing the process of training
as a linear mapping from streams of gradients **g** to streams of models **Ag**.
Taking this view, we may ask about a *factorization* of the matrix **A** which
yields minimal expected squared error as a stream function when the arguments
are noised; IE, for what pair **(B, C)** is **B(Cg+z)** closest to **Ag**, where
**C** is chosen to ensure differential privacy of the output?

Any notion of differential privacy assumes a notion of neighboring dataset,
roughly corresponding to the manner in which a single user's data may be
accessed. The analysis in preceding work has leveraged a restriction here for
tractability: restricting to training that only runs for a *single epoch*; this
assumption significantly limits application of the resulting mechanisms, as many
applications require more than one epoch. This assumption also suppresses
certain subtle technical challenges which only appear in the presence of
multiple epochs, like NP-hardness of computing sensitivity of a general matrix
and the non-equivalence between vector-valued and matrix-valued sensitivities.

# Try it out!

## Decide on Training Dynamics

The first step is to decide on the training dynamics that will be used. This is
because these training dynamics will be directly modeled into the optimization
procedure, allowing us to generate a mechanism that is optimal (in the total
error). The following are what must be decided on a priori.

1.  The number of epochs (max number of participations per example/user). This
    is *k*. As well, the number of steps between each participation, this is
    *b*. This can be calculated as the dataset size (*m*) divided by the batch
    size, i.e., both must be chosen in advance.

2.  The workload matrix **A** must be chosen in advance. In particular, there
    are a few options, where all are distinctly lower-triangular:

*   The prefix sum workload, corresponding to lower triangular **A** with
    entries all 1. This corresponds to machine learning of the prefix sums and,
    through a simple post-processing via the
    `tff_aggregator.create_residual_prefix_sum_dp_factory` function, to SGD.
    Notice that instead of learning the current model at each step, as in prefix
    sums, this will return a decoder matrix **B** augmented for a workload **A**
    for the residuals, i.e., the gradient update as returned by SGD at each
    step. Our CIFAR-10 results use these workloads, with learning rate cooldown
    and momentum passed to the SGD optimizer rather than directly optimized for
    in the matrices, as is possible and discussed below.

*   The momentum workload, for some momentum parameter. The exact parameter must
    be chosen in advance.

*   The learning rate cooldown workload, for some per-round learning rates
    chosen a priori.

*   Momentum workload and learning rate cooldown. This is the setting used for
    our Stack Overflow Next Word Prediction Experiments.

## Generating Matrices

To generate matrices for CIFAR-10, the
`dp_ftrl`.`centralized`.`generate_and_write_matrices`.py run script will do
this. As mentioned above, our CIFAR-10 results assume the prefix sum workload.
Thus, this is assumed in this script.

To generate matrices for other workloads, in particular for the Stackoverflow
results, one may use the binary
`multiple_partitipations.factorize_multi_epoch_prefix_sum.py`. Simply set the
flags to express the matrix you are interested in factorizing, and the library
will log progress in its optimization procedure as well as write the results to
a provided directory. There may be some required tuning of optimization
parameters to ensure convergence.

Take note: for both paths abovs, the optimization code here will become
computationally quite intensive as matrix size grows. The results in the present
paper were generated for approximately 2000 steps, which ; in some situations
matrices can be generated up to around 10000 steps, but this should likely be
understood as a more or less strict upper bound on feasible matrix factorization
(using a single machine, at least), with the code here.

## Train!

To train models using our approach, it suffices to call the run scripts with the
path to the saved matrix using the corresponding training dynamics chosen
earlier, and specifying any remaining choices. For example, the noise multiplier
must be specified to attain a chosen DP guarantee. The `accounting`.py module
contains the relevant code under the assumption that the l2_clip_norm is 1.

The run scripts needed are:

-   `dp_ftrl`.`centralized`.`run_training`.py for CIFAR-10.
-   `dp_ftrl`.`run_stackoverlow`.py for Stackoverflow.

# Citations

This manuscript will be appearing in ICML 2023 as an oral presentation and
poster!

```
@article{choquette2022multi,
  title={Multi-Epoch Matrix Factorization Mechanisms for Private Machine Learning},
  author={Choquette-Choo, Christopher A and McMahan, H Brendan and Rush, Keith and Thakurta, Abhradeep},
  journal={arXiv preprint arXiv:2211.06530},
  year={2022}
}
```

# Troubleshooting

-   If errors occur with calls to the TFF util libraries (mainly used by the FL
    training scripts), try using a TFF nightly build.

# References

[1] Choquette-Choo, Christopher A., et al. "Multi-Epoch Matrix Factorization
Mechanisms for Private Machine Learning." arXiv preprint arXiv:2211.06530
(2022).

[2] Goodfellow, Ian, Yoshua Bengio, and Aaron Courville. Deep learning. MIT
press, 2016.
