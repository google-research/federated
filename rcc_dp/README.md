# Source code for "Optimal Compression of Locally Differentially Private Mechanisms"

Reference: Abhin Shah, Wei-Ning Chen, Johannes Balle, Peter Kairouz, Lucas
Theis, "Optimal Compression of Locally Differentially Private Mechanisms," The
25th International Conference on Artificial Intelligence and Statistics
(AISTATS), 2022

Contact: abhin@mit.edu, jballe@google.com

Arxiv:
[https://arxiv.org/pdf/2111.00092.pdf](https://arxiv.org/pdf/2111.00092.pdf)

### Dependencies:

In order to successfully execute the code, the following libraries must be
installed:

Python --- json, math, time, matplotlib, numpy, scipy,
[absl](https://github.com/abseil/abseil-py),
[ml_collections](https://pypi.org/project/ml-collections/)

### Wrapper functions:

This repository contains the code for (a) mean estimation and (b) frequency
estimation. To run the code, a wrapper function needs to be written. For
example, to run the mean estimation code, the following could be used:

```
from mean_estimation import config as defaults
from mean_estimation import experiment
from mean_estimation import experiment_coding_cost

def main():
    config = defaults.get_config()
    experiment.evaluate('path-to-the-mean-estimation-code', config)

if __name__ == "__main__":
    main()
```

Similarly, to run the frequency estimation code, the following could be used:

```
from frequency_estimation import config as defaults
from frequency_estimation import experiment
from frequency_estimation import experiment_coding_cost

def main():
    config = defaults.get_config()
    experiment.evaluate('path-to-the-frequency-estimation-code', config)

if __name__ == "__main__":
    main()
```

### Reproducing the figures

1.  To reproduce Figure 1(Top), make the following changes in
    mean_estimation/config.py: `num_itr = 10 vary = "cc"` and add the following
    commands in the main function of the wrapper: `config =
    defaults.get_config()
    experiment_coding_cost.evaluate('path-to-the-mean-estimation-code', config)`
2.  To reproduce Figure 1(Bottom), make the following changes in
    mean_estimation/config.py: `num_itr = 10 vary = "eps"` and add the following
    commands in the main function of the wrapper: `config =
    defaults.get_config()
    experiment.evaluate('path-to-the-mean-estimation-code', config)`
3.  To reproduce Figure 2(Top), make the following changes in
    frequency_estimation/config.py: `num_itr = 10 vary = "cc"` and add the
    following commands in the main function of the wrapper: `config =
    defaults.get_config()
    experiment_coding_cost.evaluate('path-to-the-frequency-estimation-code',
    config)`
4.  To reproduce Figure 2(Bottom), make the following changes in
    frequency_estimation/config.py: `num_itr = 10 vary = "eps"` and add the
    following commands in the main function of the wrapper: `config =
    defaults.get_config()
    experiment.evaluate('path-to-the-frequency-estimation-code', config)`
5.  To reproduce Figure 3, make the following changes in
    mean_estimation/config.py: `run_approx_miracle=True,
    run_modified_miracle=False, num_itr = 10 vary = "eps"` and add the following
    commands in the main function of the wrapper: `config =
    defaults.get_config()
    experiment.evaluate('path-to-the-mean-estimation-code', config)`
6.  To reproduce Figure 4(Left), make the following changes in
    mean_estimation/config.py: `num_itr = 10 vary = "d"` and add the following
    commands in the main function of the wrapper: `config =
    defaults.get_config()
    experiment.evaluate('path-to-the-mean-estimation-code', config)`
7.  To reproduce Figure 4(Right), make the following changes in
    mean_estimation/config.py: `num_itr = 10 vary = "n"` and add the following
    commands in the main function of the wrapper: `config =
    defaults.get_config()
    experiment.evaluate('path-to-the-mean-estimation-code', config)`
8.  To reproduce Figure 5, make the following changes in
    frequency_estimation/config.py: `run_approx_miracle=True,
    run_modified_miracle=False, num_itr = 10 vary = "eps"` and add the following
    commands in the main function of the wrapper: `config =
    defaults.get_config()
    experiment.evaluate('path-to-the-frequency-estimation-code', config)`
9.  To reproduce Figure 6(Left), make the following changes in
    frequency_estimation/config.py: `num_itr = 10 vary = "d"` and add the
    following commands in the main function of the wrapper: `config =
    defaults.get_config()
    experiment.evaluate('path-to-the-frequency-estimation-code', config)`
10. To reproduce Figure 6(Right), make the following changes in
    frequency_estimation/config.py: `num_itr = 10 vary = "n"` and add the
    following commands in the main function of the wrapper: `config =
    defaults.get_config()
    experiment.evaluate('path-to-the-frequency-estimation-code', config)`
