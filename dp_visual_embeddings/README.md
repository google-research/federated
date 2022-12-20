# Learning to Generate Image Embeddings with User-level Differential Privacy

## Overview

Initial code releasing for
["Learning to Generate Image Embeddings with User-level Differential Privacy"](https://arxiv.org/abs/2211.10844).

## Requirements

This code is implemented with
[TensorFlow Federated](https://www.tensorflow.org/federated). See
[Using TFF for Federated Learning Research](https://www.tensorflow.org/federated/tff_for_research)
for more instructions on using TensorFlow Federated for research.

The following packages may need to be installed

<!-- mdformat off (multiple lines of small code piece) -->

```bash
absl-py>=1.0,==1.*
dp-accounting==0.3.0
tensorflow-privacy==0.8.6
tensorflow-federated~=0.40.0
tensorflow~=2.10.0
```

<!-- mdformat on -->

## Example usage

```
bazel run run_federated -- \
   --task_type=emnist --experiment_name=emnist_fix1_NM0.62_r200 \
   --dynamic_clients=8 --client_epochs_per_round=8 \
   --clients_per_round=32 \
   --max_concurrent_threads=4 --use_client_softmax=True \
   --client_batch_size=32 --max_examples_per_client=2048 \
   --client_lr=5e-3 --server_lr=0.02 --head_lr_scale=100 \
   --client_momentum=0.9 \
   --model_backbone=lenet \
   --total_rounds=200 \
   --rounds_per_eval=40 --rounds_per_checkpoint=40 \
   --client_shuffle_buffer_size=2048 \
   --aggregator_type=dpsgd \
   --clip_norm=1 --noise_multiplier=0.62
```

## Citation

```
@article{xu2022learning,
  title={Learning to Generate Image Embeddings with User-level Differential Privacy},
  author={Xu, Zheng and Collins, Maxwell and Wang, Yuxiao and Panait, Liviu and Oh, Sewoong and Augenstein, Sean and Liu, Ting and Schroff, Florian and McMahan, H Brendan},
  journal={arXiv preprint arXiv:2211.10844},
  year={2022}
}
```
