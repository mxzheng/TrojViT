# TrojViT: Trojan Insertion in Vision Transformers
Mengxin Zheng, Qian Lou, Lei Jiang

Accepted at CVPR 2023 [[Paper Link](https://arxiv.org/abs/2208.13049)].

## Overview

- We propose a new attack framework, *TrojViT*, to breach the security of ViTs by creating a novel, stealthy, and practical ViT-specific backdoor attack TrojViT.

- We evaluate *TrojViT* on vit, deit and swin transformer. 
  

<p align="center">
  <img src="(https://d3i71xaburhd42.cloudfront.net/89b59789b98219d08209e7864486241ee36050a6/250px/1-Figure1-1.png)" width="800">
</p>


## Code Usage
Our codes support the *TrojViT* attack on SOTA Vision Transformers (e.g., DeiT-T, DeiT-S, and DeiT-B) on ImageNet validation dataset.

### Key parameters
```--data_dir```: Path to the ImageNet folder.

```--dataset_size```: Evaluate on a part of the whole dataset.

```--patch_select```: Select patches based on the saliency map, attention map, or random selection.

```--num_patch```: Number of perturbed patches.

```--sparse_pixel_num```: Total number of perturbed pixels in the whole image.

```--attack_mode```: Optimize TrojViT based on the final cross-entropy loss only, or consider both cross-entropy loss and the attention map.

```--attn_select```: Select patches based on which attention layer.



## Citation
```

@inproceedings{zheng2023trojvit,
  title={Trojvit: Trojan insertion in vision transformers},
  author={Zheng, Mengxin and Lou, Qian and Jiang, Lei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4025--4034},
  year={2023}
}
```
