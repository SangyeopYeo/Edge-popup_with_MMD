# SLT-in-Generative-Models
Code for the paper: [Can We Find Strong Lottery Tickets in Generative Models?](https://ojs.aaai.org/index.php/AAAI/article/view/25433)

## Abstract
Yes. In this paper, we investigate strong lottery tickets in generative models, the subnetworks that achieve good generative performance without any weight update. Neural network pruning is considered the main cornerstone of model compression for reducing the costs of computation and memory. Unfortunately, pruning a generative model has not been extensively explored, and all existing pruning algorithms suffer from excessive weight-training costs, performance degradation, limited generalizability, or complicated training. To address these problems, we propose to find a strong lottery ticket via moment-matching scores. Our experimental results show that the discovered subnetwork can perform similarly or better than the trained dense model even when only 10% of the weights remain. To the best of our knowledge, we are the first to show the existence of strong lottery tickets in generative models and provide an algorithm to find it stably. Our code and supplementary materials are publicly available.

## Requirements

To set up the environment for this project, follow these steps:

1. Create a new conda environment:
   ```
   conda create -n SLT python==3.12
   ```

2. Activate the environment:
   ```
   conda activate SLT
   ```

3. Install the required packages:
   ```
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip3 install scipy==1.14.1
   pip3 install tqdm==4.66.5
   pip3 install prdc==0.2
   pip3 install pandas==2.2.3
   pip3 install pyyaml==6.0.2

Note: The specific versions of the packages are listed to ensure reproducibility. If you encounter any issues, you may try installing without version specifiers, but be aware that this might lead to compatibility problems. This code was tested on CUDA version 11.4

## Usage

To run the main script, use the following command:

```
python main.py --config config/config.yaml
```


## Architecture

The project supports the following generator architectures:
- ResNet
- SNGAN

## Loss

The loss function used in this project is:
- MMD loss by feature matching

## Algorithms for finding subnetworks

The following algorithms are implemented for finding subnetworks:
- Conventional training
- Edge-popup
- Global Edge-popup
- IMP
- Gem-miner


