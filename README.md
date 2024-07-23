# SLT-in-Generative-Models
code for this paper: [Can We Find Strong Lottery Tickets in Generative Models?](https://ojs.aaai.org/index.php/AAAI/article/view/25433)

# Abs
Yes. In this paper, we investigate strong lottery tickets in generative models, the subnetworks that achieve good generative performance without any weight update. Neural network pruning is considered the main cornerstone of model compression for reducing the costs of computation and memory. Unfortunately, pruning a generative model has not been extensively explored, and all existing pruning algorithms suffer from excessive weight-training costs, performance degradation, limited generalizability, or complicated training. To address these problems, we propose to find a strong lottery ticket via moment-matching scores. Our experimental results show that the discovered subnetwork can perform similarly or better than the trained dense model even when only 10% of the weights remain. To the best of our knowledge, we are the first to show the existence of strong lottery tickets in generative models and provide an algorithm to find it stably. Our code and supplementary materials are publicly available.

# Requirements
`python=3.6.0` `pytorch==1.10.0` `tqdm` `prdc` `pandas`

# Command
`python main.py --config config/config.yaml`

# Architecture
`Generator: ResNet, SNGAN`

# Loss
`MMD loss by feature matching`

# Algorithms for finding subnetworks
`Conventional training, Edge-popup, Global Edge-popup, IMP, Gem-miner`