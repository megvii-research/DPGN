# DPGN: Distribution Propagation Graph Network for Few-shot Learning

This repository is the official implementation of [DPGN: Distribution Propagation Graph Network for Few-shot Learning](https://arxiv.org/abs/2003.14247). 

<p align='center'>
  <img src='figure/dpgn.png' width="800px">
</p>

## Abstract
Most graph-network-based meta-learning approaches model instance-level relation of examples. We extend this idea further to explicitly model the distribution-level relation of one example to all other examples in a 1-vs-N manner. We propose a novel approach named distribution propagation graph network (DPGN) for few-shot learning. It
conveys both the distribution-level relations and instance-level relations in each few-shot learning task. To combine the distribution-level relations and instance-level relations
for all examples, we construct a dual complete graph network which consists of a point graph and a distribution graph with each node standing for an example. Equipped
with dual graph architecture, DPGN propagates label information from labeled examples to unlabeled examples within several update generations. In extensive experiments on
few-shot learning benchmarks, DPGN outperforms state-of-the-art results by a large margin in 5% ∼ 12% under supervised settings and 7% ∼ 13% under semi-supervised settings. Code will be released.

## Requirements

CUDA Version: 10.1

Python : 3.5.2

To install dependencies:

```setup
sudo pip3 install -r requirements.txt
```
## Dataset
Download the *mini*-Imagenet dataset from [here](https://drive.google.com/open?id=1RGhzbN1C8gPMop6XBtr7a1afx3rmUGK5). Those pickle files are inhereted from [EGNN](https://drive.google.com/drive/folders/15WuREBvhEbSWo4fTr1r-vMY0C_6QWv4w) paper.

The dataset directory should look like this:
```bash
├── dataset
    ├── mini-imagenet
        ├── mini_imagenet_test.pickle   
        ├── mini_imagenet_train.pickle  
        ├── mini_imagenet_val.pickle

```

## Training

To train the model(s) in the paper, run:

```train
python3 main.py --dataset_root dataset --config config/5way_1shot_resnet12_mini-imagenet.py --num_gpu 1 --mode train
```


## Evaluation

To evaluate the model(s) in the paper, run:

```eval
python3 main.py --dataset_root dataset --config config/5way_1shot_resnet12_mini-imagenet.py --num_gpu 1 --mode eval
```

## Pre-trained Models
<!--[**best_checkpoints**](best_checkpoints) directory contains pre-trained model under settings of 5way-1shot and 5way-5shots for mini-ImageNet dataset with ResNet12 backbone.-->
[This Google Drive](https://drive.google.com/open?id=1ZF4wB4tId7YZX6m_HYdo87f_Q1RjaiYN) contains pre-trained model under settings of 5way-1shot and 5way-5shots for mini-ImageNet dataset with ResNet12 backbone.

## Results
```bash
# Default checkpoints directory is:
./checkpoints
```

```bash
# Default logs directory is:
./logs
```

Our model achieves the following performance on mini-ImageNet, tiered-ImageNet, CUB-200-2011 and CIFAR-FS:


| Dataset            |    Backbone     |   5way-1shot   |   5way-5shot   |
| ------------------ |---------------- | -------------- | -------------- |
| mini-ImageNet      |    ResNet12     |   67.77±0.32   |   84.60±0.43   |
| mini-ImageNet      |    ResNet18     |   66.63±0.51   |   84.07±0.42   |
| mini-ImageNet      |      WRN        |   67.24±0.51   |   83.72±0.44   |
| mini-ImageNet      |    ConvNet      |   66.01±0.36   |   82.83±0.41   |
| tiered-ImageNet    |    ResNet12     |   72.45±0.51   |   87.24±0.39   |
| tiered-ImageNet    |    ResNet18     |   70.46±0.52   |   86.44±0.41   |
| tiered-ImageNet    |    ConvNet      |   69.43±0.49   |   85.92±0.42   |
| CUB-200-2011       |    ResNet12     |   75.71±0.47   |   91.48±0.33   |
| CUB-200-2011       |    ConvNet      |   76.05±0.51   |   89.08±0.38   |
| CIFAR-FS           |    ResNet12     |   77.9±0.5     |    90.2±0.4    |
| CIFAR-FS           |    ConvNet      |   76.4±0.5     |    88.4±0.4    |