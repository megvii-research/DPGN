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
few-shot learning benchmarks, DPGN outperforms state-of-the-art results by a large margin in 5% ∼ 12% under supervised settings and 7% ∼ 13% under semi-supervised settings.

## Requirements

CUDA Version: 10.1

Python : 3.5.2

To install dependencies:

```setup
sudo pip3 install -r requirements.txt
```
## Dataset
For your convenience, you can download the datasets directly from links on the left, or you can make them from scratch following the original splits on the right. 

|    Dataset    | Original Split |
| :-----------: |:----------------:|
|  [Mini-ImageNet](https://drive.google.com/open?id=15WuREBvhEbSWo4fTr1r-vMY0C_6QWv4w)  |  [Matching Networks](https://arxiv.org/pdf/1606.04080.pdf)  | 
|    [Tiered-ImageNet](https://drive.google.com/file/d/1nVGCTd9ttULRXFezh4xILQ9lUkg0WZCG)   |   [SSL](https://arxiv.org/abs/1803.00676)   |
|  [CIFAR-FS](https://drive.google.com/file/d/1GjGMI0q3bgcpcB_CjI40fX54WgLPuTpS)  |   [R2D2](https://arxiv.org/pdf/1805.08136.pdf)   |
|      [CUB-200-2011](https://github.com/wyharveychen/CloserLookFewShot/tree/master/filelists/CUB)     |   [Closer Look](https://arxiv.org/pdf/1904.04232.pdf)   |


The dataset directory should look like this:
```bash
├── dataset
    ├── mini-imagenet
        ├── mini_imagenet_test.pickle   
        ├── mini_imagenet_train.pickle  
        ├── mini_imagenet_val.pickle
    ├── tiered-imagenet
        ├── class_names.txt   
        ├── synsets.txt  
        ├── test_images.npz
        ├── test_labels.pkl   
        ├── train_images.npz  
        ├── train_labels.pkl
        ├── val_images.npz
        ├── val_labels.pkl
    ├── cifar-fs
        ├── cifar_fs_test.pickle   
        ├── cifar_fs_train.pickle  
        ├── cifar_fs_val.pickle
    ├── cub-200-2011
        ├── attributes   
        ├── bounding_boxes.txt 
        ├── classes.txt
        ├── image   
        ├── image_class_labels.txt 
        ├── images
        ├── images.txt   
        ├── parts
        ├── README
        ├── split
        ├── train_test_split.txt
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

Our model achieves the following performance on mini-ImageNet, tiered-ImageNet, CUB-200-2011 and CIFAR-FS (more detailed experimental results are in the paper).

**miniImageNet**:

|     Method    |   Backbone   |   5way-1shot   |   5way-5shot   |
| :-----------: |:------------:|----------------|:--------------:|
|  MatchingNet  |    ConvNet   |   43.56±0.84   |   55.31± 0.73  |
|    ProtoNet   |    ConvNet   |   49.42±0.78   |   68.20±0.66   |
|  RelationNet  |    ConvNet   |   50.44±0.82   |   65.32±0.70   |
|      MAML     |    ConvNet   |   48.70±1.84   |   55.31±0.73   |
|      GNN      |    ConvNet   |   50.33±0.36   |   66.41±0.63   |
|      TPN      |    ConvNet   |   55.51±0.86   |   69.86±0.65   |
|   Edge-label  |    ConvNet   |   59.63±0.52   |   76.34±0.48   |
|    **DPGN**   |  **ConvNet** | **66.01±0.36** | **82.83±0.41** |
|      LEO      |      WRN     |   61.76±0.08   |   77.59±0.12   |
|      wDAE     |      WRN     |   61.07±0.15   |   76.75±0.11   |
|    **DPGN**   |    **WRN**   | **67.24±0.51** | **83.72±0.44** |
|   CloserLook  |   ResNet18   |   51.75±0.80   |   74.27±0.63   |
|      CTM      |   ResNet18   |   62.05±0.55   |   78.63±0.06   |
|    **DPGN**   | **ResNet18** | **66.63±0.51** | **84.07±0.42** |
|    MetaGAN    |   ResNet12   |   52.71±0.64   |   68.63±0.67   |
|     SNAIL     |   ResNet12   |   55.71±0.99   |   68.88±0.92   |
|     TADAM     |   ResNet12   |   58.50±0.30   |   76.70±0.30   |
|   Shot-Free   |   ResNet12   |   59.04±0.43   |   77.64±0.39   |
| Meta-Transfer |   ResNet12   |   61.20±1.80   |   75.53±0.80   |
|      FEAT     |   ResNet12   |   62.96±0.02   |   78.49±0.02   |
|   MetaOptNet  |   ResNet12   |   62.64±0.61   |   78.63±0.46   |
|    **DPGN**   | **ResNet12** | **67.77±0.32** | **84.60±0.43** |


**tieredImageNet**:

|     Method    |   backbone   |   5way-1shot   |   5way-5shot   |
| :-----------: |:------------:|----------------|:--------------:|
|      MAML     |    ConvNet   |   51.67±1.81   |   70.30±1.75   |
|    ProtoNet   |    ConvNet   |   53.34±0.89   |   72.69±0.74   |
|  RelationNet  |    ConvNet   |   54.48±0.93   |   71.32±0.78   |
|      TPN      |    ConvNet   |   59.91±0.94   |   73.30±0.75   |
|   Edge-label  |    ConvNet   |   63.52±0.52   |   80.24±0.49   |
|    **DPGN**   |  **ConvNet** | **69.43±0.49** | **85.92±0.42** |
|      CTM      |   ResNet18   |   64.78±0.11   |   81.05±0.52   |
|    **DPGN**   | **ResNet18** | **70.46±0.52** | **86.44±0.41** |
|     TapNet    |   ResNet12   |   63.08±0.15   |   80.26±0.12   |
| Meta-Transfer |   ResNet12   |   65.62±1.80   |   80.61±0.90   |
|   MetaOptNet  |   ResNet12   |   65.81±0.74   |   81.75±0.53   |
|   Shot-Free   |   ResNet12   |   66.87±0.43   |   82.64±0.39   |
|    **DPGN**   | **ResNet12** | **72.45±0.51** | **87.24±0.39** |


**CUB-200-2011**:

|    Method   |   backbone   | 5way-1shot     |   5way-5shot   |
|:-----------:|:------------:|----------------|:--------------:|
|   ProtoNet  |    ConvNet   | 51.31±0.91     |   70.77±0.69   |
|     MAML    |    ConvNet   | 55.92±0.95     |   72.09±0.76   |
| MatchingNet |    ConvNet   | 61.16±0.89     |   72.86±0.70   |
| RelationNet |    ConvNet   | 62.45±0.98     |   76.11±0.69   |
|  CloserLook |    ConvNet   | 60.53±0.83     |   79.34±0.61   |
|     DN4     |    ConvNet   | 53.15±0.84     |   81.90±0.60   |
|   **DPGN**  |  **ConvNet** | **76.05±0.51** | **89.08±0.38** |
|     FEAT    |   ResNet12   | 68.87±0.22     |   82.90±0.15   |
|   **DPGN**  | **ResNet12** | **75.71±0.47** | **91.48±0.33** |


**CIFAR-FS**:

|    Method   |   backbone   |  5way-1shot  |  5way-5shot  |
|:-----------:|:------------:|:------------:|:------------:|
|   ProtoNet  |    ConvNet   |   55.5±0.7   |   72.0±0.6   |
|     MAML    |    ConvNet   |   58.9±1.9   |   71.5±1.0   |
| RelationNet |    ConvNet   |   55.0±1.0   |   69.3±0.8   |
|     R2D2    |    ConvNet   |   65.3±0.2   |   79.4±0.1   |
|   **DPGN**  |  **ConvNet** | **76.4±0.5** | **88.4±0.4** |
|  Shot-Free  |   ResNet12   |   69.2±0.4   |   84.7±0.4   |
|  MetaOptNet |   ResNet12   |   72.0±0.7   |   84.2±0.5   |
|   **DPGN**  | **ResNet12** | **77.9±0.5** | **90.2±0.4** |
