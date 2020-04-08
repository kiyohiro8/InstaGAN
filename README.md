# Pytorch-InstaGAN
Pytorch implementation of InstaGAN (https://openreview.net/pdf?id=ryxwJhC9YX)

![](https://github.com/kiyohiro8/InstaGAN/blob/master/samples/zebra2giraffe.png)

## Libraries

### Assumed
- PyTorch==1.3.1

### Required
- pycocotools (https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools)

## Dataset Preparation
Learning InstaGAN requires annotation information on a per-instance basis. In this repository, The format of the annotations should follow the COCO dataset.

## Train

```
python train.py [path to parameter.yaml]
```

## Train from checkpoint

```
python resume_train.py [path to checkpoint directory]
```
