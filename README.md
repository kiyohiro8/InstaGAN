# Pytorch-InstaGAN
Pytorch implementation of InstaGAN (https://openreview.net/pdf?id=ryxwJhC9YX)

![](https://github.com/kiyohiro8/InstaGAN/blob/master/samples/zebra2giraffe.png)

## Libraries

### Assumed
- PyTorch==1.3.1

### Required
- pycocotools (https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools)

## Dataset preparation
Learning InstaGAN requires annotation information on a per-instance basis. In this repository, The format of the annotations should follow the COCO dataset.

### Download COCO dataset

To get COCO dataset, run get_dataset.py
CAUTION: By this script, you will download a large amount of image data (> 18GB).

```
python get_dataset.py
```



## Train

run train.py 

```
python train.py params.yaml
```

## Train from checkpoint

```
python resume_train.py [path to checkpoint directory]
```
