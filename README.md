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
$ python get_dataset.py
```

After execution, a directory with the following structure will be created.

```
├── data
│   ├── instances_train2017.json
│   └── train2017
           ├── 000000000009.jpg
           ├── 000000000025.jpg
           ├── ...
```

## Train

run train.py 

```
$ python train.py params.yaml
```

After execution, a directory with the following structure will be created to store the learning results.

```
├── result
│   └── yymmdd_HHMM_[domain X]2[domain Y]
           ├── params.json
           ├── weights
           └── samples
```

## Train from checkpoint

```
$ python resume_train.py [path to checkpoint directory]
```
