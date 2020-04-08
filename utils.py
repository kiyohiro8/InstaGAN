
import os
from glob import glob
import random
import time
import datetime
import sys

from torch.autograd import Variable
import torch
import numpy as np
import pandas as pd
from skimage.transform import resize
from skimage.io import imread, imsave

from pycocotools.coco import COCO
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class UnalignedImgMaskDataset(object):
    def __init__(self, X_name, Y_name, image_dir, annotation_file, image_size, max_instances, simple_resize=True, test_size=20):
        # annotationsの読み込み
        # imageへのpathとマスク情報を格納した辞書のリストを作成
        self.coco = COCO(annotation_file)

        self.category_ids_X = self.coco.getCatIds(catNms=[X_name])
        self.category_ids_Y = self.coco.getCatIds(catNms=[Y_name])
        Ids_X = self.coco.getImgIds(catIds=self.category_ids_X)
        Ids_Y = self.coco.getImgIds(catIds=self.category_ids_Y)
        intersection = set(Ids_X) & set(Ids_Y)
        Ids_X = list(set(Ids_X) - intersection)
        Ids_Y = list(set(Ids_Y) - intersection)

        self.Ids_X = Ids_X[test_size:]
        self.Ids_Y = Ids_Y[test_size:]
        self.test_size = test_size
        if self.test_size <= len(self.Ids_X) // 10 and test_size <= len(self.Ids_Y) // 10:
            self.test_Ids_X = Ids_X[:test_size]
            self.test_Ids_Y = Ids_Y[:test_size]
        else:
            self.test_Ids_X = []
            self.test_Ids_Y = []
            print("No test images.")

        print(f"{len(self.Ids_X)} images for {X_name}.")
        print(f"{len(self.Ids_Y)} images for {Y_name}.")

        self.image_dir = image_dir
        self.max_instances = max_instances
        self.image_size = image_size

        # transformsの定義
        if simple_resize:
            self.spatial_transforms = [SimpleResize(image_size=self.image_size),
                               ToTensor()]
        else:
            self.spatial_transforms = [ResizeRandomCrop(image_size=self.image_size),
                            RandomFlip(),
                            ToTensor()]

        self.color_transforms = transforms.Compose([transforms.ColorJitter(brightness=0.05, contrast=0.05)])

    
    def __getitem__(self, idx):
        annotation_X = self.coco.loadImgs(self.Ids_X[idx])[0]
        annotation_Y = self.coco.loadImgs(self.Ids_Y[idx])[0]
        image_X = Image.open(os.path.join(self.image_dir, annotation_X["file_name"]))
        image_Y = Image.open(os.path.join(self.image_dir, annotation_Y["file_name"]))

        image_id_X = annotation_X["id"]
        image_id_Y = annotation_Y["id"]

        annotation_ids_X = self.coco.getAnnIds(imgIds=image_id_X, catIds=self.category_ids_X)
        annotation_ids_Y = self.coco.getAnnIds(imgIds=image_id_Y, catIds=self.category_ids_Y)

        anns_X = self.coco.loadAnns(annotation_ids_X)
        anns_Y = self.coco.loadAnns(annotation_ids_Y)

        masks_X = [self.coco.annToMask(instance) for instance in anns_X]
        random.shuffle(masks_X)
        if len(masks_X) > self.max_instances:
            masks_X = masks_X[:self.max_instances]
        masks_Y = [self.coco.annToMask(instance) for instance in anns_Y]
        random.shuffle(masks_Y)
        if len(masks_Y) > self.max_instances:
            masks_Y = masks_Y[:self.max_instances]

        image_X = np.asarray(self.color_transforms(image_X))
        image_Y = np.asarray(self.color_transforms(image_Y))

        if len(self.spatial_transforms) > 0:
            for transform in self.spatial_transforms:
                image_X, masks_X = transform(image_X, masks_X)
                image_Y, masks_Y = transform(image_Y, masks_Y)

        mask_tensor_X = torch.zeros((self.max_instances, self.image_size, self.image_size))
        mask_tensor_Y = torch.zeros((self.max_instances, self.image_size, self.image_size))

        for i, mask in enumerate(masks_X):
            mask_tensor_X[i, :, :] = mask
        
        for i, mask in enumerate(masks_Y):
            mask_tensor_Y[i, :, :] = mask

        image_mask_X = torch.cat([image_X, mask_tensor_X], axis=0)
        image_mask_Y = torch.cat([image_Y, mask_tensor_Y], axis=0)

        return image_mask_X, image_mask_Y

    def __len__(self):
        return min(len(self.Ids_X), len(self.Ids_Y))
    
    def shuffle(self):
        random.shuffle(self.Ids_X)
        random.shuffle(self.Ids_Y)

    def get_test_data(self, idx, domain):
        assert domain in ["X", "Y"]

        if domain == "X":
            annotation = self.coco.loadImgs(self.test_Ids_X[idx])[0]
        else:
            annotation = self.coco.loadImgs(self.test_Ids_Y[idx])[0]
        
        image = imread(os.path.join(self.image_dir, annotation["file_name"]))
        image_id = annotation["id"]
        
        if domain == "X":
            annotation_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.category_ids_X)
        else:
            annotation_ids = self.coco.getAnnIds(imgIds=image_id, catIds=self.category_ids_Y)

        anns = self.coco.loadAnns(annotation_ids)
        masks = [self.coco.annToMask(instance) for instance in anns]


        image = resize(image, output_shape=(self.image_size, self.image_size, 3))
        masks = [resize(mask, output_shape=(self.image_size, self.image_size), anti_aliasing=False, preserve_range=True) for mask in masks]

            
        image = (torch.FloatTensor(image.copy()) - 0.5) * 2
        masks = [torch.FloatTensor(mask.copy()) for mask in masks]

        image = np.transpose(image, (2, 0, 1))
        masks = torch.stack(masks)

        image_mask = torch.cat([image, masks], axis=0)
        return image_mask.unsqueeze(0)

        
class SimpleResize(object):
    def __init__(self, image_size):
        self.image_size = image_size
    
    def __call__(self, image, masks):
        image = resize(image, output_shape=(self.image_size, self.image_size, 3))
        masks = [resize(mask, output_shape=(self.image_size, self.image_size), anti_aliasing=False, preserve_range=True) for mask in masks]
        return image, masks
    

class ResizeRandomCrop(object):
    def __init__(self, image_size):
        self.image_size = image_size

    def __call__(self, image, masks):

        image = resize(image, output_shape=(int(self.image_size*1.2), int(self.image_size*1.2), 3))
        masks = [resize(mask, output_shape=(int(self.image_size*1.2), int(self.image_size*1.2)), anti_aliasing=False, preserve_range=True) for mask in masks]

        th = np.random.randint(0, 1.2*self.image_size - self.image_size)
        tw = np.random.randint(0, 1.2*self.image_size - self.image_size)

        image = image[th:th+self.image_size, tw:tw+self.image_size, :]
        masks = [mask[th:th+self.image_size, tw:tw+self.image_size] for mask in masks]


        return image, masks


class RandomFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, masks):
        if self.p < np.random.rand():
            return image, masks
        else:
            image = np.flip(image, 1)
            masks = [np.flip(mask, 1) for mask in masks]
            return image, masks

class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, image, masks):
        image = np.transpose(image, (2, 0, 1))
        image = (torch.FloatTensor(image.copy()) - 0.5) *2
        masks = [torch.FloatTensor(mask.copy()) for mask in masks]
        return image, masks

class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, image_masks):
        if self.pool_size == 0:
            self.images.append(image_masks)
            return image_masks
        return_images = []
        for i in range(image_masks.size(0)):
            image_mask = torch.unsqueeze(image_masks[i, :, :, :], 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image_mask)
                return_images.append(image_mask)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images.pop(random_id)
                    self.images.append(image_mask)
                    return_images.append(tmp)
                else:
                    return_images.append(image_mask)
        return_images = torch.cat(return_images, dim=0)
        return return_images
    
    def __len__(self):
        return len(self.images)
