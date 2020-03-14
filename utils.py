# -*- coding: utf-8 -*-

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
    def __init__(self, X_name, Y_name, image_dir, annotation_file, simple_resize=True, test_size=10):
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
        # transformsの定義
        if simple_resize:
            self.transforms = [SimpleResize(image_size=200),
                               ToTensor()]
        else:
            self.transforms = [ResizeRandomCrip(image_size=300),
                            RandomFlip(),
                            ToTensor()]
        self.image_dir = image_dir
        self.max_instances = 8
        self.image_size = 200
    
    def __getitem__(self, idx):
        annotation_X = self.coco.loadImgs(self.Ids_X[idx])[0]
        annotation_Y = self.coco.loadImgs(self.Ids_Y[idx])[0]
        image_X = imread(os.path.join(self.image_dir, annotation_X["file_name"]))
        image_Y = imread(os.path.join(self.image_dir, annotation_Y["file_name"]))
        #image_X = np.transpose(image_X, (2, 0, 1))
        #image_Y = np.transpose(image_Y, (2, 0, 1))

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
        """
        if len(self.transforms) > 0:
            for transform in self.transforms:
                image_X, masks_X = transform(image_X, masks_X)
                image_Y, masks_Y = transform(image_Y, masks_Y)
        """
        image_X = resize(image_X, output_shape=(200, 200, 3))
        masks_X = [resize(mask, output_shape=(200, 200), anti_aliasing=False, preserve_range=True) for mask in masks_X]
        image_Y = resize(image_Y, output_shape=(200, 200, 3))
        masks_Y = [resize(mask, output_shape=(200, 200), anti_aliasing=False, preserve_range=True) for mask in masks_Y]

        image_X = (torch.FloatTensor(image_X.copy()) - 0.5) * 2
        masks_X = [torch.FloatTensor(mask.copy()) for mask in masks_X]
        image_Y = (torch.FloatTensor(image_Y.copy()) - 0.5) * 2
        masks_Y = [torch.FloatTensor(mask.copy()) for mask in masks_Y]

        image_X = np.transpose(image_X, (2, 0, 1))
        image_Y = np.transpose(image_Y, (2, 0, 1))

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
        
        h, w, c = image.shape


        image = resize(image, output_shape=(200, 200, 3))
        masks = [resize(mask, output_shape=(200, 200), anti_aliasing=False, preserve_range=True) for mask in masks]

            
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
        masks = [resize(mask, output_shape=(self.image_size, self.image_size, 1)) for mask in masks]
        for mask in masks:
            print(mask.sum())
        return image, masks
    

class ResizeRandomCrip(object):
    def __init__(self, image_size):
        self.image_size = image_size

    def __call__(self, image, masks):

        c, h, w = image.shape
        areas = [np.sum(mask) for mask in masks]
        if h < w:
            output_shape = (3, self.image_size, int(w * self.image_size / h))
        else:
            output_shape = (3, int(h * self.image_size / w), self.image_size)
        image = resize(image, output_shape=output_shape)
        masks = [resize(mask, output_shape=(1,)+output_shape[1:]) for mask in masks]
        area_ratio = [image.shape[1] * image.shape[2] / h / w * area for area in areas]

        c, h, w = image.shape


        mask_list = []
        while not mask_list:
            if h != self.image_size:
                th = np.random.randint(0, h - self.image_size)
            else:
                th = 0
            if w != self.image_size:
                tw = np.random.randint(0, w - self.image_size)
            else:
                tw = 0

            temp_image = image[:, th:th+self.image_size, tw:tw+self.image_size]
            temp_masks = [mask[:, th:th+self.image_size, tw:tw+self.image_size] for mask in masks]

            for i, mask in enumerate(temp_masks):
                if np.sum(mask) < area_ratio[i] / 3:
                    mask_list.append(mask)

        return temp_image, mask_list


class RandomFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, masks):
        if self.p < np.random.rand():
            return image, masks
        else:
            image = np.flip(image, 2)
            masks = [np.flip(mask, 1) for mask in masks]
            return image, masks

class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, image, masks):
        image = torch.FloatTensor(image.copy())
        masks = [torch.FloatTensor(mask.copy()) for mask in masks]
        for mask in masks:
            print(mask.sum())
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
