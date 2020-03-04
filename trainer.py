
import os
import itertools
from skimage.io import imsave
import sys

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np

from utils import ImagePool, UnalignedImgMaskDataset
from model import InstaGAN
import losses

from pytorch_memlab import profile

class Trainer():
    def __init__(self, params):
        self.params = params
        common_params = params["common"]
        training_params = params["training"]

        self.max_epoch = training_params["max_epoch"]

        self.learning_rate = training_params["learning_rate"]
        self.beta1 = training_params["beta1"]

        self.X_name = training_params["X_name"]
        self.Y_name = training_params["Y_name"]
        self.image_dir = training_params["image_dir"]
        self.annotation_file = training_params["annotation_file"]
        self.ins_max = 8
        self.ins_per = 2
        self.checkpoint_root = training_params["checkpoint_root"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")

        self.lambda_X = training_params["lambda_X"]
        self.lambda_Y = training_params["lambda_Y"]
        self.lambda_idt = training_params["lambda_idt"]
        self.lambda_ctx = training_params["lambda_ctx"]

        pool_size = 50
        self.fake_X_pool = ImagePool(pool_size)
        self.fake_Y_pool = ImagePool(pool_size)

    def train(self):

        model = InstaGAN(self.params)
        model.cast_device(self.device)

        # construct dataloader
        train_dataset = UnalignedImgMaskDataset(X_name=self.X_name, Y_name=self.Y_name, 
                                                image_dir=self.image_dir, annotation_file=self.annotation_file)
        train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=2) 

        # construct optimizers
        optimizer_G = Adam(filter(lambda p: p.requires_grad, itertools.chain(model.G_XY.parameters(), model.G_YX.parameters())),
                           lr=self.learning_rate,
                           betas=(self.beta1, 0.999))
        optimizer_D_X = Adam(model.D_X.parameters(),
                            lr=self.learning_rate*5, 
                            betas=(self.beta1, 0.999))
        optimizer_D_Y = Adam(model.D_Y.parameters(),
                    lr=self.learning_rate*5, 
                    betas=(self.beta1, 0.999))
        # training roop

        criterionGAN = losses.LSGAN()
        criterionCyc = torch.nn.L1Loss()
        criterionIdt = torch.nn.L1Loss()
        criterionCtx = losses.WeightedL1Loss()
        ins_iter = self.ins_max // self.ins_per

        result_dir = "./result/test/"
        os.makedirs(result_dir, exist_ok=True)
        for epoch in range(1, self.max_epoch + 1):
            print(f"epoch {epoch} start")
            for batch in train_dataloader:
                #batch = train_dataset.getitem(idx)
                real_image_mask_X, real_image_mask_Y= self.cast_device(batch)
                real_image_X = real_image_mask_X[:, :3, :, :]
                real_image_Y = real_image_mask_Y[:, :3, :, :]
                real_masks_X = real_image_mask_X[:, 3:, :, :]
                real_masks_Y = real_image_mask_Y[:, 3:, :, :]

                fake_mask_X_list = []
                fake_mask_Y_list = []
                rec_mask_X_list = []
                rec_mask_Y_list = []

                
                for i in range(ins_iter):
                    #i 番目のインスタンスセットに関する変換
                    masks_X = real_masks_X[:, self.ins_per*i:self.ins_per*(i+1), :, :]
                    masks_Y = real_masks_Y[:, self.ins_per*i:self.ins_per*(i+1), :, :]
                    #empty = - torch.ones(real_mask_X).to(self.device)

                    remain_instance_X = (masks_X.to("cpu").numpy()).sum() > 0
                    remain_instance_Y = (masks_Y.to("cpu").numpy()).sum() > 0

                    if (not remain_instance_X) and (not remain_instance_Y):
                        continue

                    if remain_instance_X:
                        temp_real_image_mask_X = torch.cat([real_image_X, masks_X], dim=1)
                        fake_image_mask_Y = model.G_XY(temp_real_image_mask_X)
                        fake_image_Y = fake_image_mask_Y[:, :3, :, :]
                        fake_masks_Y = fake_image_mask_Y[:, 3:, :, :]
                        rec_image_mask_X = model.G_YX(fake_image_mask_Y)

                    if remain_instance_Y:
                        temp_real_image_mask_Y = torch.cat([real_image_Y, masks_Y], dim=1)
                        fake_image_mask_X = model.G_YX(temp_real_image_mask_Y)
                        fake_image_X = fake_image_mask_X[:, :3, :, :]
                        fake_masks_X = fake_image_mask_X[:, 3:, :, :]
                        rec_image_mask_Y = model.G_XY(fake_image_mask_X)

                    
                    # Update Generators
                    optimizer_G.zero_grad()

                    if remain_instance_X:
                        loss_G_XY = criterionGAN(model.D_Y(torch.cat([fake_image_mask_Y, *fake_mask_Y_list], dim=1)), is_real=True)
                        loss_cyc_XYX = criterionCyc(rec_image_mask_X, temp_real_image_mask_X)
                        loss_idt_X = criterionIdt(model.G_YX(temp_real_image_mask_X), temp_real_image_mask_X) #★Idt lossの計算式見直す
                        weight_X = self.get_weight_for_ctx(masks_X, fake_masks_Y) 
                        loss_ctx_XY = criterionCtx(real_image_X, fake_image_Y, weight=weight_X)
                    else:
                        loss_G_XY = 0
                        loss_cyc_XYX = 0
                        loss_idt_X = 0
                        loss_ctx_XY = 0

                    if remain_instance_Y:
                        loss_G_YX = criterionGAN(model.D_X(torch.cat([fake_image_mask_X, *fake_mask_X_list], dim=1)), is_real=True)
                        loss_cyc_YXY = criterionCyc(rec_image_mask_Y, temp_real_image_mask_Y)
                        loss_idt_Y = criterionIdt(model.G_XY(temp_real_image_mask_Y), temp_real_image_mask_Y)
                        weight_Y = self.get_weight_for_ctx(masks_Y, fake_masks_X)
                        loss_ctx_YX = criterionCtx(real_image_Y, fake_image_X, weight=weight_Y)
                    else:
                        loss_G_YX = 0
                        loss_cyc_YXY = 0
                        loss_idt_Y = 0
                        loss_ctx_YX = 0

                    if remain_instance_X or remain_instance_Y:
                        loss_G = loss_G_XY + loss_G_YX + \
                            self.lambda_X * (loss_cyc_XYX + self.lambda_ctx * loss_ctx_XY + self.lambda_idt * loss_idt_Y) + \
                            self.lambda_Y * (loss_cyc_YXY + self.lambda_ctx * loss_ctx_YX + self.lambda_idt * loss_idt_X)

                        loss_G.backward()

                        optimizer_G.step()
                        loss_G = loss_G.item()

                                       
                    # Update Discriminators

                    fake_image_X = fake_image_X.detach()
                    fake_image_Y = fake_image_Y.detach()
                    fake_masks_X = fake_masks_X.detach()
                    fake_masks_Y = fake_masks_Y.detach()

                    fake_mask_X_list.append(fake_masks_X)
                    fake_mask_Y_list.append(fake_masks_Y)

                    fake_image_mask_X = torch.cat([fake_image_X, *fake_mask_X_list], dim=1)
                    fake_image_mask_Y = torch.cat([fake_image_Y, *fake_mask_Y_list], dim=1)

                    #rec_image_masks_X = rec_image_mask_X.detach()
                    #rec_image_masks_Y = rec_image_mask_Y.detach()
                    
                    #fake_image_mask_X = fake_image_mask_X.to("cpu")
                    #fake_image_mask_Y = fake_image_mask_Y.to("cpu")

                    fake_image_mask_X_D = self.fake_X_pool.query(fake_image_mask_X)
                    fake_image_mask_Y_D = self.fake_Y_pool.query(fake_image_mask_Y)

                    #fake_image_mask_X = fake_image_mask_X.to(self.device)
                    #fake_image_mask_Y = fake_image_mask_Y.to(self.device)
                    
                    if remain_instance_Y:
                        optimizer_D_X.zero_grad()
                        loss_D_X = 0.5 * (criterionGAN(model.D_X(temp_real_image_mask_X), is_real=True) + criterionGAN(model.D_X(fake_image_mask_X_D), is_real=False))
                        loss_D_X.backward()
                        optimizer_D_X.step()
                        loss_D_X = loss_D_X.item()
                    
                    if remain_instance_X:
                        optimizer_D_Y.zero_grad()
                        loss_D_Y = 0.5 * (criterionGAN(model.D_Y(temp_real_image_mask_Y), is_real=True) + criterionGAN(model.D_Y(fake_image_mask_Y_D), is_real=False))
                        loss_D_Y.backward()
                        optimizer_D_Y.step()
                        loss_D_Y = loss_D_Y.item()

                    print(f"loss_D_X: {loss_D_X:.4f}, loss_D_Y: {loss_D_Y:.4f}, loss_G: {loss_G:.4f}")

                    # detach all images and masks
                    real_images_X = fake_image_X
                    real_images_Y = fake_image_Y
                    #rec_mask_X_list.append(rec_masks_X)
                    #rec_mask_Y_list.append(rec_masks_Y)
            
            
            real_images_X = real_image_mask_X[:, :3, :, :].to("cpu").numpy()
            real_images_Y = real_image_mask_Y[:, :3, :, :].to("cpu").numpy()
            fake_image_X = fake_image_X.to("cpu").numpy()
            fake_image_Y = fake_image_Y.to("cpu").numpy()
            h, w = real_images_X.shape[2:]
            result_image = np.zeros((3, 2*h, 4*w))
            result_image[:, :h, :w] = real_images_X[0, :, :, :]
            result_image[:, :h, w:2*w] = fake_image_Y[0, :, :, :]
            #result_image[:, :h, 2*w:] = rec_image_X[0, :, :, :]
            result_image[:, h:2*h, :w] = real_images_Y[0, :, :, :]
            result_image[:, h:2*h, w:2*w] = fake_image_X[0, :, :, :]
            #result_image[:, h*2*h, 2*w:3*w] = rec_image_Y[0, :, :, :]

            result_image = (result_image + 1) * 127.5
            real_mask_X = real_image_mask_X[:, 3:, :, :].sum(dim=1).to("cpu").numpy() * 255
            real_mask_Y = real_image_mask_Y[:, 3:, :, :].sum(dim=1).to("cpu").numpy() * 255
            fake_mask_X = torch.cat([*fake_mask_X_list], dim=1)
            fake_mask_Y = torch.cat([*fake_mask_Y_list], dim=1)
            fake_mask_X = fake_mask_X.sum(dim=1).to("cpu").numpy() * 255
            fake_mask_Y = fake_mask_Y.sum(dim=1).to("cpu").numpy() * 255
            
            result_image[0, :h, 2*w:3*w] = real_mask_X
            result_image[0, :h, 3*w:4*w] = fake_mask_Y
            result_image[0, h:2*h, 2*w:3*w] = real_mask_Y
            result_image[0, h:2*h, 3*w:4*w] = fake_mask_X

            result_image = np.transpose(result_image, (1, 2, 0))
            result_image = result_image.astype(np.uint8)
            sample_image_file = f"{result_dir}/{epoch}.png"
            imsave(sample_image_file, result_image)
            torch.save(model.G_XY.state_dict(), f"{result_dir}/{epoch}_{self.X_name}2{self.Y_name}.pth")
            torch.save(model.G_YX.state_dict(), f"{result_dir}/{epoch}_{self.Y_name}2{self.X_name}.pth")


    def get_weight_for_ctx(self, x, y):
        z = torch.cat([x, y], dim=1)
        z = torch.sum(z, dim=1, keepdim=True)
        z = z.clamp(max=1, min=0)
        return 1 - z  # [0,1] -> [1,0]
            
    def cast_device(self, batch):
        image_mask_X = batch[0]
        image_mask_Y = batch[1]
        image_mask_X = image_mask_X.to(self.device)
        image_mask_Y = image_mask_Y.to(self.device)
        return image_mask_X, image_mask_Y


